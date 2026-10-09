import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from deeplab.core.trainer import Trainer


class CheckpointLoadingTest(unittest.TestCase):
    def make_trainer(self, output_dir):
        # Exercise the real checkpoint loader and fit loop with a tiny CPU model.
        trainer = Trainer.__new__(Trainer)
        trainer.device = torch.device("cpu")
        trainer.is_distributed = False
        trainer.modules = {"spk_model": torch.nn.Linear(2, 2)}
        trainer.init_epoch_idx = 0
        trainer.init_logs_list = []
        trainer.init_optimizer_state = None
        trainer.init_scheduler_states = {}
        trainer.init_amp_scaler_state = None
        trainer.init_rng_state = None
        trainer.hparams = {"num_epochs": 4, "output_dir": str(output_dir), "use_amp": False}
        trainer.yaml_strings = "num_epochs: 4\n"
        trainer.exps_tag = "test_"
        trainer.use_wandb = False
        trainer.optimizer = torch.optim.SGD(trainer.modules["spk_model"].parameters(), lr=0.1)
        trainer.scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, step_size=1)
        trainer.scheduler_lmft = None
        trainer.initialize_training = mock.Mock()
        trainer.train_one_epoch = mock.Mock(side_effect=lambda epoch, *_: {"epoch_idx": epoch})
        return trainer

    def test_pretrain_epoch_15_runs_all_four_new_stage_epochs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer = self.make_trainer(root / "output")
            weights = {name: torch.ones_like(value) for name, value in
                       trainer.modules["spk_model"].state_dict().items()}
            checkpoint = root / "pretrain.pth"
            torch.save({"modules": {"spk_model": weights}, "epoch_idx": 15,
                        "optimizer": {"unused": True}, "scheduler": {"unused": True}}, checkpoint)
            (root / "logs.json").write_text(json.dumps([{"epoch_idx": i} for i in range(1, 16)]))
            trainer.load_checkpoints(checkpoint)
            self.assertEqual(trainer.init_epoch_idx, 0)
            self.assertEqual(trainer.init_logs_list, [])
            self.assertIsNone(trainer.init_optimizer_state)
            self.assertEqual(trainer.init_scheduler_states, {})
            for name, value in trainer.modules["spk_model"].state_dict().items():
                torch.testing.assert_close(value, weights[name])
            trainer.fit()
            self.assertEqual([call.args[0] for call in trainer.train_one_epoch.call_args_list],
                             [1, 2, 3, 4])
            self.assertTrue(Path(trainer.ckpts_dir, "ckpt_0004.pth").is_file())
            self.assertEqual(len(json.loads(Path(trainer.ckpts_dir, "logs.json").read_text())), 4)

    def test_pretrain_accepts_weights_without_epoch_metadata(self):
        # Weights-only checkpoints need neither an epoch counter nor logs.json.
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(directory)
            checkpoint = Path(directory, "weights.pth")
            torch.save({"modules": {"spk_model": trainer.modules["spk_model"].state_dict()}}, checkpoint)
            trainer.load_checkpoints(checkpoint)
            self.assertEqual(trainer.init_epoch_idx, 0)

    def test_resume_preserves_progress_and_training_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer = self.make_trainer(root / "output")
            optimizer_state = trainer.optimizer.state_dict()
            optimizer_state["param_groups"][0]["lr"] = 0.025
            scheduler_state = trainer.scheduler.state_dict()
            scheduler_state["last_epoch"] = 2
            rng = trainer.get_rng_state()
            checkpoint = root / "resume.pth"
            torch.save({"modules": {"spk_model": trainer.modules["spk_model"].state_dict()},
                        "epoch_idx": 2, "optimizer": optimizer_state,
                        "scheduler": scheduler_state, "amp_scaler": {"scale": 128.0},
                        "rng": rng}, checkpoint)
            history = [{"epoch_idx": i} for i in (1, 2, 3)]
            (root / "logs.json").write_text(json.dumps(history))
            trainer.resume_checkpoints(checkpoint)
            self.assertEqual(trainer.init_epoch_idx, 2)
            self.assertEqual(trainer.init_logs_list, history[:2])
            self.assertEqual(trainer.init_optimizer_state, optimizer_state)
            self.assertEqual(trainer.init_scheduler_states, {"scheduler": scheduler_state})
            self.assertEqual(trainer.init_amp_scaler_state, {"scale": 128.0})
            torch.testing.assert_close(trainer.init_rng_state["torch"], rng["torch"])
            trainer.fit()
            self.assertEqual([call.args[0] for call in trainer.train_one_epoch.call_args_list], [3, 4])
            self.assertEqual(trainer.optimizer.param_groups[0]["lr"], 0.025)
            self.assertEqual(trainer.scheduler.last_epoch, 2)
            self.assertEqual(len(json.loads(Path(trainer.ckpts_dir, "logs.json").read_text())), 4)

    def test_strict_pretrain_module_rejects_partial_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(directory)
            trainer.hparams["strict_pretrain_modules"] = ["spk_model"]
            weights = dict(trainer.modules["spk_model"].state_dict())
            weights.pop("bias")
            checkpoint = Path(directory, "partial.pth")
            torch.save({"modules": {"spk_model": weights}}, checkpoint)
            with self.assertRaisesRegex(RuntimeError, "1/2 weights missing or mismatched"):
                trainer.load_checkpoints(checkpoint)
            # without the strict setting the old partial load still works
            trainer.hparams["strict_pretrain_modules"] = []
            trainer.load_checkpoints(checkpoint)

    def test_failed_checkpoint_write_leaves_no_file(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(directory)
            ckpt_path = Path(directory, "ckpt_0001.pth")

            def partial_save(obj, path):
                Path(path).write_bytes(b"truncated")
                raise OSError("Disk quota exceeded")

            with mock.patch("torch.save", side_effect=partial_save):
                with self.assertRaises(OSError):
                    trainer.save_checkpoint({"modules": {}}, str(ckpt_path))
            self.assertEqual(list(Path(directory).iterdir()), [])
            trainer.save_checkpoint({"modules": {}}, str(ckpt_path))
            self.assertEqual(list(Path(directory).iterdir()), [ckpt_path])


if __name__ == "__main__":
    unittest.main()

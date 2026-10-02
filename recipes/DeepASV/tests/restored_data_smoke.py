"""Check restored VoxCeleb data through the production CPU dataloaders."""

import argparse
import json
from itertools import islice
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from deeplab.utils.misc import seed_worker
from local.dataset import Valid_Dataset
from local.sampler import WavBatchSampler
from local.tar_dataset import TarTrainDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--result-json", required=True)
    args = parser.parse_args()
    # Read data settings without constructing the model or executing YAML tags.
    config = yaml.load(Path(args.config).read_text(), Loader=yaml.BaseLoader)
    hparams = {
        "training_loop": int(config["training_loop"]),
        "sample_rate": int(config["sample_rate"]),
        "max_valid_dur": float(config["max_valid_dur"]),
        "speed_perturbation": [float(x) for x in config["speed_perturbation"]],
        "data_aug": config["data_aug"].lower() == "true",
        "tar_max_open_shards": int(config["tar_max_open_shards"]),
        **{key: config[key] for key in (
            "train_tar_index", "musan_path", "rirs_path", "valid_data"
        )},
    }
    torch.set_num_threads(1)
    torch.manual_seed(24)
    np.random.seed(24)

    validation = Valid_Dataset(hparams)
    assert len(validation) == 4708, len(validation)
    recordings = {row["reco"] for row in validation.scp_list}
    for trial in validation.trial_list:
        assert trial["utt1"] in recordings and trial["utt2"] in recordings
    for row in validation.scp_list:
        assert Path(row["wav_path"]).is_file(), row["wav_path"]
        assert "/wav/wav/" not in row["wav_path"], row["wav_path"]
    validated = 0
    for batch in DataLoader(validation, batch_size=1, num_workers=4):
        audio = batch["aud_inputs"]
        assert audio.numel() > 0 and torch.isfinite(audio).all()
        validated += 1
    assert validated == len(validation)
    print(f"Validated all {validated} VoxCeleb1 audio files and trial references", flush=True)

    training = TarTrainDataset(hparams)
    try:
        assert len(training.records) == 1_092_009, len(training.records)
        assert training.spk_num == 5994, training.spk_num
        assert len(training.shard_paths) == 779, len(training.shard_paths)
        _, first_indices = np.unique(training.records[:, 0], return_index=True)
        # Decode at least one real clip from every shard, without augmentation.
        training.data_aug = False
        shard_sampler = [[(int(index), 2.0)] for index in first_indices]
        decoded_shards = 0
        for batch in DataLoader(training, batch_sampler=shard_sampler, num_workers=4):
            audio = batch["aud_inputs"]
            assert tuple(audio.shape) == (1, 32000) and torch.isfinite(audio).all()
            decoded_shards += 1
        assert decoded_shards == 779
        print(f"Decoded audio from all {decoded_shards} VoxCeleb2 shards", flush=True)

        training.data_aug = hparams["data_aug"]
        sampler = WavBatchSampler(
            training, dur_range=[float(x) for x in config["dur_range"]],
            shuffle=True, batch_size=int(config["batch_size"]),
            drop_last=True, distributed=False,
        )
        # Bound the sampler itself so workers cannot prefetch an entire epoch.
        batches = list(islice(iter(sampler), 10))
        variants_seen = set()
        for batch in DataLoader(
            training, batch_sampler=batches, num_workers=4, worker_init_fn=seed_worker
        ):
            audio, labels = batch["aud_inputs"], batch["spk_labels"]
            assert audio.shape[0] == int(config["batch_size"])
            assert 32000 <= audio.shape[1] <= 48000
            assert torch.isfinite(audio).all()
            assert 0 <= int(labels.min()) <= int(labels.max()) < 17982
            variants_seen.update((labels // training.spk_num).tolist())
        assert variants_seen == {0, 1, 2}, variants_seen
        result = {
            "config": args.config, "validation_files_decoded": validated,
            "validation_trials": len(validation.trial_list),
            "training_records": len(training.records),
            "training_shards_decoded": decoded_shards,
            "augmented_training_batches": len(batches),
            "batch_size": int(config["batch_size"]), "workers": 4,
            "speed_variants_seen": sorted(variants_seen), "passed": True,
        }
        Path(args.result_json).write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    finally:
        training.close()


if __name__ == "__main__":
    main()

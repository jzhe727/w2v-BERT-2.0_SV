import tempfile
import unittest
from pathlib import Path

import torch

from local.prepare_voxceleb1 import prepare_protocol
from local.spk_classifier import ArcFace


class ArcFaceTest(unittest.TestCase):
    def test_cpu_forward_and_backward_are_finite(self):
        classifier = ArcFace(in_features=4, out_features=3)
        embeddings = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            requires_grad=True,
        )
        labels = torch.tensor([0, 1])

        logits = classifier(embeddings, labels)
        logits.sum().backward()

        self.assertEqual(logits.device.type, "cpu")
        self.assertEqual(logits.shape, (2, 3))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(embeddings.grad).all())

    def test_bf16_autocast_saturated_cosine_has_finite_gradients(self):
        # Under bf16 a cosine of ~0.999 rounds to 1.0; sqrt(1 - cos^2)'s backward is then inf.
        classifier = ArcFace(in_features=4, out_features=3, s=32, m=0.5)
        with torch.no_grad():
            classifier.weight.copy_(torch.eye(3, 4))
        embeddings = torch.tensor(
            [[1.0, 0.03, 0.0, 0.0], [0.02, 1.0, 0.0, 0.0]],
            requires_grad=True,
        )
        labels = torch.tensor([0, 1])

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            logits = classifier(embeddings, labels)
        torch.nn.functional.cross_entropy(logits, labels).backward()

        self.assertEqual(logits.dtype, torch.float32)
        self.assertTrue(torch.isfinite(embeddings.grad).all())
        self.assertTrue(torch.isfinite(classifier.weight.grad).all())


class PrepareVoxCeleb1Test(unittest.TestCase):
    def test_prepare_protocol_writes_only_referenced_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wav_root = root / "wav"
            first = "id1/video1/00001.wav"
            second = "id2/video2/00002.wav"
            for utterance in (first, second):
                path = wav_root / utterance
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"wav")
            trials = root / "source.txt"
            trials.write_text(f"0 {first} {second}\n1 {first} {first}\n")

            counts = prepare_protocol(wav_root, trials, root / "protocol")

            self.assertEqual(counts, (2, 2))
            self.assertEqual(
                (root / "protocol" / "trials").read_text(),
                f"0 {first} {second}\n1 {first} {first}\n",
            )
            scp_lines = (root / "protocol" / "wav.scp").read_text().splitlines()
            self.assertEqual([line.split("\t")[0] for line in scp_lines], [first, second])


if __name__ == "__main__":
    unittest.main()
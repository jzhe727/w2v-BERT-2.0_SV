import io
import json
import sqlite3
import struct
import tarfile
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from local.sampler import WavBatchSampler
from local.tar_dataset import TarTrainDataset
from local.tar_index import build_tar_index, pread_exact, read_member_at


def _wav_bytes(sample_count=800, sample_rate=16000):
    buffer = io.BytesIO()
    samples = [int(12000 * ((index % 20) / 10 - 1)) for index in range(sample_count)]
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return buffer.getvalue()


def _create_shard(path, sample_count=8):
    payloads = {}
    with tarfile.open(path, "w") as archive:
        for label in range(sample_count):
            key = f"id{label:05d}/video{label:05d}/00000"
            cls_data = str(label).encode("utf-8")
            audio_data = _wav_bytes(sample_count=800 + label)
            payloads[key] = audio_data
            for extension, data in (("cls", cls_data), ("m4a", audio_data)):
                member = tarfile.TarInfo(f"{key}.{extension}")
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
    return payloads


def _hparams(index_path, speed_perturbation=None):
    return {
        "training_loop": 1,
        "sample_rate": 16000,
        "speed_perturbation": speed_perturbation,
        "data_aug": False,
        "tar_max_open_shards": 2,
        "train_tar_index": str(index_path),
    }


def _distributed_worker(rank, world_size, init_path, index_path, result_dir):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        dataset = TarTrainDataset(_hparams(index_path))
        sampler = WavBatchSampler(
            dataset,
            dur_range=[0.01, 0.01],
            shuffle=True,
            batch_size=2,
            drop_last=False,
            distributed=True,
        )
        sampler.set_epoch(3)
        loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
        labels = []
        for batch in loader:
            labels.extend(batch["spk_labels"].tolist())
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(labels))
        dataset.close()
    finally:
        dist.destroy_process_group()


class TarIndexTest(unittest.TestCase):
    def test_exact_read_handles_partial_pread(self):
        chunks = [b"ab", b"c", b"def"]
        with mock.patch("local.tar_index.os.pread", side_effect=chunks) as pread:
            self.assertEqual(pread_exact(7, 6, 11), b"abcdef")
        self.assertEqual(
            pread.call_args_list,
            [mock.call(7, 6, 11), mock.call(7, 4, 13), mock.call(7, 3, 14)],
        )

    def test_offsets_read_exact_member_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            shard_path = Path(directory, "samples.tar")
            index_path = Path(directory, "samples.sqlite3")
            payloads = _create_shard(shard_path, sample_count=3)
            build_tar_index([shard_path], index_path)

            connection = sqlite3.connect(index_path)
            rows = connection.execute(
                """
                SELECT sample_key, path, offset, samples.size, speaker_label
                FROM samples JOIN shards USING(shard_id)
                ORDER BY sample_id
                """
            ).fetchall()
            connection.close()

            self.assertEqual(len(rows), 3)
            for key, path, offset, size, label in rows:
                self.assertEqual(read_member_at(path, offset, size), payloads[key])
                self.assertEqual(label, int(key[2:7]))

    def test_incomplete_sample_is_rejected_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            shard_path = Path(directory, "incomplete.tar")
            index_path = Path(directory, "samples.sqlite3")
            with tarfile.open(shard_path, "w") as archive:
                data = b"0"
                member = tarfile.TarInfo("id00000/video/00000.cls")
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))

            with self.assertRaisesRegex(ValueError, "Incomplete sample"):
                build_tar_index([shard_path], index_path)
            self.assertFalse(index_path.exists())


class TarTrainDatasetTest(unittest.TestCase):
    def _indexed_dataset(self, directory, sample_count=8, speed_perturbation=None):
        shard_path = Path(directory, "samples.tar")
        index_path = Path(directory, "samples.sqlite3")
        _create_shard(shard_path, sample_count=sample_count)
        build_tar_index([shard_path], index_path)
        return TarTrainDataset(_hparams(index_path, speed_perturbation)), index_path

    def test_decode_crop_and_speed_label_expansion(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, _ = self._indexed_dataset(
                directory, sample_count=3, speed_perturbation=[0.9, 1.1]
            )
            clean = dataset[(0, 0.02)]
            with mock.patch(
                "local.tar_dataset.speed_augmentation", side_effect=lambda signal, *_: signal
            ) as augment:
                shifted = dataset[(len(dataset.records), 0.02)]

            self.assertEqual(len(dataset), 9)
            self.assertEqual(clean["aud_inputs"].shape, torch.Size([320]))
            self.assertEqual(clean["spk_labels"].item(), 0)
            self.assertEqual(shifted["spk_labels"].item(), 3)
            augment.assert_called_once()
            dataset.close()

    def test_multiple_dataloader_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, _ = self._indexed_dataset(directory)
            sampler = WavBatchSampler(
                dataset,
                dur_range=[0.01, 0.01],
                shuffle=False,
                batch_size=2,
                drop_last=False,
                distributed=False,
            )
            loader = DataLoader(dataset, batch_sampler=sampler, num_workers=2)
            labels = []
            shapes = []
            for batch in loader:
                labels.extend(batch["spk_labels"].tolist())
                shapes.append(tuple(batch["aud_inputs"].shape))

            self.assertEqual(labels, list(range(8)))
            self.assertEqual(shapes, [(2, 160)] * 4)
            dataset.close()

    def test_distributed_sampler_epoch_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, _ = self._indexed_dataset(directory)
            dist.init_process_group(
                "gloo",
                init_method=f"file://{Path(directory, 'epoch-init')}",
                rank=0,
                world_size=1,
            )
            try:
                sampler = WavBatchSampler(
                    dataset,
                    dur_range=[0.01, 0.01],
                    shuffle=True,
                    batch_size=2,
                    drop_last=False,
                    distributed=True,
                )

                sampler.set_epoch(4)
                first = [idx for batch in sampler for idx, _ in batch]
                sampler.set_epoch(4)
                replayed = [idx for batch in sampler for idx, _ in batch]
                sampler.set_epoch(5)
                next_epoch = [idx for batch in sampler for idx, _ in batch]
            finally:
                dist.destroy_process_group()

            self.assertEqual(first, replayed)
            self.assertNotEqual(first, next_epoch)
            self.assertEqual(set(first), set(range(8)))
            dataset.close()

    def test_two_cpu_distributed_loaders_are_disjoint_and_exhaustive(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, index_path = self._indexed_dataset(directory)
            dataset.close()
            init_path = Path(directory, "distributed-init")
            result_dir = Path(directory, "results")
            result_dir.mkdir()

            mp.spawn(
                _distributed_worker,
                args=(2, str(init_path), str(index_path), str(result_dir)),
                nprocs=2,
                join=True,
            )

            rank_labels = [
                json.loads(Path(result_dir, f"rank-{rank}.json").read_text())
                for rank in range(2)
            ]
            self.assertEqual(set(rank_labels[0]) & set(rank_labels[1]), set())
            self.assertEqual(set(rank_labels[0]) | set(rank_labels[1]), set(range(8)))
            self.assertEqual([len(labels) for labels in rank_labels], [4, 4])


if __name__ == "__main__":
    unittest.main()
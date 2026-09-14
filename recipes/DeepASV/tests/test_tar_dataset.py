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

import av
import numpy as np
import torch
from scipy.signal import fftconvolve
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from deeplab.dataio.audio import add_noise_from_musan_dict
from deeplab.utils.corpus import build_musan_tar_index, load_musan_dict
from deeplab.utils.fileio import TarAudioSource, load_audio
from local.sampler import WavBatchSampler
from local.tar_dataset import TarTrainDataset, decode_m4a_bytes
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


def _create_musan_tar(path):
    files = {
        "musan/noise/free-sound/noise.wav": _wav_bytes(500),
        "musan/music/fma/music-clean.wav": _wav_bytes(501),
        "musan/music/fma/music-vocal.wav": _wav_bytes(502),
        "musan/speech/librivox/speech.wav": _wav_bytes(503),
        "musan/music/fma/ANNOTATIONS": (
            b"music-clean genre N artist\nmusic-vocal genre Y artist\n"
        ),
    }
    with tarfile.open(path, "w") as archive:
        for name, payload in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    return files


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


class _MusanAugDataset(torch.utils.data.Dataset):
    def __init__(self, tar_path):
        self.path_dict = load_musan_dict(tar_path)

    def __len__(self):
        return 4

    def __getitem__(self, index):
        signal = np.zeros(320, dtype=np.float32)
        augmented = add_noise_from_musan_dict(
            signal, 16000, self.path_dict, prob=1.0, snr=[5, 20]
        )
        return torch.from_numpy(augmented).float()


class TarIndexTest(unittest.TestCase):
    def test_musan_index_selects_categories_and_excludes_vocals(self):
        with tempfile.TemporaryDirectory() as directory:
            tar_path = Path(directory, "musan.tar")
            index_path = Path(directory, "musan.sqlite3")
            _create_musan_tar(tar_path)

            build_musan_tar_index(tar_path, index_path)
            path_dict = load_musan_dict(tar_path)

            self.assertTrue(index_path.exists())
            self.assertTrue(Path(str(tar_path) + ".sqlite3").exists())
            self.assertEqual(
                {key: len(value) for key, value in path_dict.items()},
                {"noise": 1, "music": 1, "babb": 1},
            )
            self.assertEqual(path_dict["music"][0].member_name, "musan/music/fma/music-clean.wav")

            signal = np.zeros(320, dtype=np.float32)
            for noise_types in (["noise"], ["music"], ["babb", "babb", "babb"]):
                with self.subTest(noise_types=noise_types), mock.patch(
                    "deeplab.dataio.audio.random.choice", return_value=noise_types
                ):
                    augmented = add_noise_from_musan_dict(
                        signal, 16000, path_dict, prob=1.0, snr=[5, 20]
                    )
                self.assertEqual(augmented.shape, signal.shape)
                self.assertTrue(np.isfinite(augmented).all())

    def test_musan_tar_augmentation_with_multiple_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            tar_path = Path(directory, "musan.tar")
            _create_musan_tar(tar_path)
            dataset = _MusanAugDataset(tar_path)
            load_audio(dataset.path_dict["noise"][0], sample_rate=16000)

            batches = list(DataLoader(dataset, batch_size=2, num_workers=2))

            self.assertEqual([tuple(batch.shape) for batch in batches], [(2, 320)] * 2)
            self.assertTrue(all(torch.isfinite(batch).all() for batch in batches))

    def test_segment_load_matches_full_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            tar_path = Path(directory, "audio.tar")
            rng = np.random.default_rng(7)
            payload = (rng.uniform(-0.8, 0.8, size=16000 * 10) * 32000).astype("<i2")
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(16000)
                audio.writeframes(payload.tobytes())
            wav_bytes = buffer.getvalue()
            with tarfile.open(tar_path, "w") as archive:
                member = tarfile.TarInfo("musan/noise/free-sound/segment.wav")
                member.size = len(wav_bytes)
                archive.addfile(member, io.BytesIO(wav_bytes))
            with tarfile.open(tar_path, "r:") as archive:
                info = archive.getmember("musan/noise/free-sound/segment.wav")
            source = TarAudioSource(str(tar_path), info.offset_data, info.size, info.name)

            for _ in range(8):
                segment, sr = load_audio(source, 16000, duration=2.5)
                self.assertEqual(sr, 16000)
                self.assertEqual(segment.shape, (40000,))
            full, sr = load_audio(source, 16000)
            self.assertEqual(sr, 16000)

            # Deterministic byte-exactness: replicate the random frame index
            # the segment reader draws (samples = frames for mono 16-bit) and
            # compare against the matching slice of the full decode.
            np.random.seed(1234)
            expected_frame = np.random.randint(0, (len(full) - 40000) + 1)
            np.random.seed(1234)
            segment, _ = load_audio(source, 16000, duration=2.5)
            self.assertTrue(
                np.array_equal(segment, full[expected_frame:expected_frame + 40000])
            )

    def test_short_member_falls_back_to_full_read(self):
        with tempfile.TemporaryDirectory() as directory:
            tar_path = Path(directory, "short.tar")
            payload = _wav_bytes(sample_count=200)
            with tarfile.open(tar_path, "w") as archive:
                member = tarfile.TarInfo("musan/noise/free-sound/tiny.wav")
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
            with tarfile.open(tar_path, "r:") as archive:
                info = archive.getmember("musan/noise/free-sound/tiny.wav")
            source = TarAudioSource(str(tar_path), info.offset_data, info.size, info.name)

            segment, _ = load_audio(source, 16000, duration=2.5)

            self.assertEqual(segment.shape, (200,))

    def test_listen_artifact_written_from_segment_loads(self):
        output_path = Path(
            tempfile.gettempdir(), "segment_load_listen.wav"
        )
        with tempfile.TemporaryDirectory() as directory:
            tar_path = Path(directory, "audio.tar")
            payload = _wav_bytes(sample_count=16000 * 8)
            with tarfile.open(tar_path, "w") as archive:
                member = tarfile.TarInfo("musan/speech/librivox/segment.wav")
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
            with tarfile.open(tar_path, "r:") as archive:
                info = archive.getmember("musan/speech/librivox/segment.wav")
            source = TarAudioSource(str(tar_path), info.offset_data, info.size, info.name)

            segment, sr = load_audio(source, 16000, duration=2.0)
            self.assertEqual(segment.shape, (32000,))

        with wave.open(str(output_path), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(sr)
            audio.writeframes(
                np.clip(segment * 32000, -32768, 32767).astype("<i2").tobytes()
            )
        print(f"\nListen artifact written to {output_path}", flush=True)

    def test_tar_audio_source_decodes_indexed_wav(self):
        with tempfile.TemporaryDirectory() as directory:
            tar_path = Path(directory, "audio.tar")
            payload = _wav_bytes(sample_count=321)
            with tarfile.open(tar_path, "w") as archive:
                member = tarfile.TarInfo("musan/noise/free-sound/noise.wav")
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))

            with tarfile.open(tar_path, "r:") as archive:
                member = archive.getmember("musan/noise/free-sound/noise.wav")
            source = TarAudioSource(
                str(tar_path), member.offset_data, member.size, member.name
            )

            signal, sample_rate = load_audio(source, sample_rate=16000)

            self.assertEqual(sample_rate, 16000)
            self.assertEqual(signal.shape, (321,))
            self.assertTrue(np.isfinite(signal).all())

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


def _m4a_bytes(sample_count=16000 * 8, sample_rate=16000, seed=0):
    buffer = io.BytesIO()
    container = av.open(buffer, mode="w", format="mp4")
    stream = container.add_stream("aac", rate=sample_rate)
    stream.layout = "mono"
    rng = np.random.default_rng(seed)
    samples = rng.uniform(-0.5, 0.5, size=sample_count).astype(np.float32)
    frame_stride = 1024
    for start in range(0, sample_count, frame_stride):
        chunk = samples[start:start + frame_stride].reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(chunk, format="fltp", layout="mono")
        frame.sample_rate = sample_rate
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)
    container.close()
    return buffer.getvalue()


def _best_alignment_offset(full, segment):
    # FFT correlation over every offset: broadband noise decorrelates within
    # a few samples, so a strided search would miss the alignment peak.
    centered = segment - segment.mean()
    full_centered = full - full.mean()
    corr = fftconvolve(full_centered, centered[::-1], mode="valid")
    peak = int(np.argmax(np.abs(corr)))
    denom = (
        np.linalg.norm(full_centered[peak:peak + len(centered)])
        * np.linalg.norm(centered)
        + 1e-9
    )
    return peak, float(corr[peak]) / denom


class WindowedM4ADecodeTest(unittest.TestCase):
    def test_window_no_larger_than_payload_matches_full_decode(self):
        payload = _m4a_bytes()
        full = decode_m4a_bytes(payload, 16000)
        self.assertTrue(
            np.array_equal(
                decode_m4a_bytes(payload, 16000, window_samples=len(full)),
                full,
            )
        )
        self.assertTrue(
            np.array_equal(
                decode_m4a_bytes(payload, 16000, window_samples=len(full) * 2),
                full,
            )
        )

    def test_decode_without_window_ignores_none(self):
        payload = _m4a_bytes(seed=2)
        full = decode_m4a_bytes(payload, 16000)
        self.assertTrue(
            np.array_equal(
                decode_m4a_bytes(payload, 16000, window_samples=None), full
            )
        )

    def test_window_mid_payload_matches_full_decode_segment(self):
        payload = _m4a_bytes(seed=1)
        full = decode_m4a_bytes(payload, 16000)
        window = 16000 * 2
        np.random.seed(99)
        decoded = decode_m4a_bytes(payload, 16000, window_samples=window)
        self.assertGreaterEqual(len(decoded), window)
        _, score = _best_alignment_offset(full, decoded[:window])
        self.assertGreater(score, 0.9)


class TarTrainDatasetM4ATest(unittest.TestCase):
    def test_windowed_dataset_items_have_requested_length(self):
        with tempfile.TemporaryDirectory() as directory:
            shard_path = Path(directory, "samples.tar")
            index_path = Path(directory, "samples.sqlite3")
            files = {}
            for label in range(4):
                key = f"id{label:05d}/video/{label:05d}"
                files[key + ".m4a"] = _m4a_bytes(seed=label)
                files[key + ".cls"] = str(label).encode("utf-8")
            with tarfile.open(shard_path, "w") as archive:
                for name, payload in sorted(files.items()):
                    member = tarfile.TarInfo(name)
                    member.size = len(payload)
                    archive.addfile(member, io.BytesIO(payload))
            build_tar_index([shard_path], index_path)
            dataset = TarTrainDataset(_hparams(index_path))

            for index in range(8):
                item = dataset[(index, 2.5)]
                self.assertEqual(tuple(item["aud_inputs"].shape), (40000,))
                self.assertTrue(torch.isfinite(item["aud_inputs"]).all())
            dataset.close()


if __name__ == "__main__":
    unittest.main()
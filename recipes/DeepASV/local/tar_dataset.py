import io
import math
import os
import sqlite3
from collections import OrderedDict

import av
import numpy as np
import torch

from deeplab.dataio.audio import (
    add_noise_from_musan_dict,
    add_reverberation,
    speed_augmentation,
    truncate_audio_random,
)
from deeplab.utils.corpus import load_musan_dict, load_rirs
try:
    from .tar_index import SCHEMA_VERSION, pread_exact
except ImportError:
    from tar_index import SCHEMA_VERSION, pread_exact


# Extra samples decoded past the end of a seek window, in output samples:
# covers AAC sync-frame backoff at the seek point and resampler priming.
_WINDOW_MARGIN_SAMPLES = 4000

# Extra input samples cropped before speed perturbation, so the sox output is
# longer than the target plus its trimmed edges and never has to be tiled with itself.
_SPEED_MARGIN_SAMPLES = 400


def decode_m4a_bytes(data, sample_rate, window_samples=None):
    """Decode an m4a payload to float32 mono at ``sample_rate``.

    With ``window_samples``, decode a random window of that length (plus a
    small margin) instead of the whole payload. Payloads without duration
    metadata, shorter than the window, or whose seeked decode falls short
    of the window fall back to a full decode.
    """
    if window_samples is not None and window_samples > 0:
        signal = _decode_m4a_window(data, sample_rate, window_samples)
        if signal is not None:
            return signal
    chunks = []
    with av.open(io.BytesIO(data), mode="r") as container:
        if not container.streams.audio:
            raise ValueError("M4A payload has no audio stream")
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=sample_rate)
        for frame in container.decode(audio=0):
            for resampled in resampler.resample(frame):
                chunks.append(resampled.to_ndarray()[0])
        for resampled in resampler.resample(None):
            chunks.append(resampled.to_ndarray()[0])
    if not chunks:
        raise ValueError("M4A payload decoded to no samples")
    signal = np.concatenate(chunks).astype(np.float32, copy=False)
    if not np.isfinite(signal).all():
        raise ValueError("M4A payload decoded to non-finite samples")
    return signal


def _decode_m4a_window(data, sample_rate, window_samples):
    """Decode one random ``window_samples`` window; None means fall back."""
    try:
        with av.open(io.BytesIO(data), mode="r") as container:
            if not container.streams.audio:
                return None
            stream = container.streams.audio[0]
            native_rate = stream.codec_context.sample_rate
            if not native_rate or not stream.duration:
                return None
            total_native = int(stream.duration * stream.time_base * native_rate)
            needed_native = int(
                (window_samples + _WINDOW_MARGIN_SAMPLES) * native_rate / sample_rate
            )
            if total_native <= needed_native:
                return None
            start = np.random.randint(0, total_native - needed_native)
            start_pts = (
                start * stream.time_base.denominator
            ) // (native_rate * stream.time_base.numerator)
            try:
                container.seek(start_pts, stream=stream, backward=True, any_frame=False)
            except av.error.FFmpegError:
                return None
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=sample_rate)
            chunks = []
            count = 0
            for frame in container.decode(stream):
                for resampled in resampler.resample(frame):
                    chunk = resampled.to_ndarray()[0]
                    chunks.append(chunk)
                    count += chunk.shape[0]
                if count >= window_samples + _WINDOW_MARGIN_SAMPLES:
                    break
            for resampled in resampler.resample(None):
                chunk = resampled.to_ndarray()[0]
                chunks.append(chunk)
                count += chunk.shape[0]
            if count < window_samples:
                return None
            signal = np.concatenate(chunks).astype(np.float32, copy=False)
            if not np.isfinite(signal).all():
                raise ValueError("M4A payload decoded to non-finite samples")
            return signal
    except av.error.FFmpegError:
        return None


def load_tar_index(index_path):
    connection = sqlite3.connect(index_path)
    try:
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        if int(metadata.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported tar index schema {metadata.get('schema_version')!r}"
            )
        shards = list(
            connection.execute(
                "SELECT path, size, mtime_ns FROM shards ORDER BY shard_id"
            )
        )
        rows = connection.execute(
            """
            SELECT shard_id, offset, size, speaker_label
            FROM samples
            ORDER BY sample_id
            """
        ).fetchall()
    finally:
        connection.close()

    for path, expected_size, expected_mtime_ns in shards:
        stat = os.stat(path)
        if stat.st_size != expected_size or stat.st_mtime_ns != expected_mtime_ns:
            raise ValueError(f"Tar shard changed after indexing: {path}")
    records = np.asarray(rows, dtype=np.int64)
    if records.ndim != 2 or records.shape[1] != 4:
        raise ValueError(f"Invalid sample records in {index_path}")
    return metadata, [path for path, _, _ in shards], records


class TarTrainDataset(torch.utils.data.Dataset):
    def __init__(self, hparams, feature_extractor=None):
        super().__init__()
        # With a feature extractor, __getitem__ returns fbank features instead
        # of the waveform, so extraction runs in the DataLoader workers.
        self.feature_extractor = feature_extractor
        self.repeat = hparams["training_loop"]
        self.sr = hparams["sample_rate"]
        self.speed_perturbation = hparams["speed_perturbation"]
        self.data_aug = hparams["data_aug"]
        self.max_open_shards = int(hparams.get("tar_max_open_shards", 32))
        if self.max_open_shards < 1:
            raise ValueError("tar_max_open_shards must be positive")

        metadata, self.shard_paths, self.records = load_tar_index(
            hparams["train_tar_index"]
        )
        self.spk_num = int(metadata["num_speakers"])
        self.spk_ids = list(range(self.spk_num))
        self.speed_values = (
            [] if self.speed_perturbation is None else list(self.speed_perturbation)
        )
        self.utt_list = range(len(self.records) * (1 + len(self.speed_values)))
        self._descriptors = OrderedDict()

        if self.data_aug:
            self.musan_dict = load_musan_dict(hparams["musan_path"])
            self.rirs_list = load_rirs(hparams["rirs_path"])
        else:
            self.musan_dict = None
            self.rirs_list = None

    def __len__(self):
        return len(self.utt_list) * self.repeat

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_descriptors"] = OrderedDict()
        return state

    def close(self):
        descriptors = getattr(self, "_descriptors", None)
        if descriptors is None:
            return
        for descriptor in descriptors.values():
            os.close(descriptor)
        descriptors.clear()

    def __del__(self):
        self.close()

    def _descriptor(self, shard_id):
        descriptor = self._descriptors.pop(shard_id, None)
        if descriptor is None:
            descriptor = os.open(self.shard_paths[shard_id], os.O_RDONLY)
        self._descriptors[shard_id] = descriptor
        if len(self._descriptors) > self.max_open_shards:
            _, oldest = self._descriptors.popitem(last=False)
            os.close(oldest)
        return descriptor

    def _read_audio(self, shard_id, offset, size, window_samples):
        data = pread_exact(self._descriptor(shard_id), size, offset)
        return decode_m4a_bytes(data, self.sr, window_samples=window_samples)

    def __getitem__(self, idx_data):
        idx, dur = idx_data
        idx %= len(self.utt_list)
        variant, record_idx = divmod(idx, len(self.records))
        shard_id, offset, size, speaker_label = self.records[record_idx]
        tlen = int(dur * self.sr)
        if variant:
            speed = self.speed_values[variant - 1]
            # crop enough input that the sped-up/slowed output still covers tlen
            crop_len = int(math.ceil(tlen * speed)) + _SPEED_MARGIN_SAMPLES
        else:
            crop_len = tlen
        signal = self._read_audio(int(shard_id), int(offset), int(size), crop_len)
        signal = truncate_audio_random(signal, crop_len)

        if variant:
            signal = speed_augmentation(signal, self.sr, speed, tlen=tlen)
            speaker_label += variant * self.spk_num

        if self.data_aug:
            aug_type = np.random.choice(["none", "noise", "reverb"])
            if aug_type == "noise":
                signal = add_noise_from_musan_dict(
                    signal, self.sr, self.musan_dict, prob=1.0, snr=[5, 20]
                )
            if aug_type == "reverb":
                signal = add_reverberation(
                    signal, self.sr, self.rirs_list, prob=1.0
                )

        if self.feature_extractor is not None:
            aud_inputs = self.feature_extractor(
                signal.astype(np.float32, copy=False),
                sampling_rate=self.sr,
                return_tensors="pt",
                padding=False,
                truncation=False,
                return_attention_mask=False,
            ).input_features[0]
        else:
            aud_inputs = torch.from_numpy(signal).float()

        return {
            "aud_inputs": aud_inputs,
            "spk_labels": torch.tensor(speaker_label).long(),
        }
import io
import os
import tarfile
import wave
from dataclasses import dataclass

import numpy as np
import scipy.io.wavfile as sciwav
import soundfile as sf
import json
import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml, dump_hyperpyyaml


@dataclass(frozen=True)
class TarAudioSource:
    tar_path: str
    offset: int
    size: int
    member_name: str


# One TarFile per tar, per process. A TarFile is NOT thread-safe (extractfile
# shares the underlying file position); DataLoader workers are processes, and
# the pid check below prevents children from inheriting parent file positions
# after fork.
_tar_readers = {}
_tar_reader_pid = None


def _open_tar(tar_path):
    global _tar_reader_pid

    pid = os.getpid()
    if _tar_reader_pid != pid:
        for reader in _tar_readers.values():
            reader.close()
        _tar_readers.clear()
        _tar_reader_pid = pid

    reader = _tar_readers.get(tar_path)
    if reader is None:
        reader = tarfile.open(tar_path, mode="r:")
        _tar_readers[tar_path] = reader
    return reader


def _read_tar_audio(source):
    """Read a full tar member into memory."""
    with _open_tar(source.tar_path).extractfile(source.member_name) as stream:
        return stream.read()


def _write_wav_container(stream, payload_bytes):
    """Wrap raw PCM frames in an in-memory RIFF/WAVE container."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(stream.getnchannels())
        output.setsampwidth(stream.getsampwidth())
        output.setframerate(stream.getframerate())
        output.writeframes(payload_bytes)
    return buffer.getvalue()


def _read_tar_segment(source, duration=None):
    """Read `duration` seconds from a PCM WAV member at a random frame offset.

    Falls back to the full member when the window covers it or the member is
    not plain PCM, which preserves the original whole-file behavior.
    """
    if duration is None:
        return _read_tar_audio(source)

    reader = _open_tar(source.tar_path)
    stream = reader.extractfile(source.member_name)
    try:
        with wave.open(stream, "rb") as audio:
            frames = int(duration * audio.getframerate())
            total_frames = audio.getnframes()
            if frames <= 0 or frames >= total_frames:
                return _write_wav_container(audio, audio.readframes(total_frames))
            frame_offset = np.random.randint(0, total_frames - frames + 1)
            audio.setpos(frame_offset)
            payload = _write_wav_container(audio, audio.readframes(frames))
        return payload
    except (wave.Error, EOFError):
        # Non-PCM or truncated member: fall back to the whole file.
        stream.seek(0)
        return stream.read()


def init_output_dir(output_path):
    target_dir = os.path.split(output_path)[0]
    if len(target_dir) > 0 and (not os.path.exists(target_dir)):
        os.makedirs(target_dir, exist_ok=True)


def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f) 
    return data


def save_json(path, data):
    init_output_dir(path)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    

def read_hyperyaml(path):
    with open(path, 'r') as f:
        data = load_hyperpyyaml(f.read())
    return data


def save_hyperyaml(path, yaml_data):
    init_output_dir(path)
    with open(path, 'w') as f:
        dump_hyperpyyaml(yaml_data, f)


def load_audio(path, sample_rate=16000, channels=0, duration=None):
    if isinstance(path, TarAudioSource):
        payload = _read_tar_segment(path, duration)
        signal, sr = sf.read(io.BytesIO(payload))
    else:
        signal, sr = sf.read(path)

    if sample_rate and sample_rate != sr:
        effects = [['remix', '1'], ['lowpass', f'{sample_rate//2}'], ['rate', f'{sample_rate}'],]
        if isinstance(path, TarAudioSource):
            signal = torch.from_numpy(np.asarray(signal)).float()
            signal = signal.unsqueeze(0) if signal.ndim == 1 else signal.transpose(0, 1)
            signal, sr = torchaudio.sox_effects.apply_effects_tensor(
                signal, sr, effects
            )
        else:
            signal, sr = torchaudio.sox_effects.apply_effects_file(path, effects)
        signal = signal.squeeze(0).numpy()

    if len(signal.shape)==2 and channels!='all':
        signal  = signal[:, channels]
    return signal, sr

def load_concatenated_audio_by_rttm(rttm_list, path, sr=16000, channels=0, min_dt=0):
    signal_list = []
    for rttm_data in rttm_list:
        if rttm_data['dt'] >= min_dt:
            p1 = int(rttm_data['st'] * sr)
            p2 = int(rttm_data['st'] * sr + rttm_data['dt'] * sr)
            signal = load_audio(path, sr, channels)[0][p1:p2]
            signal_list.append(signal)
    
    if len(signal_list) > 0:
        return np.concatenate(signal_list, axis=0)


    
def load_scp(path, sep='\t'):
    scp_list = []
    with open(path, 'r') as f:
        for line in f.readlines():
            scp_data = line.strip('\n').split(sep)
            scp_list.append(dict(reco=scp_data[0],wav_path=scp_data[1]))
    return scp_list

    
def save_scp(path, scp_list, sep='\t'):
    init_output_dir(path)
    with open(path, 'w') as f:
        for scp_data in scp_list:
            line = scp_data['reco'] + sep + scp_data['wav_path'] + '\n'
            f.writelines(line)


def load_trial(path):
    trial_list = []
    with open(path, 'r') as f:
        for line in f:
            key, utt1, utt2 = line.strip('\n').split(' ')
            trial_list.append(dict(key=key, utt1=utt1, utt2=utt2))
    return trial_list


def save_trial(path, trial_list):
    init_output_dir(path)
    with open(path, 'w') as f:
        for trial_data in trial_list:
            line = '{} {} {}\n'.format(trial_data['key'], trial_data['utt1'], trial_data['utt2'])
            f.writelines(line)






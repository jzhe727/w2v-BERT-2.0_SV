import fcntl
import os
import posixpath
import sqlite3
import tarfile
from pathlib import Path

from deeplab.utils.fileio import TarAudioSource


MUSAN_INDEX_SCHEMA_VERSION = 1
_MUSAN_CATEGORIES = {"noise": "noise", "music": "music", "speech": "babb"}


def _normalized_tar_name(name):
    return name.removeprefix("./").lstrip("/")


def _musan_category(name):
    parts = _normalized_tar_name(name).split("/")
    if parts and parts[0] == "musan":
        parts = parts[1:]
    if len(parts) >= 2:
        return _MUSAN_CATEGORIES.get(parts[0])
    return None


def build_musan_tar_index(tar_path, output_path):
    tar_path = str(Path(tar_path).resolve())
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    temporary_path.unlink(missing_ok=True)

    wav_members = {}
    nonvocal_music = set()
    with tarfile.open(tar_path, mode="r:") as archive:
        for member in archive:
            if not member.isfile():
                continue
            name = _normalized_tar_name(member.name)
            if name.lower().endswith(".wav"):
                if name in wav_members:
                    raise ValueError(f"Duplicate MUSAN member {name!r}")
                wav_members[name] = (member.offset_data, member.size)
            elif name.endswith("/ANNOTATIONS") and "/music/" in f"/{name}":
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError(f"Could not read {member.name!r}")
                for line in extracted.read().decode("utf-8").splitlines():
                    fields = line.split()
                    if len(fields) >= 3 and fields[2] == "N":
                        nonvocal_music.add(
                            posixpath.join(posixpath.dirname(name), fields[0] + ".wav")
                        )

    missing_music = nonvocal_music - wav_members.keys()
    if missing_music:
        example = sorted(missing_music)[0]
        raise ValueError(f"Annotated MUSAN music member is missing: {example}")

    rows = []
    for name, (offset, size) in wav_members.items():
        category = _musan_category(name)
        if category is None or (category == "music" and name not in nonvocal_music):
            continue
        rows.append((category, name, offset, size))
    rows.sort()

    counts = {category: 0 for category in _MUSAN_CATEGORIES.values()}
    for category, _, _, _ in rows:
        counts[category] += 1
    empty_categories = [category for category, count in counts.items() if count == 0]
    if empty_categories:
        raise ValueError(f"MUSAN archive has no members for {empty_categories}")

    stat = os.stat(tar_path)
    connection = sqlite3.connect(temporary_path)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode = OFF;
            PRAGMA synchronous = OFF;
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE members (
                member_id INTEGER PRIMARY KEY,
                category TEXT NOT NULL,
                name TEXT UNIQUE NOT NULL,
                offset INTEGER NOT NULL,
                size INTEGER NOT NULL
            );
            """
        )
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            (
                ("schema_version", str(MUSAN_INDEX_SCHEMA_VERSION)),
                ("tar_path", tar_path),
                ("tar_size", str(stat.st_size)),
                ("tar_mtime_ns", str(stat.st_mtime_ns)),
                ("num_noise", str(counts["noise"])),
                ("num_music", str(counts["music"])),
                ("num_babb", str(counts["babb"])),
            ),
        )
        connection.executemany(
            "INSERT INTO members(category, name, offset, size) VALUES (?, ?, ?, ?)",
            rows,
        )
        connection.commit()
        connection.close()
        os.replace(temporary_path, output_path)
    except BaseException:
        connection.close()
        temporary_path.unlink(missing_ok=True)
        raise


def _load_musan_tar_index(tar_path, index_path):
    tar_path = str(Path(tar_path).resolve())
    connection = sqlite3.connect(index_path)
    try:
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        rows = connection.execute(
            "SELECT category, name, offset, size FROM members ORDER BY member_id"
        ).fetchall()
    finally:
        connection.close()

    if int(metadata.get("schema_version", -1)) != MUSAN_INDEX_SCHEMA_VERSION:
        raise ValueError(f"Unsupported MUSAN index schema in {index_path}")
    stat = os.stat(tar_path)
    expected = (
        metadata.get("tar_path"),
        int(metadata.get("tar_size", -1)),
        int(metadata.get("tar_mtime_ns", -1)),
    )
    if expected != (tar_path, stat.st_size, stat.st_mtime_ns):
        raise ValueError(f"MUSAN tar changed after indexing: {tar_path}")

    path_dict = dict(noise=[], music=[], babb=[])
    for category, name, offset, size in rows:
        path_dict[category].append(TarAudioSource(tar_path, offset, size, name))
    return path_dict


def _load_musan_tar(tar_path):
    if str(tar_path).endswith((".tar.gz", ".tgz")):
        raise ValueError("MUSAN must be an uncompressed .tar for random access")
    index_path = Path(str(tar_path) + ".sqlite3")
    lock_path = Path(str(index_path) + ".lock")
    with open(lock_path, "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not index_path.exists():
            build_musan_tar_index(tar_path, index_path)
    return _load_musan_tar_index(tar_path, index_path)


def init_spk2utt(dataset_dir, subset, spk2utt):
    cache_path = os.path.join(dataset_dir, '{}.spk2utt'.format(subset))
    if not os.path.exists(cache_path):
        print(f'No_{dataset_dir}_{subset}')
    
    with open(cache_path, 'r') as f:
        for line in f.readlines():
            spk_id, utt_path = line.strip('\n').split('\t')
            if spk_id not in spk2utt:
                spk2utt[spk_id] = []
            spk2utt[spk_id].append(utt_path)
    return 


def load_musan_dict(dataset_dir):
    "Load musan noises without vocals."
    if os.path.isfile(dataset_dir):
        return _load_musan_tar(dataset_dir)

    path_dict = dict(noise=[], music=[], babb=[])

    # noise part
    for cls in ['noise/free-sound', 'noise/sound-bible']:
        cls_dir = os.path.join(dataset_dir, cls)
        for file in os.listdir(cls_dir):
            audio = os.path.join(dataset_dir, cls_dir, file)
            if os.path.exists(audio) and audio.endswith('.wav'):
                path_dict['noise'].append(audio)

    # music part
    for cls in ['music/fma', 'music/fma-western-art', 'music/hd-classical', 'music/jamendo', 'music/rfm']:
        anno_path = os.path.join(dataset_dir, cls, 'ANNOTATIONS')
        with open(anno_path, 'r') as f:
            annos = f.readlines()
        for d in annos:
            vocal = d.split(' ')[2]
            audio = os.path.join(dataset_dir, cls, d.split(' ')[0]+'.wav')
            if vocal=='N' and os.path.exists(audio):
                path_dict['music'].append(audio)

    # babb part         
    for cls in ['speech/librivox', 'speech/us-gov']:
        cls_dir = os.path.join(dataset_dir, cls)
        for file in os.listdir(cls_dir):
            audio = os.path.join(dataset_dir, cls_dir, file)
            if os.path.exists(audio) and audio.endswith('.wav'):
                path_dict['babb'].append(audio)
                
    return path_dict


def load_rirs(dataset_dir):
    
    path_list = []
    for d in ['simulated_rirs/mediumroom','simulated_rirs/smallroom']:
        sub_dir = os.path.join(dataset_dir, d)
        if os.path.exists(sub_dir) and os.path.isdir(sub_dir):
            for r in os.listdir(sub_dir):
                room_dir = os.path.join(sub_dir, r)
                if not os.path.isdir(room_dir):   
                    continue
                for file in os.listdir(room_dir):
                    audio = os.path.join(room_dir, file)
                    if os.path.exists(audio) and audio.endswith('.wav'):
                        path_list.append(audio)
                        
    return path_list


def load_audio_corpus(dataset_dir,
                    subsets=['audio']):
    spk2utt = {}
    for subset in subsets:
        init_spk2utt(dataset_dir, subset, spk2utt)

    return spk2utt

    
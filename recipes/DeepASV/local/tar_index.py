import argparse
import glob
import os
import sqlite3
import tarfile
from pathlib import Path


SCHEMA_VERSION = 1


def _finish_sample(connection, shard_id, key, members, speaker_labels):
    if set(members) != {"cls", "m4a"}:
        raise ValueError(f"Incomplete sample {key!r}: found {sorted(members)}")

    label = int(members["cls"].decode("utf-8").strip())
    speaker_id = key.split("/", 1)[0]
    previous_label = speaker_labels.setdefault(speaker_id, label)
    if previous_label != label:
        raise ValueError(
            f"Speaker {speaker_id!r} has labels {previous_label} and {label}"
        )

    offset, size = members["m4a"]
    connection.execute(
        """
        INSERT INTO samples(sample_key, shard_id, offset, size, speaker_id, speaker_label)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (key, shard_id, offset, size, speaker_id, label),
    )


def build_tar_index(shard_paths, output_path, progress=None):
    shard_paths = sorted(str(Path(path).resolve()) for path in shard_paths)
    if not shard_paths:
        raise ValueError("No tar shards were provided")

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    temporary_path.unlink(missing_ok=True)

    connection = sqlite3.connect(temporary_path)
    speaker_labels = {}
    try:
        connection.executescript(
            """
            PRAGMA journal_mode = OFF;
            PRAGMA synchronous = OFF;
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE shards (
                shard_id INTEGER PRIMARY KEY,
                path TEXT UNIQUE NOT NULL,
                size INTEGER NOT NULL,
                mtime_ns INTEGER NOT NULL
            );
            CREATE TABLE samples (
                sample_id INTEGER PRIMARY KEY,
                sample_key TEXT UNIQUE NOT NULL,
                shard_id INTEGER NOT NULL,
                offset INTEGER NOT NULL,
                size INTEGER NOT NULL,
                speaker_id TEXT NOT NULL,
                speaker_label INTEGER NOT NULL,
                FOREIGN KEY(shard_id) REFERENCES shards(shard_id)
            );
            """
        )
        connection.execute(
            "INSERT INTO metadata(key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )

        for shard_id, shard_path in enumerate(shard_paths):
            stat = os.stat(shard_path)
            connection.execute(
                "INSERT INTO shards(shard_id, path, size, mtime_ns) VALUES (?, ?, ?, ?)",
                (shard_id, shard_path, stat.st_size, stat.st_mtime_ns),
            )

            current_key = None
            members = {}
            with tarfile.open(shard_path, mode="r:") as archive:
                for member in archive:
                    if not member.isfile():
                        continue
                    key, extension = os.path.splitext(member.name)
                    extension = extension.lstrip(".").lower()
                    if extension not in {"cls", "m4a"}:
                        continue
                    if current_key is not None and key != current_key:
                        _finish_sample(
                            connection, shard_id, current_key, members, speaker_labels
                        )
                        members = {}
                    current_key = key
                    if extension in members:
                        raise ValueError(
                            f"Duplicate .{extension} member for {key!r} in {shard_path}"
                        )
                    if extension == "cls":
                        extracted = archive.extractfile(member)
                        if extracted is None:
                            raise ValueError(f"Could not read {member.name!r}")
                        members[extension] = extracted.read()
                    else:
                        members[extension] = (member.offset_data, member.size)

            if current_key is not None:
                _finish_sample(connection, shard_id, current_key, members, speaker_labels)
            if progress is not None:
                progress(shard_id + 1, len(shard_paths), shard_path)

        labels = list(speaker_labels.values())
        if any(label < 0 for label in labels):
            raise ValueError("Speaker labels must be nonnegative")
        if len(set(labels)) != len(labels):
            raise ValueError("Each speaker label must identify exactly one speaker")
        num_speakers = max(labels, default=-1) + 1
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            (
                ("num_shards", str(len(shard_paths))),
                ("num_samples", str(connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0])),
                ("num_speakers", str(num_speakers)),
            ),
        )
        connection.commit()
        connection.close()
        os.replace(temporary_path, output_path)
    except BaseException:
        connection.close()
        temporary_path.unlink(missing_ok=True)
        raise


def pread_exact(descriptor, size, offset):
    chunks = []
    bytes_read = 0
    while bytes_read < size:
        chunk = os.pread(descriptor, size - bytes_read, offset + bytes_read)
        if not chunk:
            raise OSError(f"Short read: expected {size} bytes, got {bytes_read}")
        chunks.append(chunk)
        bytes_read += len(chunk)
    return b"".join(chunks)


def read_member_at(path, offset, size):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        return pread_exact(descriptor, size, offset)
    finally:
        os.close(descriptor)


def _main():
    parser = argparse.ArgumentParser(description="Build a random-access index for WebDataset tar shards")
    parser.add_argument("--shard-glob", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    shards = glob.glob(args.shard_glob)

    def report(completed, total, path):
        print(f"[{completed}/{total}] {path}", flush=True)

    build_tar_index(shards, args.output, progress=report)


if __name__ == "__main__":
    _main()
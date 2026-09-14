import argparse
import os
from pathlib import Path


def prepare_protocol(wav_root, trial_source, output_dir):
    wav_root = Path(wav_root).resolve()
    trial_source = Path(trial_source).resolve()
    output_dir = Path(output_dir).resolve()

    trials = []
    utterances = set()
    with open(trial_source, "r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            fields = line.split()
            if len(fields) != 3 or fields[0] not in {"0", "1"}:
                raise ValueError(
                    f"Invalid trial at {trial_source}:{line_number}: {line.rstrip()!r}"
                )
            label, first, second = fields
            trials.append((label, first, second))
            utterances.update((first, second))

    missing = [utterance for utterance in sorted(utterances) if not (wav_root / utterance).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} trial utterances are missing under {wav_root}; first: {missing[0]}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    scp_path = output_dir / "wav.scp"
    trial_path = output_dir / "trials"
    temporary_scp = scp_path.with_name(f".{scp_path.name}.{os.getpid()}.tmp")
    temporary_trials = trial_path.with_name(f".{trial_path.name}.{os.getpid()}.tmp")
    try:
        with open(temporary_scp, "w", encoding="utf-8") as output:
            for utterance in sorted(utterances):
                output.write(f"{utterance}\t{wav_root / utterance}\n")
        with open(temporary_trials, "w", encoding="utf-8") as output:
            for label, first, second in trials:
                output.write(f"{label} {first} {second}\n")
        os.replace(temporary_scp, scp_path)
        os.replace(temporary_trials, trial_path)
    except BaseException:
        temporary_scp.unlink(missing_ok=True)
        temporary_trials.unlink(missing_ok=True)
        raise

    return len(trials), len(utterances)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare wav.scp and trials files for an extracted VoxCeleb1 protocol"
    )
    parser.add_argument("--wav-root", required=True)
    parser.add_argument("--trial-source", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    trial_count, utterance_count = prepare_protocol(
        args.wav_root, args.trial_source, args.output_dir
    )
    print(
        f"Prepared {trial_count} trials over {utterance_count} utterances in {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
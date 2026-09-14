"""Two-GPU dataloading smoke test mirroring the DeepASV training data path.

Launched via torchrun exactly like train.py (native DDP, NCCL backend). Builds
TarTrainDataset + WavBatchSampler + DataLoader with the same settings as
conf/w2v-bert/s1.yaml, then iterates batches with a dummy task: validate
shapes/labels/finiteness and exercise a tiny NCCL all_reduce so collective
errors surface without allocating real model memory.

Run (2 ranks):
    torchrun --nnodes 1 --nproc_per_node=2 tests/gpu_dataloader_smoke.py \
        --tar-index /scratch/46889734/voxceleb2-dev-wds/voxceleb2.index.sqlite3
"""

import argparse
import json
import os
import sys
import time
from itertools import islice

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from deeplab.utils.misc import seed_worker
from local.sampler import WavBatchSampler
from local.tar_dataset import TarTrainDataset


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tar-index", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-iters", type=int, default=50)
    parser.add_argument("--dur-range", type=float, nargs=2, default=[2.0, 3.0])
    parser.add_argument("--speed-perturbation", type=float, nargs="*", default=[0.9, 1.1])
    parser.add_argument("--data-aug", action="store_true")
    parser.add_argument("--musan-path")
    parser.add_argument("--rirs-path")
    parser.add_argument("--result-json", default=None)
    args = parser.parse_args()
    if args.data_aug and (not args.musan_path or not args.rirs_path):
        parser.error("--data-aug requires --musan-path and --rirs-path")
    return args


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)

    hparams = {
        "training_loop": 1,
        "sample_rate": 16000,
        "speed_perturbation": args.speed_perturbation or None,
        "data_aug": args.data_aug,
        "musan_path": args.musan_path,
        "rirs_path": args.rirs_path,
        "tar_max_open_shards": 32,
        "train_tar_index": args.tar_index,
    }
    dataset = TarTrainDataset(hparams)
    num_variants = 1 + len(args.speed_perturbation or [])
    num_classes = dataset.spk_num * num_variants
    if rank == 0:
        print(
            f"dataset: {len(dataset.records)} clips, {dataset.spk_num} speakers, "
            f"{num_classes} classes, {len(dataset.shard_paths)} shards",
            flush=True,
        )
    check(dataset.spk_num == 5994, f"Expected 5994 speakers, got {dataset.spk_num}")
    check(len(dataset.records) == 1_092_009, f"Expected 1092009 clips, got {len(dataset.records)}")

    sampler = WavBatchSampler(
        dataset,
        args.dur_range,
        shuffle=True,
        batch_size=args.batch_size,
        drop_last=True,
        distributed=True,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        pin_memory=True,
    )

    # Mismatched lengths across ranks cause silent DDP hangs in real training.
    length = torch.tensor([len(loader)], dtype=torch.long, device=device)
    lengths = [torch.zeros_like(length) for _ in range(world_size)]
    dist.all_gather(lengths, length)
    check(
        len({int(item.item()) for item in lengths}) == 1,
        f"Dataloader length differs across ranks: {[int(i.item()) for i in lengths]}",
    )

    # A bounded prefix catches rank partitioning regressions without gathering
    # millions of Python integers. DistributedSampler may duplicate one padded
    # tail item when the dataset length is not divisible by world size.
    sampler.set_epoch(0)
    local_indices = set(islice(iter(sampler.sampler), 4096))
    gathered_indices = [None] * world_size
    dist.all_gather_object(gathered_indices, local_indices)
    if rank == 0:
        union = set().union(*gathered_indices)
        total = sum(len(part) for part in gathered_indices)
        check(len(union) == total, "Rank sampler prefixes overlap")
        print(f"prefix partitioning ok: {total} indices across {world_size} ranks", flush=True)

    stats = {"epochs": [], "world_size": world_size, "batch_size": args.batch_size}
    min_samples = int(args.dur_range[0] * hparams["sample_rate"])
    max_samples = int(args.dur_range[1] * hparams["sample_rate"])

    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch)
        start = time.perf_counter()
        batches = 0
        label_min, label_max = num_classes, -1
        running = torch.zeros(1, device=device)
        for iter_idx, inputs in enumerate(loader, start=1):
            audio = inputs["aud_inputs"]
            labels = inputs["spk_labels"]
            check(audio.shape[0] == args.batch_size, f"Bad batch dim {tuple(audio.shape)}")
            check(
                min_samples <= audio.shape[1] <= max_samples,
                f"Crop length {audio.shape[1]} outside [{min_samples}, {max_samples}]",
            )
            check(audio.dtype == torch.float32 and labels.dtype == torch.int64, "Bad dtypes")
            check(torch.isfinite(audio).all().item(), "Non-finite audio in batch")
            check(
                0 <= int(labels.min()) and int(labels.max()) < num_classes,
                f"Labels outside [0, {num_classes})",
            )
            label_min = min(label_min, int(labels.min()))
            label_max = max(label_max, int(labels.max()))
            # Dummy task: scalar to GPU + collective, negligible memory.
            running += audio.abs().mean().to(device)
            dist.all_reduce(running)
            batches += 1
            if batches >= args.max_iters:
                break
        elapsed = time.perf_counter() - start
        clips = batches * args.batch_size
        epoch_stats = {
            "epoch": epoch,
            "batches": batches,
            "clips_per_second_per_rank": round(clips / elapsed, 2),
            "label_min": label_min,
            "label_max": label_max,
            "reduced_mean_abs": float(running.item()) / max(batches, 1),
        }
        stats["epochs"].append(epoch_stats)
        check(batches == min(args.max_iters, len(loader)), "Loader ended early")
        check(float(running.item()) > 0.0, "All-reduced audio energy is zero")
        if rank == 0:
            print(json.dumps(epoch_stats), flush=True)

    dist.barrier()
    if rank == 0:
        print("GPU DATALOADER SMOKE TEST PASSED", flush=True)
        if args.result_json:
            with open(args.result_json, "w") as f:
                json.dump(stats, f, indent=2)
    dataset.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

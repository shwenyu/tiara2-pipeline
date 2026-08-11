#!/usr/bin/env python3
"""Cache frozen v2.3.2 logits for crop-derived k7 features with torchrun."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

HEADS = ("root", "euk", "prok", "organelle")


def split_range(n: int, rank: int, world: int):
    start = n * rank // world
    return start, n * (rank + 1) // world


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args(argv)

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    from tiara.hierarchical.model import HierarchicalClassifier

    features = Path(args.features)
    meta = json.loads((features / "v241b_features.json").read_text())
    label_meta = json.loads((Path(args.labels) / "composite_features.json").read_text())
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = HierarchicalClassifier(**checkpoint["model"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    out = Path(args.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    if world > 1:
        dist.barrier()
    result = {"version": "2.4.1-B", "checkpoint": str(Path(args.checkpoint).resolve()), "splits": {}}

    for split, tag in (("train", "train"), ("validation", "val")):
        n, dim = meta["shapes"][tag]["7"]
        n, dim = int(n), int(dim)
        if int(label_meta["splits"][split]["rows"]) != n:
            raise ValueError(f"crop/label row mismatch {split}")
        x_path = features / "first" / "k7" / f"{tag}_X.f32"
        split_out = out / split
        if rank == 0:
            split_out.mkdir(parents=True, exist_ok=True)
            for head, size in checkpoint["model"]["head_sizes"].items():
                path = split_out / f"{head}.f32"
                with path.open("wb") as handle:
                    handle.truncate(n * int(size) * 4)
        if world > 1:
            dist.barrier()
        x = np.memmap(x_path, dtype=np.float32, mode="r", shape=(n, dim))
        outputs = {
            head: np.memmap(split_out / f"{head}.f32", dtype=np.float32, mode="r+", shape=(n, int(size)))
            for head, size in checkpoint["model"]["head_sizes"].items()
        }
        start, end = split_range(n, rank, world)
        with torch.inference_mode():
            for left in range(start, end, args.batch_size):
                right = min(end, left + args.batch_size)
                batch = torch.from_numpy(np.asarray(x[left:right]).copy()).to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    logits = model(batch)
                for head in HEADS:
                    outputs[head][left:right] = logits[head].float().cpu().numpy()
                if rank == 0 and (left - start) % (args.batch_size * 100) == 0:
                    print(f"[{split}] {left-start:,}/{end-start:,} rank0 rows", flush=True)
        for values in outputs.values():
            values.flush()
        if world > 1:
            dist.barrier()
        result["splits"][split] = {"rows": n, "head_sizes": checkpoint["model"]["head_sizes"]}

    if rank == 0:
        (out / "base_logits.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train the zero-initialized v2.4.1-B residual adapter with DDP."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler

HEADS = ("root", "euk", "prok", "organelle")


class ResidualAdapter(nn.Module):
    def __init__(self, dim_in=5376, hidden=(1024, 512), head_sizes=None, dropout=0.1):
        super().__init__()
        head_sizes = head_sizes or {"root": 3, "euk": 8, "prok": 2, "organelle": 2}
        layers, last = [], dim_in
        for width in hidden:
            layers += [nn.Linear(last, width), nn.GELU(), nn.Dropout(dropout)]
            last = width
        self.encoder = nn.Sequential(*layers)
        self.heads = nn.ModuleDict({name: nn.Linear(last, size) for name, size in head_sizes.items()})
        for head in self.heads.values():
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, features):
        encoded = self.encoder(features)
        return {name: head(encoded) for name, head in self.heads.items()}


class CropDataset(Dataset):
    def __init__(self, features, logits, label_manifest, split):
        tag = "train" if split == "train" else "val"
        meta = json.loads((Path(features) / "v241b_features.json").read_text())
        self.n = int(meta["shapes"][tag]["4"][0])
        self.features_root, self.logits_root = Path(features), Path(logits)
        labels = json.loads((Path(label_manifest) / "composite_features.json").read_text())
        shard = labels["splits"][split]["shards"]
        if len(shard) != 1 or int(shard[0]["rows"]) != self.n:
            raise ValueError(f"label contract mismatch for {split}")
        self.label_files = shard[0]["files"]
        self.split, self.tag = split, tag
        self._x = self._base = self._y = None

    def __len__(self): return self.n

    def _open(self):
        if self._x is not None: return
        self._x = {
            k: np.memmap(self.features_root / "first" / f"k{k}" / f"{self.tag}_X.f32",
                         dtype=np.float32, mode="r", shape=(self.n, 4**k))
            for k in (4, 5, 6)
        }
        sizes = {"root": 3, "euk": 8, "prok": 2, "organelle": 2}
        self._base = {
            head: np.memmap(self.logits_root / self.split / f"{head}.f32",
                            dtype=np.float32, mode="r", shape=(self.n, size))
            for head, size in sizes.items()
        }
        self._y = {head: np.memmap(self.label_files[head], dtype=np.int64, mode="r", shape=(self.n,)) for head in HEADS}

    def __getitem__(self, index):
        self._open()
        x = np.concatenate([self._x[k][index] for k in (4, 5, 6)]).astype(np.float32, copy=False)
        base = np.concatenate([self._base[h][index] for h in HEADS]).astype(np.float32, copy=False)
        y = np.asarray([self._y[h][index] for h in HEADS], dtype=np.int64)
        return torch.from_numpy(x.copy()), torch.from_numpy(base.copy()), torch.from_numpy(y.copy())


def unpack_base(values):
    return {"root": values[:, :3], "euk": values[:, 3:11], "prok": values[:, 11:13], "organelle": values[:, 13:15]}


def unpack_y(values): return {head: values[:, i] for i, head in enumerate(HEADS)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", required=True); p.add_argument("--base-logits", required=True)
    p.add_argument("--labels", required=True); p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=12); p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=24102); p.add_argument("--delta-l2", type=float, default=1e-4)
    args = p.parse_args(argv)
    rank, local_rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("LOCAL_RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank); dist.init_process_group("nccl"); device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(args.seed + rank)
    from tiara.hierarchical.model import MaskedHierarchicalLoss
    root_index = {"euk_nuclear": 0, "prok": 1, "organelle": 2}
    criterion = MaskedHierarchicalLoss(root_index)
    model = ResidualAdapter().to(device)
    # Release gate: before the first optimizer step, final logits must equal the
    # frozen base exactly in fp32. This prevents accidental non-zero adapters.
    probe = torch.zeros(2, 5376, device=device)
    with torch.no_grad():
        delta = model(probe)
        max_abs = max(float(value.abs().max()) for value in delta.values())
    if max_abs != 0.0: raise RuntimeError(f"zero-init equivalence failed: {max_abs}")
    if rank == 0: print("ZERO_INIT_EQUIVALENCE max_abs=0.0 PASS", flush=True)
    model = DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    datasets = {s: CropDataset(args.features, args.base_logits, args.labels, s) for s in ("train", "validation")}
    samplers = {s: DistributedSampler(datasets[s], num_replicas=world, rank=rank, shuffle=(s == "train"), seed=args.seed) for s in datasets}
    loaders = {s: DataLoader(datasets[s], batch_size=args.batch_size, sampler=samplers[s], num_workers=args.workers,
                             pin_memory=True, persistent_workers=args.workers > 0) for s in datasets}
    out = Path(args.out)
    if rank == 0: out.mkdir(parents=True, exist_ok=True)
    best, history = float("inf"), []
    for epoch in range(1, args.epochs + 1):
        samplers["train"].set_epoch(epoch); model.train(); train_sum = torch.zeros(2, device=device)
        for x, base_flat, y_flat in loaders["train"]:
            x, base_flat, y_flat = x.to(device, non_blocking=True), base_flat.to(device, non_blocking=True), y_flat.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                delta = model(x); base = unpack_base(base_flat); final = {h: base[h] + delta[h] for h in HEADS}
                loss, _ = criterion(final, unpack_y(y_flat))
                loss = loss + args.delta_l2 * sum(v.float().square().mean() for v in delta.values())
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            train_sum += torch.tensor([float(loss.detach()), 1.0], device=device)
        model.eval(); val_sum = torch.zeros(2, device=device)
        with torch.inference_mode():
            for x, base_flat, y_flat in loaders["validation"]:
                x, base_flat, y_flat = x.to(device, non_blocking=True), base_flat.to(device, non_blocking=True), y_flat.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    delta = model(x); base = unpack_base(base_flat); final = {h: base[h] + delta[h] for h in HEADS}
                    loss, _ = criterion(final, unpack_y(y_flat))
                val_sum += torch.tensor([float(loss), 1.0], device=device)
        dist.all_reduce(train_sum); dist.all_reduce(val_sum)
        train_loss, val_loss = float(train_sum[0] / train_sum[1]), float(val_sum[0] / val_sum[1])
        row = {"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss}; history.append(row)
        if rank == 0:
            print(json.dumps(row), flush=True)
            payload = {"version": "2.4.1-B", "epoch": epoch, "model": {"dim_in": 5376, "hidden": [1024, 512], "dropout": 0.1},
                       "state_dict": model.module.state_dict(), "history": history, "zero_init_equivalence": True}
            torch.save(payload, out / "last.pt")
            if val_loss < best:
                best = val_loss; torch.save(payload, out / "best.pt")
                (out / "training_history.json").write_text(json.dumps(history, indent=2) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__": main()

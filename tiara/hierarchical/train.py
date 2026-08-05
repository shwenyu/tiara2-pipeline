#!/usr/bin/env python3
"""Efficient single-/multi-GPU trainer for Tiara2 v2.3 hierarchical models.

Single GPU:
  python -m tiara.hierarchical.train --features ... --out ... --device cuda:0

8-GPU DDP (recommended):
  torchrun --standalone --nproc_per_node=8 -m tiara.hierarchical.train \
    --features ... --out ... --workers 8 --prefetch 4 --amp

Only rank 0 prints progress and writes checkpoints. The balanced sampler is
sharded across DDP ranks, so class balancing is retained in multi-GPU runs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler

try:
    from .schema import HierarchySchema
    from .model import HierarchicalClassifier, MaskedHierarchicalLoss
except ImportError:  # allow copying this file and running it directly
    from tiara.hierarchical.schema import HierarchySchema
    from tiara.hierarchical.model import HierarchicalClassifier, MaskedHierarchicalLoss

HEADS = ("root", "euk", "prok", "organelle")
_STOP = False


def _stop_handler(_sig, _frame):
    global _STOP
    _STOP = True


class MemmapDataset(Dataset):
    """Worker-safe lazy memmap dataset; no giant arrays are pickled."""

    def __init__(self, root, manifest):
        self.root = str(root)
        self.n = int(manifest["rows"])
        self.d = int(manifest["dim"])
        self._x = None
        self._y = None

    def _open(self):
        if self._x is None:
            root = Path(self.root)
            self._x = np.memmap(
                root / "X.f32", dtype=np.float32, mode="r", shape=(self.n, self.d)
            )
            self._y = {
                head: np.memmap(
                    root / f"{head}.i64", dtype=np.int64, mode="r", shape=(self.n,)
                )
                for head in HEADS
            }

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_x"] = None
        state["_y"] = None
        return state

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        self._open()
        x = torch.from_numpy(np.asarray(self._x[index]).copy())
        y = {head: torch.tensor(int(self._y[head][index])) for head in HEADS}
        return x, y

    def root_labels(self):
        path = Path(self.root) / "root.i64"
        return np.memmap(path, dtype=np.int64, mode="r", shape=(self.n,))


class DistributedWeightedSampler(Sampler[int]):
    """Deterministic class-balanced sampling, evenly sharded across DDP ranks."""

    def __init__(self, weights, num_replicas, rank, seed=42, drop_last=False):
        self.weights = torch.as_tensor(weights, dtype=torch.double, device="cpu")
        self.replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        if self.drop_last:
            self.total = (len(self.weights) // self.replicas) * self.replicas
        else:
            self.total = math.ceil(len(self.weights) / self.replicas) * self.replicas
        self.per_rank = self.total // self.replicas

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.per_rank

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights, self.total, replacement=True, generator=generator
        )
        return iter(indices[self.rank : self.total : self.replicas].tolist())


class Progress:
    def __init__(self, total, label, enabled=True, every=25):
        self.total = max(1, int(total))
        self.label = label
        self.enabled = enabled
        self.every = max(1, int(every))
        self.start = time.time()
        self.last = self.start

    def update(self, step, *, loss=None, device=None, force=False):
        if not self.enabled or (not force and step % self.every != 0):
            return
        now = time.time()
        elapsed = max(now - self.start, 1e-6)
        rate = step / elapsed
        eta = (self.total - step) / rate if rate > 0 else 0
        pct = 100.0 * step / self.total
        fields = [
            f"[{self.label}] {step:,}/{self.total:,} ({pct:5.1f}%)",
            f"{rate:6.2f} batch/s",
            f"elapsed={elapsed/60:6.1f}m",
            f"eta={eta/60:6.1f}m",
        ]
        if loss is not None:
            fields.append(f"loss={loss:.5f}")
        if device is not None and device.type == "cuda":
            used = torch.cuda.max_memory_allocated(device) / 1024**3
            fields.append(f"gpu_peak={used:.2f}GiB")
        print("  ".join(fields), flush=True)
        self.last = now


class Confusion:
    def __init__(self, classes, device):
        self.n = int(classes)
        self.mat = torch.zeros((self.n, self.n), dtype=torch.long, device=device)

    def add(self, truth, pred):
        valid = (truth >= 0) & (truth < self.n)
        encoded = truth[valid] * self.n + pred[valid]
        self.mat += torch.bincount(encoded, minlength=self.n * self.n).reshape(self.n, self.n)

    def sync(self):
        if dist.is_initialized():
            dist.all_reduce(self.mat, op=dist.ReduceOp.SUM)

    def macro_f1(self):
        m = self.mat.double()
        tp = m.diag()
        fp = m.sum(0) - tp
        fn = m.sum(1) - tp
        den = 2 * tp + fp + fn
        f1 = torch.where(den > 0, 2 * tp / den, torch.zeros_like(den))
        return float(f1.mean().cpu()), [float(x) for x in f1.cpu()]


def ddp_setup(device_arg=None):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP training requires CUDA/NCCL")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device("cuda", local_rank)
    elif device_arg:
        device = torch.device(device_arg)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return distributed, rank, local_rank, world, device


def make_loader(dataset, sampler, batch, workers, prefetch, pin, drop_last):
    kwargs = dict(
        dataset=dataset,
        batch_size=int(batch),
        sampler=sampler,
        num_workers=int(workers),
        pin_memory=bool(pin),
        drop_last=bool(drop_last),
    )
    if workers > 0:
        kwargs.update(
            persistent_workers=True,
            prefetch_factor=max(1, int(prefetch)),
        )
    return DataLoader(**kwargs)


def unwrap_model(model):
    """Return the plain nn.Module under DDP and/or torch.compile."""
    target = model.module if isinstance(model, DDP) else model
    return getattr(target, "_orig_mod", target)


def evaluate(model, loader, criterion, device, schema, rank, log_every):
    model.eval()
    confusion = Confusion(len(schema.profile.root), device)
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    rows = torch.zeros((), dtype=torch.long, device=device)
    progress = Progress(len(loader), "validation", rank == 0, log_every)
    with torch.inference_mode():
        for step, (x, y) in enumerate(loader, 1):
            x = x.to(device, non_blocking=True)
            y = {key: value.to(device, non_blocking=True) for key, value in y.items()}
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(x)
                loss, _ = criterion(logits, y)
            batch_n = x.shape[0]
            loss_sum += loss.detach().double() * batch_n
            rows += batch_n
            confusion.add(y["root"], logits["root"].argmax(1))
            progress.update(step, loss=float(loss.detach()), device=device)
    if dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(rows, op=dist.ReduceOp.SUM)
    confusion.sync()
    macro, per_class = confusion.macro_f1()
    progress.update(len(loader), loss=float(loss_sum / rows.clamp_min(1)), device=device, force=True)
    return {
        "loss": float((loss_sum / rows.clamp_min(1)).cpu()),
        "root_macro_f1": macro,
        "root_per_class_f1": per_class,
        "rows": int(rows.cpu()),
    }


def train(args):
    distributed, rank, local_rank, world, device = ddp_setup(args.device)
    is_main = rank == 0
    try:
        signal.signal(signal.SIGTERM, _stop_handler)
        signal.signal(signal.SIGINT, _stop_handler)
        seed = int(args.seed) + rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

        root = Path(args.features)
        feature_manifest = json.loads((root / "hierarchy_features.json").read_text())
        schema = HierarchySchema.from_dict(feature_manifest["schema"])
        train_manifest = feature_manifest["splits"]["train"]
        val_manifest = feature_manifest["splits"]["validation"]
        train_set = MemmapDataset(root / "train", train_manifest)
        val_set = MemmapDataset(root / "validation", val_manifest)

        root_y = np.asarray(train_set.root_labels())
        counts = np.bincount(root_y, minlength=len(schema.profile.root))
        class_weight = 1.0 / np.maximum(counts, 1)
        sample_weight = class_weight[root_y]
        if distributed:
            train_sampler = DistributedWeightedSampler(
                sample_weight, world, rank, seed=args.seed, drop_last=True
            )
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_set, num_replicas=world, rank=rank, shuffle=False, drop_last=False
            )
        else:
            train_sampler = WeightedRandomSampler(
                torch.as_tensor(sample_weight, dtype=torch.double),
                len(sample_weight), replacement=True
            )
            val_sampler = None

        workers = args.workers
        if workers < 0:
            workers = max(1, min(8, (os.cpu_count() or 8) // world))
        train_loader = make_loader(
            train_set, train_sampler, args.batch, workers, args.prefetch,
            device.type == "cuda", True,
        )
        val_loader = make_loader(
            val_set, val_sampler, args.val_batch or args.batch, workers,
            args.prefetch, device.type == "cuda", False,
        )

        head_sizes = {
            "root": len(schema.profile.root),
            "euk": len(schema.euk),
            "prok": len(schema.prok),
            "organelle": len(schema.organelle),
        }
        hidden = tuple(int(x) for x in args.hidden.split(",") if x.strip())
        model_config = {
            "dim_in": int(train_manifest["dim"]),
            "head_sizes": head_sizes,
            "hidden": list(hidden),
            "dropout": float(args.dropout),
        }
        model = HierarchicalClassifier(**model_config).to(device)
        if args.compile:
            model = torch.compile(model, mode=args.compile_mode)
        if distributed:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)

        loss_weights = {
            "root": args.loss_root,
            "euk": args.loss_euk,
            "prok": args.loss_prok,
            "organelle": args.loss_organelle,
        }
        criterion = MaskedHierarchicalLoss(
            schema.index("root"), loss_weights, label_smoothing=args.label_smoothing
        )
        optimizer_kwargs = dict(lr=args.lr, weight_decay=args.weight_decay)
        if device.type == "cuda":
            try:
                optimizer = torch.optim.AdamW(model.parameters(), fused=True, **optimizer_kwargs)
            except (TypeError, RuntimeError):
                optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
        scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

        out = Path(args.out)
        if is_main:
            out.mkdir(parents=True, exist_ok=True)
        if distributed:
            dist.barrier()

        start_epoch = 1
        best = -1.0
        history = []
        if args.resume:
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            target = unwrap_model(model)
            target.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            if checkpoint.get("scaler"):
                scaler.load_state_dict(checkpoint["scaler"])
            start_epoch = int(checkpoint["epoch"]) + 1
            best = float(checkpoint.get("best_root_macro_f1", -1))
            history = list(checkpoint.get("history", []))

        if is_main:
            print("[training configuration]", flush=True)
            print(f"  device/world       : {device} / {world}", flush=True)
            print(f"  train/validation   : {len(train_set):,} / {len(val_set):,}", flush=True)
            print(f"  dimensions         : {train_manifest['dim']:,}", flush=True)
            print(f"  batch per GPU      : {args.batch:,}", flush=True)
            print(f"  effective batch    : {args.batch * world * args.accumulate:,}", flush=True)
            print(f"  workers per rank   : {workers}", flush=True)
            print(f"  prefetch per worker: {args.prefetch}", flush=True)
            print(f"  AMP / TF32         : {args.amp} / {device.type == 'cuda'}", flush=True)
            print(f"  root counts        : {counts.tolist()}", flush=True)

        for epoch in range(start_epoch, args.epochs + 1):
            if hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)
            model.train()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            seen = 0
            epoch_start = time.time()
            progress = Progress(
                len(train_loader), f"epoch {epoch}/{args.epochs}", is_main, args.log_every
            )

            for step, (x, y) in enumerate(train_loader, 1):
                x = x.to(device, non_blocking=True)
                y = {key: value.to(device, non_blocking=True) for key, value in y.items()}
                sync_step = step % args.accumulate == 0 or step == len(train_loader)
                sync_context = (
                    model.no_sync() if isinstance(model, DDP) and not sync_step
                    else torch.enable_grad()
                )
                with sync_context:
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.float16,
                        enabled=args.amp and device.type == "cuda",
                    ):
                        logits = model(x)
                        loss, _parts = criterion(logits, y)
                        scaled_loss = loss / args.accumulate
                    scaler.scale(scaled_loss).backward()
                if sync_step:
                    if args.grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                batch_n = x.shape[0]
                loss_sum += float(loss.detach()) * batch_n
                seen += batch_n
                progress.update(step, loss=loss_sum / max(seen, 1), device=device)
                if _STOP:
                    break

            progress.update(
                len(train_loader), loss=loss_sum / max(seen, 1), device=device, force=True
            )
            validation = evaluate(
                model, val_loader, criterion, device, schema, rank, args.log_every
            )
            train_loss = loss_sum / max(seen, 1)
            if distributed:
                value = torch.tensor([loss_sum, seen], dtype=torch.float64, device=device)
                dist.all_reduce(value, op=dist.ReduceOp.SUM)
                train_loss = float(value[0] / value[1].clamp_min(1))

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                **validation,
                "seconds": round(time.time() - epoch_start, 3),
            }
            if is_main:
                history.append(row)
                print("[epoch complete] " + json.dumps(row, sort_keys=True), flush=True)
                target = unwrap_model(model)
                state = {
                    "epoch": epoch,
                    "state_dict": target.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "schema": schema.to_dict(),
                    "model": model_config,
                    "feature_manifest": feature_manifest,
                    "best_root_macro_f1": max(best, validation["root_macro_f1"]),
                    "version": schema.profile.version,
                    "history": history,
                    "world_size": world,
                }
                torch.save(state, out / "last_checkpoint.pt")
                if validation["root_macro_f1"] > best:
                    best = validation["root_macro_f1"]
                    state["best_root_macro_f1"] = best
                    torch.save(state, out / "hierarchical_model.pt")
                    print(f"[checkpoint] new best root macro-F1={best:.6f}", flush=True)
                (out / "training_history.json").write_text(
                    json.dumps(history, indent=2, sort_keys=True) + "\n"
                )
            if distributed:
                dist.barrier()
            if _STOP:
                if is_main:
                    print("[stop] signal received; last_checkpoint.pt was saved", flush=True)
                break
        return best
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=1024, help="batch size per GPU")
    p.add_argument("--val-batch", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden", default="2048,1024")
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--loss-root", type=float, default=1.0)
    p.add_argument("--loss-euk", type=float, default=1.0)
    p.add_argument("--loss-prok", type=float, default=0.5)
    p.add_argument("--loss-organelle", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=-1, help="workers per rank; -1=auto")
    p.add_argument("--prefetch", type=int, default=4, help="batches prefetched per worker")
    p.add_argument("--accumulate", type=int, default=1)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", help="single-process device, e.g. cuda:0 or cpu")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--compile-mode", default="reduce-overhead", choices=("default", "reduce-overhead", "max-autotune"))
    p.add_argument("--log-every", type=int, default=25, help="progress interval in batches")
    p.add_argument("--resume", help="path to last_checkpoint.pt")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.accumulate < 1:
        raise SystemExit("--accumulate must be >= 1")
    return train(args)


if __name__ == "__main__":
    main()

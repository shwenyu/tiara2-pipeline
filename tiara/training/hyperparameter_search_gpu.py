#!/usr/bin/env python3
"""GPU-parallel hyperparameter search for Tiara's two-stage classifier.

Drop-in, ~1-order-of-magnitude-faster replacement for
hyperparameter_search_first_stage.py / hyperparameter_search_second_stage.py.

What is IDENTICAL to the originals (so results stay comparable):
  * input FASTA file names + label mapping per stage
  * the exact architecture grid (same loops, same order)
  * TF-IDF k-mer features (2-bit rolling count == the old dict lookup)
  * skorch NeuralNetClassifier defaults (NLLLoss, Adam, 50 epochs, Softmax head)

What changed (the speedups):
  * every candidate trains on GPU (device=cuda:N) instead of CPU
  * candidates are spread across all GPUs and run concurrently
  * features are computed ONCE per k with an njit rolling k-mer counter and
    shared to workers via on-disk memmap (no 8x recompute, low RAM)
  * dropped the per-epoch EpochScoring(mean_f1) (it re-ran predict every epoch)
  * fixed the original val-normalization bug: train and eval both use the
    L2-normalized matrices consistently

Called once per k by 05_train.sh, e.g.:
    python -m tiara.training.hyperparameter_search_gpu \
        --stage first --k 6 --gpus 0,1,2,3,4,5,6,7 --max-parallel 8 \
        /data/shouhanyu/Tiara2/train_ready \
        /data/shouhanyu/Tiara2/log/train_v1.1/hp_first_k6.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import random
import shutil
import subprocess
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

# Keep each worker's BLAS/OpenMP footprint tiny; the GPU does the heavy lifting.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import multiprocessing as mp

import numpy as np
import torch
from Bio.SeqIO.FastaIO import SimpleFastaParser
from numba import njit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from skorch import NeuralNetClassifier
from torch import nn

import tiara
from tiara.src.transformations import TfidfWeighter
from tiara.training import featurize_cache


# --------------------------------------------------------------------------- #
# Stage definitions (files + labels are copied verbatim from the originals).
# First stage concatenates:  organelle, bacteria, archea, eukarya -> 0,1,3,4
# Second stage concatenates: plastids, mitochondria              -> 0,2
# --------------------------------------------------------------------------- #
STAGE_SPEC = {
    "first": {
        "files": ["organelle", "bacteria", "archaea", "eukarya"],
        "labels": [0, 1, 3, 4],
        "dim_out": 5,
        "idf": "first-stage",
    },
    "second": {
        "files": ["plastids", "mitochondria"],
        "labels": [0, 2],
        "dim_out": 3,
        "idf": "second-stage",
    },
}


def build_architectures(stage: str) -> list[dict[str, Any]]:
    """Architecture grid.

    v2.1.3: lr=0.1 is REMOVED from every sub-grid. With Adam + Softmax/NLL on
    4096-16384 TF-IDF dims, lr=0.1 reliably drives the net into the constant
    solution -- and on the v2.1.2 unbalanced validation split that constant
    solution scored 0.988 accuracy, so lr=0.1 candidates actually WON
    (first/k4 32-32 lr=0.1 and first/k6 256-128 lr=0.1). Removing the value
    also changes the candidate count, so old partial/complete jsons no longer
    match `search_signature` / `budget_fingerprint` and are re-searched.
    """
    architectures: list[dict[str, Any]] = []
    if stage == "first":
        for hid in [512, 1024, 2048]:
            for learning_rate in [0.001, 0.0001]:
                for dropout in [0.2]:
                    architectures.append(dict(hid1=hid, learning_rate=learning_rate, dropout=dropout))
                    architectures.append(dict(hid1=hid, hid2=hid, learning_rate=learning_rate, dropout=dropout))
        for learning_rate in [0.001, 0.0001]:
            for dropout in [0.2]:
                architectures.append(dict(hid1=512, hid2=256, learning_rate=learning_rate, dropout=dropout))
                architectures.append(dict(hid1=1024, hid2=512, learning_rate=learning_rate, dropout=dropout))
                architectures.append(dict(hid1=2048, hid2=1024, learning_rate=learning_rate, dropout=dropout))
        for hid in [32, 64, 128, 256]:
            for learning_rate in [0.01, 0.001, 0.0001]:
                for dropout in [0.2, 0.5]:
                    architectures.append(dict(hid1=hid, learning_rate=learning_rate, dropout=dropout))
                    architectures.append(dict(hid1=hid, hid2=hid, learning_rate=learning_rate, dropout=dropout))
        for learning_rate in [0.01, 0.001, 0.0001]:
            for dropout in [0.2, 0.5]:
                architectures.append(dict(hid1=64, hid2=32, learning_rate=learning_rate, dropout=dropout))
                architectures.append(dict(hid1=128, hid2=64, learning_rate=learning_rate, dropout=dropout))
                architectures.append(dict(hid1=128, hid2=64, learning_rate=learning_rate, dropout=dropout))
                architectures.append(dict(hid1=256, hid2=128, learning_rate=learning_rate, dropout=dropout))
    elif stage == "second":
        for hid in [32, 64, 128, 256]:
            for learning_rate in [0.01, 0.001, 0.0001]:
                for dropout in [0.2, 0.5]:
                    architectures.append(dict(hid1=hid, learning_rate=learning_rate, dropout=dropout))
                    architectures.append(dict(hid1=hid, hid2=hid, learning_rate=learning_rate, dropout=dropout))
        for learning_rate in [0.01, 0.001, 0.0001]:
            for dropout in [0.2, 0.5]:
                architectures.append(dict(hid1=64, hid2=32, learning_rate=learning_rate, dropout=dropout))
                architectures.append(dict(hid1=128, hid2=64, learning_rate=learning_rate, dropout=dropout))
                architectures.append(dict(hid1=256, hid2=128, learning_rate=learning_rate, dropout=dropout))
    else:
        raise ValueError(f"unknown stage: {stage}")
    return architectures


class MLP(nn.Sequential):
    """Same MLP as the originals: 1- or 2-hidden-layer, Dropout+ReLU, Softmax head."""

    def __init__(self, dim_in: int, hid1: int, hid2: int | None, dim_out: int, dropout: float):
        layers: list[nn.Module] = [nn.Linear(dim_in, hid1), nn.Dropout(dropout), nn.ReLU(inplace=True)]
        last = hid1
        if hid2:
            layers += [nn.Linear(hid1, hid2), nn.Dropout(dropout), nn.ReLU(inplace=True)]
            last = hid2
        layers += [nn.Linear(last, dim_out), nn.Softmax(1)]
        super().__init__(*layers)


@njit(cache=True)
def count_kmers_into(seq: np.ndarray, k: int, out: np.ndarray) -> None:
    """Rolling 2-bit ACGT k-mer counter. Index order == product('ACGT', repeat=k),
    so it matches the original dict-lookup indices and the stored IDF ordering.
    Windows spanning a non-ACGT base are skipped, exactly like the KeyError path.
    """
    mask = (1 << (2 * k)) - 1
    code = 0
    valid = 0
    for base in seq:
        if base == 65:       # A
            value = 0
        elif base == 67:     # C
            value = 1
        elif base == 71:     # G
            value = 2
        elif base == 84:     # T
            value = 3
        else:
            code = 0
            valid = 0
            continue
        code = ((code << 2) | value) & mask
        valid += 1
        if valid >= k:
            out[code] += 1.0


def read_fasta(path: Path) -> list[str]:
    with path.open() as handle:
        return [seq for _, seq in SimpleFastaParser(handle)]


def load_idf(stage: str, k: int, tfidf_dir: Path) -> np.ndarray:
    idf_dir = tfidf_dir / f"k{k}-{STAGE_SPEC[stage]['idf']}"
    if not idf_dir.exists():
        raise FileNotFoundError(f"Missing TF-IDF model: {idf_dir}")
    idf = np.asarray(TfidfWeighter.load_params(str(idf_dir)).idfs, dtype=np.float32)
    expected = 4 ** k
    if idf.shape[0] != expected:
        raise ValueError(f"IDF length {idf.shape[0]} != 4**{k} ({expected}) for {idf_dir}")
    return idf


def featurize(seqs: list[str], k: int, idf: np.ndarray, dim: int) -> np.ndarray:
    X = np.zeros((len(seqs), dim), dtype=np.float32)
    for i, seq in enumerate(seqs):
        raw = np.frombuffer(seq.encode("ascii", errors="ignore"), dtype=np.uint8)
        count_kmers_into(raw, k, X[i])
        if (i + 1) % 50000 == 0 or i + 1 == len(seqs):
            print(f"  features {i + 1:,}/{len(seqs):,}", flush=True)
    X *= idf
    norms = np.linalg.norm(X, axis=1)
    nz = norms > 0
    X[nz] /= norms[nz, None]
    return np.ascontiguousarray(X)


def read_group(input_dir: Path, split: str, name: str) -> list[str]:
    split_dir = input_dir / split
    if name == "organelle":
        explicit = split_dir / "organelle.fasta"
        if explicit.is_file():
            return read_fasta(explicit)
        return read_fasta(split_dir / "plastids.fasta") + read_fasta(split_dir / "mitochondria.fasta")
    path = split_dir / f"{name}.fasta"
    if name == "archaea" and not path.is_file():
        legacy = split_dir / "archea.fasta"
        if legacy.is_file():
            path = legacy
    return read_fasta(path)


def build_split(input_dir: Path, stage: str, split: str, k: int, idf: np.ndarray, dim: int):
    spec = STAGE_SPEC[stage]
    groups = [read_group(input_dir, split, name) for name in spec["files"]]
    counts = [len(g) for g in groups]
    flat = [s for g in groups for s in g]
    print(f"[{stage} k={k}] {split}: " + ", ".join(f"{n}={c}" for n, c in zip(spec["files"], counts)), flush=True)
    X = featurize(flat, k, idf, dim)
    y = np.concatenate([np.full(c, lbl, dtype=np.int64) for c, lbl in zip(counts, spec["labels"])])
    return X, y


# --------------------------------------------------------------------------- #
# Streaming featurization: peak RAM = one block, not the whole split.
# (Legacy featurize/build_split above are kept for compatibility but unused.)
# --------------------------------------------------------------------------- #
def group_paths(input_dir: Path, split: str, name: str) -> list[Path]:
    split_dir = input_dir / split
    if name == "organelle":
        explicit = split_dir / "organelle.fasta"
        if explicit.is_file():
            return [explicit]
        return [split_dir / "plastids.fasta", split_dir / "mitochondria.fasta"]
    path = split_dir / f"{name}.fasta"
    if name == "archaea" and not path.is_file():
        legacy = split_dir / "archea.fasta"
        if legacy.is_file():
            path = legacy
    return [path]


def iter_seqs(paths: list[Path]):
    for p in paths:
        with p.open() as handle:
            for _, seq in SimpleFastaParser(handle):
                yield seq


def count_seqs(paths: list[Path]) -> int:
    total = 0
    for p in paths:
        with p.open() as handle:
            for _ in SimpleFastaParser(handle):
                total += 1
    return total


def featurize_block(seqs: list[str], k: int, idf: np.ndarray, dim: int) -> np.ndarray:
    """L2-normalized TF-IDF features for a small block (peak RAM = block)."""
    X = np.zeros((len(seqs), dim), dtype=np.float32)
    for i, seq in enumerate(seqs):
        raw = np.frombuffer(seq.encode("ascii", errors="ignore"), dtype=np.uint8)
        count_kmers_into(raw, k, X[i])
    X *= idf
    norms = np.linalg.norm(X, axis=1)
    nz = norms > 0
    X[nz] /= norms[nz, None]
    return X


def build_split_to_memmap(input_dir, stage, split, k, idf, dim, scratch, tag, chunk):
    """Stream fasta -> features appended straight to an on-disk file.

    SINGLE pass over the fasta: featurize in blocks of `chunk` sequences and
    append each block's raw float32 bytes to <tag>_X.f32 (int64 labels to
    <tag>_y.i64). The row-major byte layout is identical to an np.memmap of
    shape (n, dim), which the workers open read-only. Peak RAM is O(chunk),
    and there is NO silent counting pass -- progress prints from the first
    block, so the step never looks frozen. Returns (shape, counts).
    """
    spec = STAGE_SPEC[stage]
    paths_per_group = [group_paths(input_dir, split, name) for name in spec["files"]]
    x_path = os.path.join(scratch, f"{tag}_X.f32")
    y_path = os.path.join(scratch, f"{tag}_y.i64")
    counts = [0] * len(paths_per_group)
    n = 0
    print(f"[{stage} k={k}] {split}: streaming features "
          f"(dim={dim}, chunk={chunk:,})", flush=True)

    with open(x_path, "wb") as xf, open(y_path, "wb") as yf:
        for gi, (paths, lbl) in enumerate(zip(paths_per_group, spec["labels"])):
            buf: list[str] = []

            def _flush() -> None:
                nonlocal n
                if not buf:
                    return
                block = np.ascontiguousarray(
                    featurize_block(buf, k, idf, dim), dtype=np.float32)
                xf.write(block.tobytes())
                yf.write(np.full(len(buf), lbl, dtype=np.int64).tobytes())
                counts[gi] += len(buf)
                n += len(buf)
                print(f"  features {n:,} (+{len(buf):,} {spec['files'][gi]})",
                      flush=True)
                buf.clear()

            for seq in iter_seqs(paths):
                buf.append(seq)
                if len(buf) >= chunk:
                    _flush()
            _flush()

    print(f"[{stage} k={k}] {split} done: "
          + ", ".join(f"{nm}={c}" for nm, c in zip(spec["files"], counts))
          + f", total={n:,}", flush=True)
    return (n, dim), counts


# --------------------------------------------------------------------------- #
# HP-search row subsetting: rank architectures on a small stratified subset.
# --------------------------------------------------------------------------- #
def _stratified_indices(y: np.ndarray, cap: int):
    """Deterministic, class-proportion-preserving row selection.

    Returns ASCENDING indices, or None when the data already fits `cap`.
    Within each class we take evenly spaced positions, so the subset spans the
    whole file (the cache stores one class per contiguous block) and each
    class keeps its ORIGINAL SHARE of rows -- only the absolute count shrinks.
    No RNG is involved, so a --resume reselects exactly the same rows.
    """
    n = int(y.size)
    cap = int(cap)
    if cap <= 0 or n <= cap:
        return None
    classes, counts = np.unique(y, return_counts=True)
    picks = []
    for cls, cnt in zip(classes, counts):
        cnt = int(cnt)
        take = int(round(cnt * (cap / n)))
        take = max(1, min(cnt, take))
        pos = np.flatnonzero(y == cls)
        sel = np.unique(np.linspace(0, cnt - 1, take).round().astype(np.int64))
        picks.append(pos[sel])
    out = np.concatenate(picks)
    out.sort()
    return out


def _balanced_indices(y: np.ndarray, cap: int):
    """Deterministic EQUAL-per-class row selection (ascending indices).

    Why this exists: with a proportional subset of a 98.6%-eukarya validation
    split, a model that answers "eukarya" to everything gets ~0.988 accuracy and
    a mean_f1 that still beats most honest candidates, so the HP search happily
    ranks a collapsed net first. Equal per-class rows make that impossible --
    a constant predictor scores 1/C recall and a low mean F1.

    Returns None when the data is already fine to use as-is (single class, or
    cap not set), never an empty selection.
    """
    n = int(y.size)
    cap = int(cap)
    classes, counts = np.unique(y, return_counts=True)
    if classes.size < 2:
        return None
    budget = n if cap <= 0 else min(cap, n)
    per_class = max(1, budget // int(classes.size))
    picks = []
    for cls, cnt in zip(classes, counts):
        cnt = int(cnt)
        take = min(cnt, per_class)
        pos = np.flatnonzero(y == cls)
        sel = np.unique(np.linspace(0, cnt - 1, take).round().astype(np.int64))
        picks.append(pos[sel])
    out = np.concatenate(picks)
    out.sort()
    if cap <= 0 and out.size == n:
        return None
    return out


def build_hp_subset(kdir, scratch, shapes, caps, log=None, balance=None):
    """Materialize a SMALL stratified subset of the feature cache into scratch.

    HP search only needs to RANK architectures -- the winning config is retrained
    on the full data by train_models_gpu. Ranking on a proportional subset gives
    the same ordering for a tiny fraction of the I/O, and (crucially) the small
    files fit in page cache, which removes the random-read storm that 8-40
    concurrent shuffling workers otherwise inflict on a single spinning disk.

    Falls back to symlinking the cache when a split already fits its cap, so the
    finally-block rmtree(scratch) never deletes the persistent cache.
    Returns the new {"train": (n, dim), "val": (n, dim)} shapes.
    """
    log = log or (lambda _m: None)
    new_shapes = {}
    for tag in ("train", "val"):
        n, dim = shapes[tag]
        cap = int(caps.get(tag, 0) or 0)
        xsrc = Path(kdir) / f"{tag}_X.f32"
        ysrc = Path(kdir) / f"{tag}_y.i64"
        xdst = os.path.join(scratch, f"{tag}_X.f32")
        ydst = os.path.join(scratch, f"{tag}_y.i64")
        y = np.array(np.memmap(ysrc, dtype=np.int64, mode="r", shape=(n,)))
        # `balance` may be a single mode or {tag: mode}. Default (None) keeps
        # the historical proportional behaviour for both splits.
        mode = balance.get(tag) if isinstance(balance, dict) else balance
        if str(mode or "proportional") == "equal":
            idx = _balanced_indices(y, cap)
        else:
            idx = _stratified_indices(y, cap)
        if idx is None:
            os.symlink(xsrc, xdst)
            os.symlink(ysrc, ydst)
            log(f"[hp-subset] {tag}: using all {n:,} rows (cap={cap:,})")
            new_shapes[tag] = (n, dim)
            continue
        m = int(idx.size)
        src = np.memmap(xsrc, dtype=np.float32, mode="r", shape=(n, dim))
        step = max(1, (1 << 22) // max(1, int(dim)))
        with open(xdst, "wb") as fx:
            for s in range(0, m, step):
                block = np.ascontiguousarray(src[idx[s:s + step]], dtype=np.float32)
                fx.write(block.tobytes())
        del src
        y_sub = np.asarray(y[idx], dtype=np.int64)
        y_sub.tofile(ydst)
        _, c_all = np.unique(y, return_counts=True)
        _, c_sub = np.unique(y_sub, return_counts=True)
        log(f"[hp-subset] {tag}: {n:,} -> {m:,} rows "
            f"({100.0 * m / max(1, n):.2f}%), dim={dim}, "
            f"{(m * int(dim) * 4) / (1 << 30):.1f} GiB written; "
            f"class share {np.round(c_all / max(1, n), 4).tolist()} -> "
            f"{np.round(c_sub / max(1, m), 4).tolist()}")
        new_shapes[tag] = (m, dim)
    return new_shapes


# --------------------------------------------------------------------------- #
# Worker: trains one candidate on one GPU, reading features from memmap.
# --------------------------------------------------------------------------- #
def train_one(task: tuple) -> dict[str, Any]:
    cpu_threads = int(
        os.environ.get("TIARA_HP_CPU_THREADS", "3")
    )

    torch.set_num_threads(cpu_threads)

    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    (idx, total, stage, k, arch, gpu_id, scratch, shapes, dim_out, epochs,
     batch_size, seed, class_weight_mode, max_pred_share) = task

    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    n_tr, dim = shapes["train"]
    n_va, _ = shapes["val"]
    # mode="c"（copy-on-write）：数组被标记为可写 → 消除 torch 的
    # "NumPy array is not writable" 警告；因为我们从不写入特征，页面仍按只读
    # 共享，内存不会翻倍。
    train_X = np.memmap(os.path.join(scratch, "train_X.f32"), dtype=np.float32, mode="c", shape=(n_tr, dim))
    val_X = np.memmap(os.path.join(scratch, "val_X.f32"), dtype=np.float32, mode="c", shape=(n_va, dim))
    # 标签很小，直接复制成可写数组即可（同样消除警告）。
    train_y = np.array(np.memmap(os.path.join(scratch, "train_y.i64"), dtype=np.int64, mode="r", shape=(n_tr,)))
    val_y = np.array(np.memmap(os.path.join(scratch, "val_y.i64"), dtype=np.int64, mode="r", shape=(n_va,)))

    hid1 = arch["hid1"]
    hid2 = arch.get("hid2")
    lr = arch["learning_rate"]
    drop = arch["dropout"]
    tag = f"hid1={hid1} hid2={hid2} lr={lr} drop={drop}"
    print(f"[{stage} k={k}] ({idx + 1}/{total}) START gpu={gpu_id} {tag}", flush=True)
    started = time.time()

    # Inverse-frequency class weights. The head is Softmax + NLLLoss (skorch's
    # default for NeuralNetClassifier), which accepts a `weight` tensor, so a
    # rare class stops being free to ignore.
    net_kwargs: dict[str, Any] = {}
    if str(class_weight_mode or "none") == "balanced":
        counts = np.bincount(train_y, minlength=dim_out).astype(np.float64)
        w = np.where(counts > 0, train_y.size / (dim_out * np.maximum(counts, 1)), 0.0)
        net_kwargs["criterion__weight"] = torch.tensor(
            w, dtype=torch.float32, device=device)
        print(f"[{stage} k={k}] class weights = {np.round(w, 4).tolist()}", flush=True)

    net = NeuralNetClassifier(
        MLP(dim, hid1, hid2, dim_out, drop),
        max_epochs=epochs,
        lr=lr,
        train_split=None,                 # train on all of train_X; eval val below
        iterator_train__shuffle=True,
        iterator_train__pin_memory=True,
        optimizer=torch.optim.Adam,
        device=device,
        batch_size=batch_size,
        verbose=0,
        **net_kwargs,
    )
    net.fit(np.asarray(train_X), train_y)
    y_pred = net.predict(np.asarray(val_X))

    accuracy = float(accuracy_score(val_y, y_pred))
    precision = precision_score(val_y, y_pred, average=None, zero_division=0).tolist()
    recall = recall_score(val_y, y_pred, average=None, zero_division=0).tolist()
    f1 = f1_score(val_y, y_pred, average=None, zero_division=0).tolist()
    mean_f1 = float(np.mean(f1))

    # COLLAPSE GUARD. A net that emits one class for (almost) every row is not
    # a model, whatever its score says. v2.1.2 shipped exactly this and nothing
    # in the pipeline noticed.
    pred_counts = np.bincount(np.asarray(y_pred, dtype=np.int64), minlength=dim_out)
    pred_share = float(pred_counts.max() / max(1, int(y_pred.size)))
    collapsed = bool(pred_share > float(max_pred_share))

    elapsed = time.time() - started
    flag = "  COLLAPSED" if collapsed else ""
    print(f"[{stage} k={k}] ({idx + 1}/{total}) DONE  gpu={gpu_id} "
          f"mean_f1={mean_f1:.4f} pred_share={pred_share:.4f}{flag} "
          f"in {elapsed/60:.1f} min", flush=True)

    return {
        "index": idx,
        "stage": stage,
        "k": k,
        "hid1": hid1,
        "hid2": hid2,
        "learning_rate": lr,
        "dropout": drop,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_f1": mean_f1,
        "pred_share": pred_share,
        "collapsed": collapsed,
        "gpu": gpu_id,
        "seconds": elapsed,
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp, path)


def search_signature(stage: str, k: int, epochs: int,
                     architectures: list[dict[str, Any]],
                     input_dir: Path, tfidf_dir: Path,
                     rows: tuple[int, int] | None = None) -> str:
    # `rows` (train, val) MUST be part of the signature: mean_f1 from a run on a
    # different number of rows is not comparable, so changing the HP row caps has
    # to invalidate an existing partial just like changing epochs does.
    payload = {"stage": stage, "k": k, "epochs": epochs,
               "architectures": architectures,
               "input_dir": str(input_dir), "tfidf_dir": str(tfidf_dir),
               "rows": [int(r) for r in rows] if rows else None}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


def budget_fingerprint(stage: str, args, tfidf_dir: Path, n_arch: int) -> dict:
    """What makes two COMPLETED searches comparable.

    Stored in the final json so a later run can tell "already done under the
    same budget" (skip) from "done under a different budget" (re-search). Only
    cheap, pure values -- it must be computable BEFORE any feature work.
    """
    return {"epochs": int(args.epochs),
            "hp_max_train_rows": int(args.hp_max_train_rows),
            "hp_max_val_rows": int(args.hp_max_val_rows),
            "architectures": int(n_arch),
            # v2.1.3: these three change what mean_f1 MEANS, so a search run
            # under different values must not be reused as "already done".
            "val_balance": str(args.val_balance),
            "class_weight": str(args.class_weight),
            "max_pred_share": float(args.max_pred_share),
            "tfidf_dir": str(tfidf_dir)}


def query_gpu_metrics(allowed: list[int]) -> list[dict[str, int]]:
    proc = subprocess.run([
        "nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], check=True, capture_output=True, text=True)
    allow = set(allowed)
    rows = []
    for line in proc.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 3:
            continue
        gpu, free_mib, util = map(int, fields)
        if gpu in allow:
            rows.append({"gpu": gpu, "free_mib": free_mib, "util": util})
    return rows


def dynamic_worker(task_base: tuple, gpu_id: int, result_queue: Any) -> None:
    (idx, total, stage, k, arch, scratch, shapes, dim_out, epochs, batch_size,
     seed, class_weight_mode, max_pred_share) = task_base
    task = (idx, total, stage, k, arch, gpu_id, scratch, shapes,
            dim_out, epochs, batch_size, seed, class_weight_mode,
            max_pred_share)
    try:
        result_queue.put({"ok": True, "result": train_one(task)})
    except BaseException as exc:
        result_queue.put({"ok": False, "gpu": gpu_id, "error": repr(exc),
                          "traceback": traceback.format_exc()})
        raise


def deduplicate_architectures(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result, seen = [], set()
    for item in items:
        key = json.dumps(item, sort_keys=True)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def parse_gpu_ids(value: str) -> list[int]:
    ids = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not ids or len(ids) != len(set(ids)):
        raise argparse.ArgumentTypeError("--gpus must be unique ids like 0,1,2,3")
    return ids


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input_dir", type=Path, help="train_ready dir (has train/ and validation/)")
    p.add_argument("output_file", type=Path, help="results json (a .txt summary is written next to it)")
    p.add_argument("--stage", required=True, choices=["first", "second"])
    p.add_argument("--k", required=True, type=int)
    p.add_argument("--tfidf-dir", required=True, type=Path,
                   help="versioned TF-IDF root, e.g. /data/.../tfidf_v2b")
    p.add_argument("--gpus", type=parse_gpu_ids, default=parse_gpu_ids("0,1,2,3,4,5,6,7"),
                   help="allowed GPU whitelist")
    p.add_argument("--max-parallel", type=int, default=6,
                   help="maximum candidates across all GPUs")
    p.add_argument("--min-free-mib", type=int, default=18000)
    p.add_argument("--max-gpu-util", type=int, default=30)
    p.add_argument("--max-tasks-per-gpu", type=int, default=2,
                   help="hard limit including shared candidates (default 2)")
    p.add_argument("--shared-min-free-mib", type=int, default=10000,
                   help="free VRAM required before adding a second task")
    p.add_argument("--shared-max-gpu-util", type=int, default=60,
                   help="utilization ceiling before adding a second task")
    p.add_argument("--share-launch-delay", type=float, default=30.0,
                   help="wait after first launch before evaluating that GPU for sharing")
    p.add_argument("--poll-seconds", type=float, default=15.0)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--scratch", type=Path, default=None)
    p.add_argument("--progress-secs", type=float, default=60.0,
                   help="seconds between scheduler heartbeat lines (0 disables)")
    p.add_argument("--hp-max-train-rows", type=int, default=0,
                   help="cap HP-search train rows (0 = use all); stratified, "
                        "class proportions preserved")
    p.add_argument("--hp-max-val-rows", type=int, default=0,
                   help="cap HP-search validation rows (0 = use all); ranking "
                        "only needs a fraction of the full validation split")
    p.add_argument("--val-balance", choices=["proportional", "equal"],
                   default="equal",
                   help="composition of the HP validation subset. 'equal' "
                        "takes the same number of rows per class so a constant "
                        "predictor cannot win the ranking (v2.1.2 default was "
                        "effectively 'proportional' on a 98.6%% eukarya split)")
    p.add_argument("--class-weight", choices=["none", "balanced"],
                   default="balanced",
                   help="inverse-frequency weighting of the training loss")
    p.add_argument("--max-pred-share", type=float, default=0.98,
                   help="a candidate predicting one class for more than this "
                        "share of validation rows is marked collapsed and can "
                        "never be selected as best")
    p.add_argument("--max-val-class-share", type=float, default=0.75,
                   help="abort if the HP validation subset is still dominated "
                        "by one class beyond this share")
    p.add_argument("--no-skip-complete", dest="skip_complete",
                   action="store_false", default=True,
                   help="re-run this k even if its results json is already "
                        "complete under the same budget")
    p.add_argument(
        "--feat-chunk",
        type=int,
        default=200_000,
        help="sequences per featurization block; peak feature-build RAM is "
             "O(feat-chunk x dim x feat-workers), independent of split size",
    )
    p.add_argument(
        "--feature-cache",
        type=Path,
        default=None,
        help="persistent feature cache root (reused across k and --resume). "
             "If omitted, a throw-away cache is built for this run only.",
    )
    p.add_argument(
        "--feat-workers",
        type=int,
        default=8,
        help="parallel processes for the feature builder (byte-range sharded)",
    )
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seq-pack", type=Path, default=None,
                   help="root for the shared 2-bit sequence pack (D2); when set, "
                        "features are derived from the compact pack so the "
                        "multi-TB FASTA is read only once and stays resumable")
    p.add_argument("--subsample", default=None,
                   help="JSON subsample config (see seqpack.normalize_subsample)")
    p.add_argument(
    "--cpu-threads-per-task",
    type=int,
    default=3,
    help="CPU threads available to each candidate process",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_file = args.output_file.expanduser().resolve()
    tfidf_dir = args.tfidf_dir.expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    stage, k = args.stage, args.k
    dim = 4 ** k

    # ---- k-level resume: skip a k whose search already COMPLETED ----------
    # `--resume` only consults the .partial.json, and that file is DELETED the
    # moment a k finishes. Without this check, a k that already produced a
    # complete results json gets searched again from scratch on every re-run of
    # the train stage (e.g. after a later k crashed).
    if args.skip_complete and output_file.is_file():
        try:
            done = json.loads(output_file.read_text())
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            done = None
        if isinstance(done, dict) and done.get("status") == "complete":
            n_arch = len(deduplicate_architectures(build_architectures(stage)))
            want = budget_fingerprint(stage, args, tfidf_dir, n_arch)
            got = done.get("budget")
            rows = done.get("results") or []
            best = max((r.get("mean_f1", 0.0) for r in rows), default=float("nan"))
            if got == want:
                print(f"[skip] {stage} k={k}: already complete "
                      f"({len(rows)} candidates, best mean_f1={best:.4f}) "
                      f"-> {output_file}", flush=True)
                return 0
            if got is None:
                # Written before budgets were recorded. We cannot PROVE it used
                # the current settings, so skip but say so loudly.
                print(f"[skip] WARNING {stage} k={k}: results json is complete "
                      f"but predates budget tracking ({len(rows)} candidates, "
                      f"best mean_f1={best:.4f}). Skipping. If it was produced "
                      f"with a DIFFERENT epochs/row-cap, delete {output_file} "
                      f"and re-run so all k are ranked on one budget.",
                      flush=True)
                return 0
            print(f"[resume] {stage} k={k}: results exist but the budget "
                  f"changed ({got} != {want}); re-searching.", flush=True)
    
    if args.cpu_threads_per_task < 1:
        raise ValueError("--cpu-threads-per-task must be >= 1")

    os.environ["TIARA_HP_CPU_THREADS"] = str(args.cpu_threads_per_task)

    for variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    ):
        os.environ[variable] = str(args.cpu_threads_per_task)
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this PyTorch environment")
    device_count = torch.cuda.device_count()
    invalid = [g for g in args.gpus if g < 0 or g >= device_count]
    if invalid:
        raise ValueError(f"Invalid GPU IDs {invalid}; torch sees {device_count} GPUs")
    if args.max_parallel < 1 or args.max_tasks_per_gpu < 1:
        raise ValueError("--max-parallel and --max-tasks-per-gpu must be >= 1")
    
    cpu_count = os.cpu_count() or 128

    # Grid Search 最多使用约 70% CPU；
    # 其余 CPU 留给已有任务、特征准备和系统。
    cpu_budget = int(cpu_count * 0.70)

    cpu_parallel_cap = max(
        1,
        cpu_budget // args.cpu_threads_per_task,
    )

    gpu_parallel_cap = (
        len(args.gpus) * args.max_tasks_per_gpu
    )

    max_parallel = min(
        args.max_parallel,
        cpu_parallel_cap,
        gpu_parallel_cap,
    )

    print(
        "Parallel limits: "
        f"requested={args.max_parallel}, "
        f"CPU cap={cpu_parallel_cap}, "
        f"GPU cap={gpu_parallel_cap}, "
        f"effective={max_parallel}",
        flush=True,
    )
    
    # ---- features (parallel, cached, reused across k) --------------------
    # The heavy read+parse+k-mer count is done by featurize_cache: sharded
    # across processes and written to a PERSISTENT cache, so a --resume never
    # recomputes them. When the train stage pre-builds all k in one pass, this
    # call is an instant cache hit.
    print(f"== [{stage} k={k}] computing features (dim={dim}) ==", flush=True)
    idf = load_idf(stage, k, tfidf_dir)

    scratch_parent = (args.scratch or output_file.parent).expanduser().resolve()
    scratch_parent.mkdir(parents=True, exist_ok=True)

    transient_cache = args.feature_cache is None
    if transient_cache:
        feat_root = Path(tempfile.mkdtemp(prefix=f"hpfeatcache_{stage}_k{k}_",
                                          dir=str(scratch_parent)))
    else:
        feat_root = args.feature_cache.expanduser().resolve()
        feat_root.mkdir(parents=True, exist_ok=True)

    subsample = json.loads(args.subsample) if args.subsample else None
    shapes_by_tag = featurize_cache.ensure_features(
        feat_root, input_dir, stage, [k], idf_map={k: idf},
        workers=args.feat_workers, chunk=args.feat_chunk,
        splits=("train", "validation"),
        seq_pack=(args.seq_pack.expanduser().resolve() if args.seq_pack else None),
        subsample=subsample,
        log=lambda m: print(m, flush=True))
    train_shape = shapes_by_tag["train"][k]
    val_shape = shapes_by_tag["val"][k]
    shapes = {"train": train_shape, "val": val_shape}

    # GPU workers expect canonical names train_X.f32 / val_X.f32 in `scratch`.
    # Materialize a SMALL stratified subset there (or symlink the cache when it
    # already fits the cap), so the finally-block rmtree(scratch) below never
    # deletes the persistent cache itself.
    kdir = feat_root / stage / f"k{k}"
    scratch = tempfile.mkdtemp(prefix=f"hpfeat_{stage}_k{k}_", dir=str(scratch_parent))
    shapes = build_hp_subset(
        kdir, scratch, shapes,
        {"train": args.hp_max_train_rows, "val": args.hp_max_val_rows},
        log=lambda m: print(m, flush=True),
        # Only the RANKING set is forced to equal classes. Train keeps the
        # corpus prior (already bp-balanced upstream) so the fit still sees
        # realistic composition.
        balance={"train": "proportional", "val": args.val_balance})

    # Hard gate: if the ranking set is still dominated by one class, every
    # score computed below is meaningless. Fail here rather than 26 GPU-hours
    # later.
    val_y_check = np.fromfile(os.path.join(scratch, "val_y.i64"), dtype=np.int64)
    if val_y_check.size:
        counts = np.bincount(val_y_check)
        share = float(counts.max() / val_y_check.size)
        print(f"[hp-subset] val class share = "
              f"{np.round(counts / val_y_check.size, 4).tolist()} "
              f"(max {share:.4f})", flush=True)
        if share > float(args.max_val_class_share):
            shutil.rmtree(scratch, ignore_errors=True)
            raise SystemExit(
                f"[FATAL] HP validation subset is {share:.2%} one class "
                f"(limit {args.max_val_class_share:.2%}). This is the v2.1.2 "
                f"failure mode: ranking on such a split rewards a constant "
                f"classifier. Re-run the bp-balance stage with "
                f"balance_splits including 'validation', or pass "
                f"--val-balance equal.")
    del val_y_check

    architectures = deduplicate_architectures(build_architectures(stage))
    total = len(architectures)
    spec = STAGE_SPEC[stage]
    print(f"== [{stage} k={k}] {total} candidates over GPUs {args.gpus}, "
          f"{max_parallel} at a time, batch={args.batch_size}, epochs={args.epochs} ==", flush=True)

    signature = search_signature(
        stage, k, args.epochs, architectures, input_dir, tfidf_dir,
        rows=(int(shapes["train"][0]), int(shapes["val"][0])))
    partial_file = output_file.with_suffix(output_file.suffix + ".partial.json")
    results: list[dict[str, Any]] = []
    if args.resume and partial_file.is_file():
        saved = json.loads(partial_file.read_text())
        if saved.get("signature") != signature:
            # STALE partial: left over from a run with different settings (epochs,
            # row caps, architecture grid, ...). Its mean_f1 values are NOT
            # comparable with the current budget. Discard it and re-search this k
            # instead of aborting -- otherwise one stale file kills the whole
            # multi-k sweep after earlier k values already succeeded.
            stale = partial_file.with_suffix(partial_file.suffix + ".stale")
            stale.unlink(missing_ok=True)
            partial_file.replace(stale)
            print(f"[resume] WARNING discarding stale partial for {stage} k={k} "
                  f"(signature {saved.get('signature')} != {signature}); "
                  f"re-searching from scratch. Old file kept at {stale}",
                  flush=True)
        else:
            results = list(saved.get("results", []))
            print(f"[resume] loaded {len(results)}/{total} candidates", flush=True)
    completed = {int(row["index"]) for row in results}
    pending = [(i, total, stage, k, arch, scratch, shapes, spec["dim_out"],
                args.epochs, args.batch_size, 20260715 + i,
                args.class_weight, args.max_pred_share)
               for i, arch in enumerate(architectures) if i not in completed]
    pending.sort(key=lambda t: -(int(t[4]["hid1"]) * int(t[4].get("hid2") or t[4]["hid1"])))

    started = time.time()
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    active: dict[int, dict[str, Any]] = {}
    failure: str | None = None
    last_wait_log = 0.0
    last_heartbeat = time.time()
    print(f"Dynamic scheduler: allowed={args.gpus} max_parallel={max_parallel} "
          f"exclusive=({args.min_free_mib}MiB,{args.max_gpu_util}%) "
          f"shared<= {args.max_tasks_per_gpu}/GPU "
          f"({args.shared_min_free_mib}MiB,{args.shared_max_gpu_util}%)", flush=True)

    try:
        while pending or active:
            for pid, record in list(active.items()):
                proc = record["process"]
                if not proc.is_alive():
                    proc.join()
                    print(f"[scheduler] RELEASE gpu={record['gpu']} pid={pid} exit={proc.exitcode}", flush=True)
                    if proc.exitcode != 0 and failure is None:
                        failure = f"candidate failed: pid={pid} gpu={record['gpu']} exit={proc.exitcode}"
                    del active[pid]

            while True:
                try:
                    message = result_queue.get_nowait()
                except queue.Empty:
                    break
                if message.get("ok"):
                    results.append(message["result"])
                    atomic_json(partial_file, {"status": "partial", "signature": signature,
                                               "stage": stage, "k": k, "results": results})
                    print(f"[{stage} k={k}] PROGRESS {len(results)}/{total} "
                          f"best_f1={max(x['mean_f1'] for x in results):.4f}", flush=True)
                elif failure is None:
                    failure = message.get("traceback") or message.get("error", "worker error")
            if failure:
                raise RuntimeError(failure)

            launched = False
            while pending and len(active) < max_parallel:
                metrics = query_gpu_metrics(args.gpus)
                now = time.time()
                counts = {gpu: 0 for gpu in args.gpus}
                oldest_launch = {gpu: now for gpu in args.gpus}
                for record in active.values():
                    gpu = record["gpu"]
                    counts[gpu] += 1
                    oldest_launch[gpu] = min(oldest_launch[gpu], record["launched_at"])

                # Always spread to empty GPUs first.
                exclusive = [row for row in metrics
                             if counts[row["gpu"]] == 0
                             and row["free_mib"] >= args.min_free_mib
                             and row["util"] <= args.max_gpu_util]
                exclusive.sort(
                    key=lambda row: (
                        row["util"],
                        -row["free_mib"],
                        row["gpu"],
                    )
                )

                shared = []
                if not exclusive and args.max_tasks_per_gpu > 1:
                    shared = [row for row in metrics
                              if 0 < counts[row["gpu"]] < args.max_tasks_per_gpu
                              and now - oldest_launch[row["gpu"]] >= args.share_launch_delay
                              and row["free_mib"] >= args.shared_min_free_mib
                              and row["util"] <= args.shared_max_gpu_util]
                    shared.sort(
                        key=lambda row: (
                            row["util"],
                            counts[row["gpu"]],
                            -row["free_mib"],
                            row["gpu"],
                        )
                    )

                candidates = exclusive or shared
                if not candidates:
                    if now - last_wait_log >= 60:
                        summary = ", ".join(
                            f"gpu{x['gpu']} tasks={counts[x['gpu']]} free={x['free_mib']}MiB util={x['util']}%"
                            for x in sorted(metrics, key=lambda x: x["gpu"]))
                        print(f"[scheduler] WAIT ({summary})", flush=True)
                        last_wait_log = now
                    break

                chosen = candidates[0]
                mode = "exclusive" if exclusive else "shared"
                task = pending.pop(0)
                proc = ctx.Process(target=dynamic_worker,
                                   args=(task, chosen["gpu"], result_queue))
                proc.start()
                active[proc.pid] = {"process": proc, "gpu": chosen["gpu"],
                                    "index": task[0], "launched_at": time.time()}
                print(f"[scheduler] LAUNCH {mode} candidate={task[0]+1}/{total} "
                      f"-> gpu={chosen['gpu']} slot={counts[chosen['gpu']]+1}/{args.max_tasks_per_gpu} "
                      f"free={chosen['free_mib']}MiB util={chosen['util']}% pid={proc.pid}", flush=True)
                launched = True

            # Heartbeat: when every slot is busy the launch loop above is never
            # entered, so WAIT never fires and the run would be SILENT for the
            # whole training window (epochs can take hours). Print a periodic
            # status line so progress is always observable.
            now_hb = time.time()
            if args.progress_secs > 0 and now_hb - last_heartbeat >= args.progress_secs:
                elapsed = max(now_hb - started, 1e-9)
                done_n = len(results)
                eta_text = "?"
                if done_n:
                    eta = (total - done_n) * (elapsed / done_n)
                    eta_text = f"{eta / 60:.1f}m"
                best_txt = (f"{max(x['mean_f1'] for x in results):.4f}"
                            if results else "n/a")
                act = ", ".join(
                    f"gpu{r['gpu']}:c{r['index'] + 1}"
                    for r in sorted(active.values(), key=lambda r: (r["gpu"], r["index"])))
                print(f"[{stage} k={k}] HEARTBEAT done={done_n}/{total} "
                      f"running={len(active)} pending={len(pending)} "
                      f"best_f1={best_txt} elapsed={elapsed / 60:.1f}m "
                      f"ETA={eta_text}"
                      + (f" active=[{act}]" if act else ""), flush=True)
                last_heartbeat = now_hb

            if pending or active:
                time.sleep(min(2.0, args.poll_seconds) if launched else args.poll_seconds)
    finally:
        for record in active.values():
            if record["process"].is_alive():
                record["process"].terminate()
        for record in active.values():
            record["process"].join(timeout=10)
        shutil.rmtree(scratch, ignore_errors=True)
        if transient_cache:
            shutil.rmtree(feat_root, ignore_errors=True)

    # Collapsed candidates are sorted to the BOTTOM regardless of score, so
    # `results[0]` (which train_models_gpu reads as "best") can never be a
    # constant predictor.
    results.sort(key=lambda r: (not bool(r.get("collapsed", False)),
                                r["mean_f1"]), reverse=True)
    if len(results) != total:
        raise RuntimeError(f"Search incomplete: expected {total}, got {len(results)}")
    n_collapsed = sum(1 for r in results if r.get("collapsed"))
    if n_collapsed:
        print(f"[collapse] {n_collapsed}/{total} candidates predicted a single "
              f"class for >{args.max_pred_share:.0%} of validation rows and "
              f"were demoted", flush=True)
    if n_collapsed == total:
        raise SystemExit(
            f"[FATAL] every {stage} k={k} candidate collapsed to one class. "
            f"The features or the balanced corpus are wrong -- do not train "
            f"final models on this.")
    atomic_json(output_file, {"status": "complete", "signature": signature,
                              "stage": stage, "k": k, "labels": spec["labels"],
                              "tfidf_dir": str(tfidf_dir),
                              "budget": budget_fingerprint(stage, args, tfidf_dir,
                                                           len(architectures)),
                              "results": results})
    partial_file.unlink(missing_ok=True)

    txt = output_file.with_suffix(".txt")
    with txt.open("w") as h:
        h.write(f"stage={stage} k={k} labels={spec['labels']}  ({total} candidates)\n")
        h.write("rank  mean_f1   acc     pshare  coll  hid1  hid2  lr        drop  gpu  min\n")
        for rank, r in enumerate(results, 1):
            h.write(f"{rank:>4}  {r['mean_f1']:.4f}  {r['accuracy']:.4f}  "
                    f"{r.get('pred_share', float('nan')):.4f}  "
                    f"{'Y' if r.get('collapsed') else 'n':>4}  "
                    f"{str(r['hid1']):>4}  {str(r['hid2']):>4}  {r['learning_rate']:<8}  "
                    f"{r['dropout']:<4}  {r['gpu']:>3}  {r['seconds']/60:.1f}\n")

    best = results[0]
    print(f"== [{stage} k={k}] BEST mean_f1={best['mean_f1']:.4f} "
          f"pred_share={best.get('pred_share', float('nan')):.4f} "
          f"hid1={best['hid1']} hid2={best['hid2']} lr={best['learning_rate']} drop={best['dropout']} ==", flush=True)
    print(f"== total wall time: {(time.time() - started)/60:.1f} min -> {output_file} ==", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

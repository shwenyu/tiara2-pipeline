#!/usr/bin/env python3
"""Train Tiara NNet models on GPUs with optimized HP JSON and model-level resume.

Examples
--------
Optimized v2.0 models from completed GPU searches::

    python -m tiara.training.train_models_gpu DATA_DIR OUTPUT_DIR 2 \
        --hp-dir /data/shouhanyu/Tiara2/log/train_v2b \
        --tfidf-dir /data/shouhanyu/Tiara2/tfidf_v2b \
        --gpus 0,1,2,3,4,5,6,7 --max-parallel 4 \
        --min-free-mib 18000 --max-gpu-util 30 --poll-seconds 15 \
        --batch-size 4096 --resume

Before every model launch, the scheduler re-queries nvidia-smi, filters the
allowed GPU whitelist, then ranks eligible cards by free VRAM descending and
GPU utilization ascending. A selected GPU stays reserved until that model exits.

If --hp-dir is omitted, the original fixed Tiara parameter tables are used.
The HP-search JSON does not contain per-epoch validation histories, so final
training uses --epochs (default: 50), matching the completed search duration.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import multiprocessing as mp
import os
import queue
import random
import shutil
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch
from Bio.SeqIO.FastaIO import SimpleFastaParser
from numba import njit
from skorch import NeuralNetClassifier
from torch import nn

import tiara
from tiara.src.transformations import TfidfWeighter

try:
    from . import seqpack
except (ImportError, ValueError):  # run as a plain script (sys.path injected)
    import seqpack

FIXED_FIRST_STAGE_PARAMS = [
    dict(k=4, hidden_1=2048, hidden_2=2048, lr=0.001, dropout=0.2, epochs=41),
    dict(k=5, hidden_1=2048, hidden_2=2048, lr=0.001, dropout=0.2, epochs=28),
    dict(k=6, hidden_1=2048, hidden_2=1024, lr=0.001, dropout=0.2, epochs=41),
]
FIXED_SECOND_STAGE_PARAMS = [
    dict(k=4, hidden_1=256, hidden_2=128, lr=0.001, dropout=0.2, epochs=45),
    dict(k=5, hidden_1=256, hidden_2=128, lr=0.001, dropout=0.2, epochs=37),
    dict(k=6, hidden_1=256, hidden_2=128, lr=0.001, dropout=0.5, epochs=30),
    dict(k=7, hidden_1=128, hidden_2=64, lr=0.01, dropout=0.2, epochs=47),
]
FILE_NAMES = {
    "mitochondria": "mitochondria_fr.fasta",
    "plastids": "plast_fr.fasta",
    "bacteria": "bacteria_fr.fasta",
    "eukarya": "eukarya_fr.fasta",
    "archaea": "archaea_fr.fasta",
}
_WORKER_GPU: int | None = None


class TiaraMLP(nn.Sequential):
    """Tiara MLP supporting both one- and two-hidden-layer search winners."""

    def __init__(self, dim_in: int, hid1: int, hid2: int | None,
                 dim_out: int, dropout: float):
        layers: list[nn.Module] = [
            nn.Linear(dim_in, hid1), nn.Dropout(dropout), nn.ReLU(inplace=True)
        ]
        last = hid1
        if hid2 is not None:
            layers.extend([
                nn.Linear(hid1, hid2), nn.Dropout(dropout), nn.ReLU(inplace=True)
            ])
            last = hid2
        layers.extend([nn.Linear(last, dim_out), nn.Softmax(1)])
        super().__init__(*layers)


@njit(cache=True)
def count_kmers_into(seq: np.ndarray, k: int, out: np.ndarray) -> None:
    mask = (1 << (2 * k)) - 1
    code = 0
    valid = 0
    for base in seq:
        if base == 65:
            value = 0
        elif base == 67:
            value = 1
        elif base == 71:
            value = 2
        elif base == 84:
            value = 3
        else:
            code = 0
            valid = 0
            continue
        code = ((code << 2) | value) & mask
        valid += 1
        if valid >= k:
            out[code] += 1.0


def read_fasta(path: Path, strict_acgt: bool = False) -> list[str]:
    seqs: list[str] = []
    with path.open() as handle:
        for _, seq in SimpleFastaParser(handle):
            seq = seq.upper()
            # Preserve the existing trainer's eukarya filtering behavior.
            if strict_acgt and not set(seq).issubset({"A", "C", "G", "T"}):
                continue
            seqs.append(seq)
    return seqs


def _subsample_seqs(seqs: list[str], spec: dict[str, Any] | None,
                    cls: str | None = None) -> list[str]:
    """Deterministically keep a subset of records.

    Uses the SAME content-hash rule as seqpack.keep_sequence (which the HP
    feature-cache path applies), so a record kept for HP is also kept here and
    vice versa -- the final models train on the same selection logic the search
    scored against. read_fasta already upper-cases, matching keep_sequence.

    `cls` selects the per-class rate when the train split is rebalanced. It is
    passed explicitly rather than derived from the filename because this path
    reads the abbreviated flat layout ('plast_fr.fasta'), and a silent miss here
    would train the final models on a different subset than HP scored.
    """
    if spec is None:
        return seqs
    keep = seqpack.keep_sequence
    rate = seqpack.rate_for(spec, cls) if cls else float(spec["rate"])
    return [s for s in seqs
            if keep(s, rate=rate, min_purity=spec["min_purity"],
                    min_len=spec["min_len"], seed=spec["seed"])]


def load_stage_sequences(data_dir: Path, stage: str,
                         spec: dict[str, Any] | None = None) -> tuple[list[str], np.ndarray]:
    mito = _subsample_seqs(read_fasta(data_dir / FILE_NAMES["mitochondria"]),
                           spec, "mitochondria")
    plast = _subsample_seqs(read_fasta(data_dir / FILE_NAMES["plastids"]),
                            spec, "plastids")
    if stage == "second":
        return plast + mito, np.asarray([0] * len(plast) + [2] * len(mito), dtype=np.int64)

    bacteria = _subsample_seqs(read_fasta(data_dir / FILE_NAMES["bacteria"]),
                               spec, "bacteria")
    archaea = _subsample_seqs(read_fasta(data_dir / FILE_NAMES["archaea"]),
                              spec, "archaea")
    eukarya = _subsample_seqs(
        read_fasta(data_dir / FILE_NAMES["eukarya"], strict_acgt=True),
        spec, "eukarya")
    seqs = plast + mito + bacteria + archaea + eukarya
    labels = (
        [0] * (len(plast) + len(mito))
        + [1] * len(bacteria)
        + [3] * len(archaea)
        + [4] * len(eukarya)
    )
    return seqs, np.asarray(labels, dtype=np.int64)


def idf_path(tfidf_dir: Path, stage: str, k: int) -> Path:
    return tfidf_dir / f"k{k}-{stage}-stage"


def make_tfidf(seqs: list[str], stage: str, k: int, tfidf_dir: Path) -> np.ndarray:
    path = idf_path(tfidf_dir, stage, k)
    if not path.exists():
        raise FileNotFoundError(f"Missing TF-IDF model: {path}")
    idf = np.asarray(TfidfWeighter.load_params(str(path)).idfs, dtype=np.float32)
    expected = 4 ** k
    if idf.shape[0] != expected:
        raise ValueError(f"IDF length {idf.shape[0]} != {expected} for {path}")

    print(f"[{stage} k={k}] computing TF-IDF for {len(seqs):,} sequences", flush=True)
    X = np.zeros((len(seqs), expected), dtype=np.float32)
    for i, seq in enumerate(seqs):
        raw = np.frombuffer(seq.encode("ascii", errors="ignore"), dtype=np.uint8)
        count_kmers_into(raw, k, X[i])
        if (i + 1) % 50000 == 0 or i + 1 == len(seqs):
            print(f"[{stage} k={k}] features {i + 1:,}/{len(seqs):,}", flush=True)
    X *= idf
    norms = np.linalg.norm(X, axis=1)
    nz = norms > 0
    X[nz] /= norms[nz, None]
    if not np.all(nz):
        print(f"[{stage} k={k}] warning: {(~nz).sum()} zero-norm rows", flush=True)
    return np.ascontiguousarray(X)


# --------------------------------------------------------------------------- #
# Feature-cache training path.
#
# The historical path (load_stage_sequences + make_tfidf) materializes a dense
# float32 matrix of EVERY training row in RAM: X = np.zeros((n, 4**k)). At this
# corpus size that is ~770 GiB for first/k=6 and ~760 GiB for second/k=7, so the
# kernel OOM-killer reaps the workers (exit=-9) before training even starts.
#
# featurize_cache already wrote those matrices to disk, so instead we take a
# deterministic, class-proportional subset of the cache -- the SAME selection
# rule the HP search ranked architectures on -- and stream it from a memmap
# whenever it is still too large to hold in RAM.
# --------------------------------------------------------------------------- #
def _cache_kdir(feature_cache: str | Path, stage: str, k: int) -> Path:
    return Path(feature_cache) / stage / f"k{k}"


def _cache_shape(kdir: Path, tag: str = "train") -> tuple[int, int]:
    meta_path = Path(kdir) / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing feature-cache meta: {meta_path}")
    meta = json.loads(meta_path.read_text())
    entry = meta.get(tag)
    if not entry:
        raise KeyError(
            f"Feature cache {kdir} has no '{tag}' entry (found {sorted(meta)})")
    return int(entry["n"]), int(entry["dim"])


class _ConcatMemmap:
    """Read-only row-wise concatenation of feature memmaps.

    This keeps train and validation as separate on-disk cache files while
    presenting them as one matrix to the existing capped/scratch builder.  It
    deliberately materialises only the requested block, never the complete
    train+validation matrix.
    """

    def __init__(self, arrays: list[np.memmap]):
        if not arrays:
            raise ValueError("at least one feature memmap is required")
        dims = {int(a.shape[1]) for a in arrays}
        if len(dims) != 1:
            raise ValueError(f"feature-cache dimensions disagree: {sorted(dims)}")
        self.arrays = arrays
        self.offsets = np.cumsum([0] + [int(a.shape[0]) for a in arrays],
                                 dtype=np.int64)
        self.shape = (int(self.offsets[-1]), int(arrays[0].shape[1]))

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(self.shape[0])
            idx = np.arange(start, stop, step, dtype=np.int64)
        else:
            idx = np.asarray(key, dtype=np.int64)
            if idx.ndim == 0:
                idx = idx.reshape(1)
        if idx.size == 0:
            return np.empty((0, self.shape[1]), dtype=np.float32)
        if int(idx.min()) < 0 or int(idx.max()) >= self.shape[0]:
            raise IndexError("feature-cache row index out of range")
        out = np.empty((idx.size, self.shape[1]), dtype=np.float32)
        for part, src in enumerate(self.arrays):
            lo, hi = int(self.offsets[part]), int(self.offsets[part + 1])
            mask = (idx >= lo) & (idx < hi)
            if np.any(mask):
                out[mask] = src[idx[mask] - lo]
        return out


def _stratified_picks(y: np.ndarray, cap: int) -> list[np.ndarray] | None:
    """Per-class ASCENDING row indices preserving each class's original share.

    Mirrors hyperparameter_search_gpu._stratified_indices so the final models
    train on the same selection rule the search ranked on. Returns None when the
    data already fits `cap`. No RNG is involved, so --resume reselects exactly
    the same rows.
    """
    n = int(y.size)
    cap = int(cap)
    if cap <= 0 or n <= cap:
        return None
    classes, counts = np.unique(y, return_counts=True)
    picks: list[np.ndarray] = []
    for cls, cnt in zip(classes, counts):
        cnt = int(cnt)
        take = max(1, min(cnt, int(round(cnt * (cap / n)))))
        pos = np.flatnonzero(y == cls)
        sel = np.unique(np.linspace(0, cnt - 1, take).round().astype(np.int64))
        picks.append(pos[sel])
    return picks


def _share(y: np.ndarray) -> list[float]:
    if y.size == 0:
        return []
    _, counts = np.unique(y, return_counts=True)
    return np.round(counts / y.size, 4).tolist()


def build_training_matrix(kdir, scratch, cap: int, ram_budget: int,
                          seed: int = 20260728, log=None,
                          progress_secs: float = 60.0,
                          tags: tuple[str, ...] = ("train",)):
    """Return (X, y, shuffle_ok) for a capped subset of feature caches.

    ``tags=("train", "val")`` performs the final refit on both splits without
    creating the old multi-TB flat FASTA or a second persistent feature cache.

    Two regimes:

    * subset fits `ram_budget` -> gather it into RAM (block reads, no scratch
      copy) and let the DataLoader shuffle normally.
    * subset does not fit -> write a scratch copy and return a memmap. The copy
      is built one class-balanced window at a time: within a window we read
      ASCENDING source rows (sequential on a spinning disk) and then permute
      them before writing, so the on-disk order is already shuffled and the
      loader can stream it with shuffle=False.
    """
    log = log or (lambda _m: None)
    kdir = Path(kdir)
    tags = tuple(tags)
    if not tags:
        raise ValueError("tags must not be empty")
    dims, arrays, labels, counts = set(), [], [], []
    for tag in tags:
        n_tag, dim_tag = _cache_shape(kdir, tag)
        dims.add(dim_tag)
        xsrc = kdir / f"{tag}_X.f32"
        ysrc = kdir / f"{tag}_y.i64"
        for path in (xsrc, ysrc):
            if not path.is_file():
                raise FileNotFoundError(f"Missing feature cache file: {path}")
        expect = n_tag * dim_tag * 4
        actual = xsrc.stat().st_size
        if actual != expect:
            raise ValueError(
                f"{xsrc}: {actual} bytes != {n_tag}x{dim_tag}x4 = {expect}")
        y_tag = np.fromfile(ysrc, dtype=np.int64)
        if y_tag.size != n_tag:
            raise ValueError(
                f"{ysrc}: {y_tag.size} labels != {n_tag} rows in meta.json")
        arrays.append(np.memmap(xsrc, dtype=np.float32, mode="r",
                                shape=(n_tag, dim_tag)))
        labels.append(y_tag)
        counts.append(n_tag)
    if len(dims) != 1:
        raise ValueError(f"feature-cache dimensions disagree for {tags}: {sorted(dims)}")
    dim = dims.pop()
    n = sum(counts)
    y_all = labels[0] if len(labels) == 1 else np.concatenate(labels)
    src = _ConcatMemmap(arrays)

    picks = _stratified_picks(y_all, cap)
    idx = None if picks is None else np.sort(np.concatenate(picks))
    m = n if idx is None else int(idx.size)
    need = m * dim * 4
    detail = "+".join(f"{tag}:{count:,}" for tag, count in zip(tags, counts))
    log(f"{n:,} cached rows ({detail}) -> {m:,} selected (cap={cap:,}), dim={dim}, "
        f"{need / (1 << 30):.1f} GiB vs budget {ram_budget / (1 << 30):.1f} GiB")
    step = max(1, (1 << 26) // max(1, dim * 4))

    if need <= ram_budget:
        X = np.empty((m, dim), dtype=np.float32)
        t0 = time.time()
        tick = t0
        for s in range(0, m, step):
            X[s:s + step] = src[s:s + step] if idx is None else src[idx[s:s + step]]
            # Pulling tens of GiB off a spinning disk takes many minutes and used
            # to be completely silent, so the step looked hung. Throttled (not a
            # \r bar) because up to 7 workers share one stdout pipe.
            now = time.time()
            if progress_secs > 0 and now - tick >= progress_secs:
                done = min(s + step, m)
                gib = done * dim * 4 / (1 << 30)
                secs = max(1e-9, now - t0)
                eta = (m - done) / max(1, done) * secs
                log(f"load {100.0 * done / max(1, m):5.1f}% | "
                    f"{done:,}/{m:,} rows | {gib:.1f} GiB | "
                    f"{gib * 1024 / secs:.0f} MiB/s | ETA {eta / 60:.1f} min")
                tick = now
        y = y_all if idx is None else np.ascontiguousarray(y_all[idx])
        del src
        log(f"held in RAM; class share {_share(y_all)} -> {_share(y)}")
        return X, y, True

    os.makedirs(scratch, exist_ok=True)
    xdst = os.path.join(scratch, "train_X.f32")
    ydst = os.path.join(scratch, "train_y.i64")
    if picks is None:
        picks = [np.flatnonzero(y_all == c) for c in np.unique(y_all)]
    win = max(step, (1 << 31) // max(1, dim * 4))
    nwin = max(1, -(-m // win))
    per = [max(1, -(-int(p.size) // nwin)) for p in picks]
    rng = np.random.default_rng(seed)
    written = 0
    t0 = time.time()
    tick = t0
    log(f"streaming {need / (1 << 30):.1f} GiB to {xdst} "
        f"in {nwin} shuffled windows")
    with open(xdst, "wb") as fx, open(ydst, "wb") as fy:
        for w in range(nwin):
            parts = [p[w * per[ci]:min((w + 1) * per[ci], p.size)]
                     for ci, p in enumerate(picks)
                     if w * per[ci] < p.size]
            if not parts:
                continue
            take = np.concatenate(parts)
            take.sort()  # ascending -> sequential reads inside the window
            block = np.ascontiguousarray(src[take], dtype=np.float32)
            yb = np.ascontiguousarray(y_all[take], dtype=np.int64)
            perm = rng.permutation(take.size)  # shuffle AFTER reading
            fx.write(block[perm].tobytes())
            fy.write(yb[perm].tobytes())
            written += int(take.size)
            del block
            now = time.time()
            if progress_secs > 0 and now - tick >= progress_secs:
                gib = written * dim * 4 / (1 << 30)
                secs = max(1e-9, now - t0)
                eta = (m - written) / max(1, written) * secs
                log(f"write {100.0 * written / max(1, m):5.1f}% | "
                    f"win {w + 1}/{nwin} | {written:,}/{m:,} rows | "
                    f"{gib:.1f} GiB | {gib * 1024 / secs:.0f} MiB/s | "
                    f"ETA {eta / 60:.1f} min")
                tick = now
    del src
    y = np.fromfile(ydst, dtype=np.int64)
    X = np.memmap(xdst, dtype=np.float32, mode="r", shape=(written, dim))
    log(f"streamed {written:,} rows to {xdst} "
        f"({written * dim * 4 / (1 << 30):.1f} GiB, {nwin} shuffled windows); "
        f"class share {_share(y_all)} -> {_share(y)}")
    return X, y, False


def hp_file(hp_dir: Path, stage: str, k: int) -> Path:
    return hp_dir / f"hp_{stage}_k{k}.json"


def load_hp_payload(path: Path, stage: str, k: int) -> dict[str, Any]:
    """Load the last valid HP object, tolerating concatenated/trailing data.

    Some resumed historical runs left more than one JSON document (or a small
    trailing fragment) in one .json file. json.load() rejects that with
    JSONDecodeError: Extra data. Decode documents incrementally and select the
    last complete object matching stage/k instead.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    decoder = json.JSONDecoder()
    pos = 0
    candidates: list[dict[str, Any]] = []
    while pos < len(text):
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text):
            break
        try:
            obj, end = decoder.raw_decode(text, pos)
        except json.JSONDecodeError as exc:
            if candidates:
                print(f"[warning] ignoring trailing non-JSON data in {path} at char {pos}: {exc}", flush=True)
                break
            raise ValueError(f"Cannot parse HP JSON {path}: {exc}") from exc
        if isinstance(obj, dict):
            try:
                matches = obj.get("stage") == stage and int(obj.get("k", -1)) == k
            except (TypeError, ValueError):
                matches = False
            if matches and isinstance(obj.get("results"), list) and obj["results"]:
                candidates.append(obj)
        pos = end
    if not candidates:
        raise ValueError(f"No complete stage={stage} k={k} result object in {path}")
    if len(candidates) > 1:
        print(f"[warning] {path} contains {len(candidates)} valid JSON documents; using the last one", flush=True)
    return candidates[-1]


def load_best_hp(hp_dir: Path, stage: str, k: int, epochs: int,
                 min_mean_f1: float = 0.0) -> dict[str, Any]:
    """Pick the best NON-COLLAPSED candidate, and refuse a bad one.

    v2.1.2 took `max(results, key=mean_f1)` unconditionally. On a 98.6%-eukarya
    ranking split the winners were lr=0.1 nets that answered "eukarya" to
    everything, and those hyper-parameters were then used for the FINAL models.
    Two guards now stand here: collapsed candidates are not eligible at all,
    and the winner must clear `min_mean_f1`.
    """
    path = hp_file(hp_dir, stage, k)
    if not path.is_file():
        raise FileNotFoundError(f"Missing HP result: {path}")
    payload = load_hp_payload(path, stage, k)
    results = payload["results"]
    healthy = [r for r in results if not r.get("collapsed")]
    dropped = len(results) - len(healthy)
    if not healthy:
        raise RuntimeError(
            f"All {len(results)} HP candidates in {path} collapsed to a single "
            f"class. Refusing to train final models on them. Re-run the bp "
            f"balance stage (validation must be balanced) and the HP search.")
    if dropped:
        print(f"[hp] {stage} k={k}: ignored {dropped} collapsed candidate(s)",
              flush=True)
    best = max(healthy, key=lambda row: float(row["mean_f1"]))
    if float(best["mean_f1"]) < float(min_mean_f1):
        raise RuntimeError(
            f"Best {stage} k={k} candidate scores mean_f1="
            f"{float(best['mean_f1']):.4f}, below the required "
            f"{float(min_mean_f1):.4f} ({path}). Something upstream is wrong; "
            f"publishing this would repeat v2.1.2.")
    return {
        "k": k,
        "hidden_1": int(best["hid1"]),
        "hidden_2": None if best.get("hid2") is None else int(best["hid2"]),
        "lr": float(best["learning_rate"]),
        "dropout": float(best["dropout"]),
        "epochs": epochs,
        "validation_mean_f1": float(best["mean_f1"]),
        "validation_pred_share": float(best.get("pred_share", float("nan"))),
        "hp_source": str(path),
    }


def build_jobs(hp_dir: Path | None, epochs: int,
               min_mean_f1: float = 0.0,
               k_first: tuple[int, ...] = (4, 5, 6),
               k_second: tuple[int, ...] = (4, 5, 6, 7)) -> list[tuple[str, dict[str, Any]]]:
    if hp_dir is None:
        return ([('first', dict(x)) for x in FIXED_FIRST_STAGE_PARAMS]
                + [('second', dict(x)) for x in FIXED_SECOND_STAGE_PARAMS])
    jobs: list[tuple[str, dict[str, Any]]] = []
    for k in k_first:
        jobs.append(("first", load_best_hp(hp_dir, "first", k, epochs,
                                           min_mean_f1)))
    for k in k_second:
        # Stage 2 only ever sees reads stage 1 already called organellar, so
        # the stage-1 floor does not apply to it.
        jobs.append(("second", load_best_hp(hp_dir, "second", k, epochs)))
    return jobs


def fmt_value(value: Any) -> str:
    if value is None:
        return "none"
    return str(value)


def model_name(stage: str, arch: dict[str, Any]) -> str:
    # Stage prefix prevents accidental first/second-stage filename collisions.
    fields = [
        ("k", arch["k"]), ("hidden_1", arch["hidden_1"]),
        ("hidden_2", arch.get("hidden_2")), ("lr", arch["lr"]),
        ("dropout", arch["dropout"]), ("epochs", arch["epochs"]),
    ]
    return stage + "_" + "_".join(f"{k}-{fmt_value(v)}" for k, v in fields) + ".pkl"


def is_complete(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def init_worker(gpu_queue: Any, cpu_threads: int) -> None:
    global _WORKER_GPU
    _WORKER_GPU = int(gpu_queue.get())
    torch.set_num_threads(cpu_threads)
    torch.cuda.set_device(_WORKER_GPU)
    print(f"[worker pid={os.getpid()}] assigned GPU {_WORKER_GPU}", flush=True)


def _collapse_probe(net, feature_cache, stage: str, k: int, X, y, dim_out: int,
                    log=print, cap: int = 200_000):
    """(max predicted-class share, probe rows) on a CLASS-BALANCED probe.

    Prefers the held-out validation cache; falls back to a balanced subset of
    the training rows when there is no cache (fixed-parameter path). Returns
    (None, 0) when no usable probe exists, so this can never fail a run for
    infrastructure reasons -- it only ever fails a genuinely collapsed model.
    """
    try:
        Xp = yp = None
        if feature_cache:
            kdir = _cache_kdir(feature_cache, stage, k)
            shape = _cache_shape(kdir, "val")
            if shape:
                n, dim = shape
                yv = np.array(np.memmap(Path(kdir) / "val_y.i64",
                                        dtype=np.int64, mode="r", shape=(n,)))
                idx = _balanced_probe_indices(yv, cap, dim_out)
                if idx is not None and idx.size:
                    Xv = np.memmap(Path(kdir) / "val_X.f32", dtype=np.float32,
                                   mode="r", shape=(n, dim))
                    Xp = np.ascontiguousarray(Xv[idx], dtype=np.float32)
                    yp = yv[idx]
                    del Xv
        if Xp is None:
            ya = np.asarray(y, dtype=np.int64)
            idx = _balanced_probe_indices(ya, cap, dim_out)
            if idx is None or not idx.size:
                return None, 0
            Xp = np.ascontiguousarray(np.asarray(X)[idx], dtype=np.float32)
            yp = ya[idx]
        pred = np.asarray(net.predict(Xp), dtype=np.int64)
        counts = np.bincount(pred, minlength=dim_out)
        return float(counts.max() / max(1, pred.size)), int(pred.size)
    except Exception as exc:                       # never mask a real failure
        log(f"sanity probe unavailable ({exc!r}); skipping collapse check")
        return None, 0


def _balanced_probe_indices(y: np.ndarray, cap: int, dim_out: int):
    """Ascending indices holding (about) the same number of rows per class."""
    classes, counts = np.unique(y, return_counts=True)
    if classes.size < 2:
        return None
    per_class = max(1, int(cap) // int(classes.size))
    picks = []
    for cls, cnt in zip(classes, counts):
        pos = np.flatnonzero(y == cls)
        take = min(int(cnt), per_class)
        sel = np.unique(np.linspace(0, int(cnt) - 1, take).round().astype(np.int64))
        picks.append(pos[sel])
    out = np.concatenate(picks)
    out.sort()
    return out


def train_one(task: tuple[int, str, dict[str, Any], str, str, str, int, Any]) -> dict[str, Any]:
    task_index, stage, arch, data_dir_s, output_dir_s, tfidf_dir_s, batch_size, spec, opts = task
    opts = opts or {}
    assert _WORKER_GPU is not None
    gpu_id = _WORKER_GPU
    data_dir = Path(data_dir_s)
    output_dir = Path(output_dir_s)
    tfidf_dir = Path(tfidf_dir_s)
    output_path = output_dir / model_name(stage, arch)
    tmp_path = output_dir / f".{output_path.name}.pid{os.getpid()}.tmp"

    seed = 20260716 + task_index
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    label = f"{stage} k={arch['k']} gpu={gpu_id}"
    print(f"[{label}] START -> {output_path.name}", flush=True)
    started = time.time()
    job_scratch: str | None = None
    try:
        k = int(arch["k"])
        feature_cache = opts.get("feature_cache")
        if feature_cache:
            # Reuse the matrices featurize_cache already wrote: no re-featurizing
            # and, crucially, no full-corpus dense matrix in RAM.
            job_scratch = os.path.join(
                opts.get("scratch_root") or str(output_dir / ".scratch"),
                f"{stage}_k{k}_pid{os.getpid()}")
            X, y, shuffle = build_training_matrix(
                _cache_kdir(feature_cache, stage, k), job_scratch,
                int(opts.get("cap", 0) or 0), int(opts.get("ram_budget", 0) or 0),
                log=lambda message: print(f"[{label}] {message}", flush=True),
                progress_secs=float(opts.get("progress_secs", 60.0) or 0.0),
                tags=(("train", "val") if opts.get("include_validation")
                      else ("train",)))
        else:
            seqs, y = load_stage_sequences(data_dir, stage, spec)
            X = make_tfidf(seqs, stage, k, tfidf_dir)
            del seqs
            shuffle = True
        dim_out = 5 if stage == "first" else 3

        # Inverse-frequency class weights (Softmax + NLLLoss accepts `weight`).
        net_kwargs: dict[str, Any] = {}
        if str(opts.get("class_weight") or "none") == "balanced":
            counts = np.bincount(np.asarray(y, dtype=np.int64),
                                 minlength=dim_out).astype(np.float64)
            weights = np.where(counts > 0,
                               len(y) / (dim_out * np.maximum(counts, 1)), 0.0)
            net_kwargs["criterion__weight"] = torch.tensor(
                weights, dtype=torch.float32, device=f"cuda:{gpu_id}")
            print(f"[{label}] class weights = "
                  f"{np.round(weights, 4).tolist()}", flush=True)

        net = NeuralNetClassifier(
            TiaraMLP(4 ** int(arch["k"]), int(arch["hidden_1"]),
                     arch.get("hidden_2"), dim_out, float(arch["dropout"])),
            max_epochs=int(arch["epochs"]),
            lr=float(arch["lr"]),
            train_split=None,
            iterator_train__shuffle=shuffle,
            iterator_train__pin_memory=True,
            optimizer=torch.optim.Adam,
            device=f"cuda:{gpu_id}",
            batch_size=batch_size,
            verbose=10,
            **net_kwargs,
        )
        net.fit(X, y)

        # ---- POST-FIT COLLAPSE SANITY CHECK -----------------------------
        # The last line of defence. If the fitted model answers one class for
        # (nearly) every row of a CLASS-BALANCED probe, the .pkl is never
        # written, so a constant predictor cannot reach publish -- which is
        # exactly what escaped in v2.1.2.
        limit = float(opts.get("sanity_max_pred_share", 0.0) or 0.0)
        if limit > 0:
            share, probe_n = _collapse_probe(
                net, feature_cache, stage, k, X, y, dim_out,
                log=lambda m: print(f"[{label}] {m}", flush=True))
            if share is not None:
                print(f"[{label}] sanity: max single-class prediction share = "
                      f"{share:.4f} over {probe_n:,} balanced probe rows",
                      flush=True)
                if share > limit:
                    raise RuntimeError(
                        f"MODEL COLLAPSE: {stage} k={k} predicts one class for "
                        f"{share:.2%} of a class-balanced probe (limit "
                        f"{limit:.2%}). Refusing to write {output_path.name}.")

        net.save_params(f_params=str(tmp_path))
        if not is_complete(tmp_path):
            raise RuntimeError(f"Temporary model was not written correctly: {tmp_path}")
        os.replace(tmp_path, output_path)  # atomic completion boundary for resume
        elapsed = (time.time() - started) / 60
        print(f"[{label}] DONE in {elapsed:.1f} min -> {output_path}", flush=True)
        return {"stage": stage, "k": arch["k"], "gpu": gpu_id,
                "output": str(output_path), "minutes": elapsed, "status": "trained"}
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        try:
            del X
        except UnboundLocalError:
            pass
        if job_scratch:
            shutil.rmtree(job_scratch, ignore_errors=True)
        gc.collect()
        torch.cuda.empty_cache()


def query_gpu_metrics(allowed_gpus: list[int]) -> list[dict[str, int]]:
    """Return allowed GPUs ranked later by free memory and utilization."""
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    allowed = set(allowed_gpus)
    metrics: list[dict[str, int]] = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 3:
            continue
        gpu, free_mib, util = map(int, fields)
        if gpu in allowed:
            metrics.append({"gpu": gpu, "free_mib": free_mib, "util": util})
    return metrics


def dynamic_worker_entry(task: tuple[int, str, dict[str, Any], str, str, str, int],
                         gpu_id: int, cpu_threads: int, result_queue: Any) -> None:
    """Run one model on an explicitly reserved GPU and report the result."""
    global _WORKER_GPU
    _WORKER_GPU = gpu_id
    torch.set_num_threads(cpu_threads)
    torch.cuda.set_device(gpu_id)
    try:
        result_queue.put({"ok": True, "result": train_one(task)})
    except BaseException as exc:
        result_queue.put({
            "ok": False,
            "gpu": gpu_id,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        })
        raise


def parse_gpu_ids(value: str) -> list[int]:
    try:
        ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--gpus must look like 1,3,4,5") from exc
    if not ids or len(ids) != len(set(ids)):
        raise argparse.ArgumentTypeError("--gpus must contain unique GPU IDs")
    return ids


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_tfidf_dir(tfidf_dir: Path,
                       k_first: tuple[int, ...] = (4, 5, 6),
                       k_second: tuple[int, ...] = (4, 5, 6, 7)) -> list[dict[str, Any]]:
    records = []
    for stage, kmers in (("first", k_first), ("second", k_second)):
        for k in kmers:
            folder = idf_path(tfidf_dir, stage, k)
            model, params = folder / "model.npy", folder / "params.txt"
            if not model.is_file() or not params.is_file():
                raise FileNotFoundError(f"Missing TF-IDF files in {folder}")
            idf = np.load(model, mmap_mode="r")
            if idf.shape != (4 ** k,):
                raise ValueError(f"Invalid TF-IDF shape in {model}: {idf.shape}")
            records.append({"stage": stage, "k": k, "folder": str(folder),
                            "model_sha256": sha256_file(model),
                            "params_sha256": sha256_file(params)})
    return records


def input_file_metadata(data_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for name in FILE_NAMES.values():
        path = data_dir / name
        stat = path.stat()
        rows.append({"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return rows


def feature_cache_metadata(feature_cache: Path,
                           jobs: list[tuple[str, dict[str, Any]]],
                           include_validation: bool) -> list[dict[str, Any]]:
    """Stat the exact cache files used by the final refit.

    The cache path no longer requires legacy flat FASTAs, so resume safety must
    fingerprint the train/validation matrices themselves instead.
    """
    rows = []
    tags = ("train", "val") if include_validation else ("train",)
    seen = set()
    for stage, arch in jobs:
        kdir = _cache_kdir(feature_cache, stage, int(arch["k"]))
        paths = [kdir / "meta.json"]
        for tag in tags:
            paths.extend((kdir / f"{tag}_X.f32", kdir / f"{tag}_y.i64"))
        for path in paths:
            if path in seen:
                continue
            seen.add(path)
            if not path.is_file():
                raise FileNotFoundError(f"Missing feature cache file: {path}")
            stat = path.stat()
            rows.append({"path": str(path), "size": stat.st_size,
                         "mtime_ns": stat.st_mtime_ns})
    return rows


def make_run_signature(source: str, jobs: list[tuple[str, dict[str, Any]]],
                       tfidf_models: list[dict[str, Any]], inputs: list[dict[str, Any]],
                       *, include_validation: bool = False) -> str:
    payload = {"source": source, "jobs": [{"stage": s, **a} for s, a in jobs],
               "tfidf": tfidf_models, "inputs": inputs,
               "include_validation": bool(include_validation)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, help="flat directory with five *_fr.fasta files")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("cpu_threads", type=int, help="CPU threads per GPU worker (recommend 1-2)")
    parser.add_argument("--hp-dir", type=Path, default=None,
                        help="directory containing hp_first_k{4,5,6}.json and hp_second_k{4,5,6,7}.json")
    parser.add_argument("--tfidf-dir", type=Path, required=True,
                        help="versioned TF-IDF root")
    parser.add_argument("--k-first", type=int, nargs="+", default=[4, 5, 6],
                        help="first-stage k values; must match HP/cache/TF-IDF")
    parser.add_argument("--k-second", type=int, nargs="+", default=[4, 5, 6, 7],
                        help="second-stage k values; must match HP/cache/TF-IDF")
    parser.add_argument("--epochs", type=int, default=50,
                        help="final epochs for optimized HP models (default: 50)")
    parser.add_argument("--gpus", type=parse_gpu_ids, default=parse_gpu_ids("0,1,2,3,4,5,6,7"),
                        help="allowed GPU whitelist; scheduler chooses among these dynamically")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--feature-cache", type=Path, default=None,
                        help="featurize_cache root; train from the cached "
                             "<stage>/k<k>/train_X.f32 matrices instead of "
                             "re-featurizing the FASTA files in RAM")
    parser.add_argument("--class-weight", choices=["none", "balanced"],
                        default="balanced",
                        help="inverse-frequency weighting of the training loss")
    parser.add_argument("--min-mean-f1", type=float, default=0.0,
                        help="refuse to train when the best stage-1 HP "
                             "candidate scores below this mean F1")
    parser.add_argument("--sanity-max-pred-share", type=float, default=0.98,
                        help="after fitting, a model predicting one class for "
                             "more than this share of a class-balanced probe "
                             "is rejected and its .pkl is NOT written "
                             "(0 disables)")
    parser.add_argument("--include-validation", action="store_true",
                        help="final refit uses train + validation feature "
                             "caches without creating flat FASTA files")
    parser.add_argument("--model-max-rows", type=int, default=4_000_000,
                        help="stratified row cap per model (0 = use every "
                             "cached row); only used with --feature-cache")
    parser.add_argument("--ram-budget-gib", type=float, default=48.0,
                        help="per-worker RAM budget; a capped subset larger "
                             "than this is streamed from a memmap instead")
    parser.add_argument("--scratch", type=Path, default=None,
                        help="scratch dir for streamed subsets "
                             "(default: OUTPUT_DIR/.scratch)")
    parser.add_argument("--progress-secs", type=float, default=60.0,
                        help="seconds between subset-build progress lines and "
                             "scheduler heartbeats (0 disables)")
    parser.add_argument("--max-parallel", type=int, default=7,
                        help="maximum concurrent model processes")
    parser.add_argument("--max-tasks-per-gpu", type=int, default=2,
                        help="maximum NNet processes on one GPU (default: 2)")
    parser.add_argument("--shared-min-free-mib", type=int, default=10000,
                        help="free VRAM required before adding another task")
    parser.add_argument("--shared-max-gpu-util", type=int, default=70,
                        help="utilization ceiling before adding another task")
    parser.add_argument("--share-launch-delay", type=float, default=45.0,
                        help="seconds to wait before sharing a just-used GPU")
    parser.add_argument("--min-free-mib", type=int, default=18000,
                        help="launch only when free VRAM is at least this many MiB (default: 18000 of 24564)")
    parser.add_argument("--max-gpu-util", type=int, default=30,
                        help="launch only when GPU utilization is at most this percent (default: 30)")
    parser.add_argument("--poll-seconds", type=float, default=15.0,
                        help="seconds between GPU availability checks (default: 15)")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                        help="skip non-empty completed model files (default: enabled)")
    parser.add_argument("--subsample", default=None,
                        help="JSON subsample config (see seqpack.normalize_subsample); "
                             "applies the HP train-split selection to the flat data")
    return parser.parse_args()


def write_manifest(path: Path, source: str, jobs: list[tuple[str, dict[str, Any]]],
                   tfidf_dir: Path, tfidf_models: list[dict[str, Any]],
                   inputs: list[dict[str, Any]], signature: str,
                   statuses: list[dict[str, Any]] | None = None) -> None:
    complete = sum(x.get("status") in {"trained", "skipped_existing"} for x in (statuses or []))
    payload = {
        "status": "complete" if complete == len(jobs) else "in_progress",
        "run_signature": signature,
        "parameter_source": source,
        "tfidf_dir": str(tfidf_dir),
        "tfidf_models": tfidf_models,
        "input_files": inputs,
        "models": [{"stage": stage, **arch, "filename": model_name(stage, arch)}
                   for stage, arch in jobs],
        "statuses": statuses or [],
    }
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp, path)

def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    hp_dir = args.hp_dir.expanduser().resolve() if args.hp_dir else None
    tfidf_dir = args.tfidf_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # The legacy path trains on flat train+val FASTAs.  The feature-cache path
    # selects train only or train+validation according to --include-validation.
    subcfg = seqpack.normalize_subsample(
        json.loads(args.subsample) if args.subsample else None)
    spec = seqpack.split_spec(subcfg, "train") if subcfg is not None else None

    # Flat FASTAs are required only on the legacy no-cache path.  With a
    # feature cache the positional data_dir is metadata-only and may be the
    # nested train_ready root.
    if args.feature_cache is None:
        required = [data_dir / name for name in FILE_NAMES.values()]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Missing training FASTA files:\n" + "\n".join(missing))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this PyTorch environment")
    invalid = [gpu for gpu in args.gpus if gpu < 0 or gpu >= torch.cuda.device_count()]
    if invalid:
        raise ValueError(f"Invalid GPU IDs {invalid}; torch sees {torch.cuda.device_count()} GPUs")
    if args.max_parallel < 1 or args.max_tasks_per_gpu < 1:
        raise ValueError("--max-parallel and --max-tasks-per-gpu must be >= 1")
    k_first, k_second = tuple(args.k_first), tuple(args.k_second)
    for name, values in (("--k-first", k_first), ("--k-second", k_second)):
        if not values or len(set(values)) != len(values):
            raise ValueError(f"{name} must contain unique k values")
        if any(not 1 <= int(k) <= 8 for k in values):
            raise ValueError(f"{name} values must be integers in [1, 8]")

    feature_cache = args.feature_cache.expanduser().resolve() if args.feature_cache else None
    if feature_cache is not None and not feature_cache.is_dir():
        raise FileNotFoundError(f"--feature-cache is not a directory: {feature_cache}")
    scratch_root = (args.scratch.expanduser().resolve() if args.scratch
                    else output_dir / ".scratch")
    opts = {
        "feature_cache": str(feature_cache) if feature_cache else None,
        "cap": int(args.model_max_rows),
        "ram_budget": int(args.ram_budget_gib * (1 << 30)),
        "scratch_root": str(scratch_root),
        "progress_secs": float(args.progress_secs),
        "include_validation": bool(args.include_validation),
        "class_weight": str(args.class_weight),
        "sanity_max_pred_share": float(args.sanity_max_pred_share),
    }

    if args.include_validation:
        # Allowed, but never silently: with validation folded in there is no
        # held-out set left, and the post-fit probe degrades to training rows.
        print("[warn] --include-validation: the final fit consumes the "
              "validation split, so the collapse probe is no longer "
              "held out. v2.1.2 failed this way.", flush=True)

    jobs = build_jobs(hp_dir, args.epochs, float(args.min_mean_f1),
                      k_first=k_first, k_second=k_second)
    source = str(hp_dir) if hp_dir else "built-in fixed Tiara parameters"
    tfidf_models = validate_tfidf_dir(tfidf_dir, k_first, k_second)
    inputs = (feature_cache_metadata(
        feature_cache, jobs, bool(args.include_validation))
        if feature_cache is not None else input_file_metadata(data_dir))
    signature = make_run_signature(
        source, jobs, tfidf_models, inputs,
        include_validation=bool(args.include_validation))
    manifest_path = output_dir / "training_manifest.json"
    existing = [output_dir / model_name(stage, arch) for stage, arch in jobs
                if (output_dir / model_name(stage, arch)).is_file()]
    if existing:
        if not manifest_path.is_file():
            raise RuntimeError("Existing models have no manifest; refusing unsafe resume")
        old = json.loads(manifest_path.read_text())
        if old.get("run_signature") != signature:
            raise RuntimeError("Existing models use different HP/data/TF-IDF; move output directory first")
    write_manifest(manifest_path, source, jobs, tfidf_dir, tfidf_models, inputs, signature)

    statuses: list[dict[str, Any]] = []
    pending: list[tuple[int, str, dict[str, Any], str, str, str, int, Any]] = []
    for idx, (stage, arch) in enumerate(jobs):
        target = output_dir / model_name(stage, arch)
        if args.resume and is_complete(target):
            print(f"[resume] SKIP {stage} k={arch['k']}: {target.name}", flush=True)
            statuses.append({"stage": stage, "k": arch["k"], "output": str(target), "status": "skipped_existing"})
        else:
            pending.append((idx, stage, arch, str(data_dir), str(output_dir),
                            str(tfidf_dir), args.batch_size, spec, opts))

    if not pending:
        print("[resume] All seven optimized models are already complete.")
        write_manifest(manifest_path, source, jobs, tfidf_dir, tfidf_models, inputs, signature, statuses)
        return 0

    max_parallel = min(args.max_parallel, len(args.gpus) * args.max_tasks_per_gpu, len(pending))
    print(f"Parameter source: {source}")
    for stage, arch in jobs:
        print(f"  {stage:6s} k={arch['k']} hid1={arch['hidden_1']} hid2={arch.get('hidden_2')} "
              f"lr={arch['lr']} drop={arch['dropout']} epochs={arch['epochs']} "
              f"valid_f1={arch.get('validation_mean_f1', 'n/a')}")
    print(f"Dynamic scheduler: pending={len(pending)}/7 max_parallel={max_parallel} "
          f"allowed={args.gpus} min_free={args.min_free_mib}MiB "
          f"max_util={args.max_gpu_util}% poll={args.poll_seconds}s")

    # Dynamic scheduling: before every launch, re-query all allowed GPUs and
    # rank eligible cards by free VRAM descending, then utilization ascending.
    # A GPU is reserved in `active` until its child process exits, so this
    # scheduler never places two of its own models on the same card.
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    active: dict[int, dict[str, Any]] = {}
    last_wait_log = 0.0
    total_jobs = len(pending)
    sched_t0 = time.time()
    last_heartbeat = time.time()
    failure: str | None = None

    try:
        while pending or active:
            # Reap completed children and release their GPU reservations.
            for pid, record in list(active.items()):
                proc = record["process"]
                if not proc.is_alive():
                    proc.join()
                    print(f"[scheduler] RELEASE GPU {record['gpu']} from pid={pid} "
                          f"exit={proc.exitcode}", flush=True)
                    if proc.exitcode != 0 and failure is None:
                        failure = (f"model process failed: pid={pid} gpu={record['gpu']} "
                                   f"exit={proc.exitcode}")
                    del active[pid]

            # Collect worker reports and update the manifest incrementally.
            while True:
                try:
                    message = result_queue.get_nowait()
                except queue.Empty:
                    break
                if message.get("ok"):
                    statuses.append(message["result"])
                    write_manifest(manifest_path, source, jobs, tfidf_dir, tfidf_models, inputs, signature, statuses)
                elif failure is None:
                    failure = message.get("traceback") or message.get("error", "unknown worker error")

            if failure is not None:
                raise RuntimeError(failure)

            launched = False
            while pending and len(active) < max_parallel:
                try:
                    metrics = query_gpu_metrics(args.gpus)
                except (OSError, subprocess.SubprocessError, ValueError) as exc:
                    now = time.time()
                    if now - last_wait_log >= 60:
                        print(f"[scheduler] nvidia-smi query failed: {exc}; retrying", flush=True)
                        last_wait_log = now
                    break

                now = time.time()
                counts = {gpu: 0 for gpu in args.gpus}
                oldest = {gpu: now for gpu in args.gpus}
                for record in active.values():
                    gpu = record["gpu"]
                    counts[gpu] += 1
                    oldest[gpu] = min(oldest[gpu], record["launched_at"])

                exclusive = [row for row in metrics
                             if counts[row["gpu"]] == 0
                             and row["free_mib"] >= args.min_free_mib
                             and row["util"] <= args.max_gpu_util]
                exclusive.sort(key=lambda row: (row["util"], -row["free_mib"], row["gpu"]))

                shared = []
                if not exclusive and args.max_tasks_per_gpu > 1:
                    shared = [row for row in metrics
                              if 0 < counts[row["gpu"]] < args.max_tasks_per_gpu
                              and now - oldest[row["gpu"]] >= args.share_launch_delay
                              and row["free_mib"] >= args.shared_min_free_mib
                              and row["util"] <= args.shared_max_gpu_util]
                    shared.sort(key=lambda row: (row["util"], counts[row["gpu"]],
                                                 -row["free_mib"], row["gpu"]))
                eligible = exclusive or shared
                if not eligible:
                    if now - last_wait_log >= 60:
                        ranked = sorted(metrics, key=lambda row: (row["util"], -row["free_mib"]))
                        summary = ", ".join(
                            f"gpu{x['gpu']} tasks={counts[x['gpu']]} free={x['free_mib']}MiB util={x['util']}%"
                            for x in ranked)
                        print(f"[scheduler] WAIT: no eligible GPU ({summary})", flush=True)
                        last_wait_log = now
                    break
                chosen = eligible[0]
                launch_mode = "exclusive" if exclusive else "shared"
                task = pending.pop(0)
                _, stage, arch, *_ = task
                proc = ctx.Process(
                    target=dynamic_worker_entry,
                    args=(task, chosen["gpu"], args.cpu_threads, result_queue),
                )
                proc.start()
                active[proc.pid] = {
                    "process": proc,
                    "gpu": chosen["gpu"],
                    "stage": stage,
                    "k": arch["k"],
                    "launched_at": time.time(),
                }
                print(f"[scheduler] LAUNCH {launch_mode} {stage} k={arch['k']} -> GPU {chosen['gpu']} "
                      f"slot={counts[chosen['gpu']]+1}/{args.max_tasks_per_gpu} "
                      f"(free={chosen['free_mib']}MiB util={chosen['util']}%) pid={proc.pid}",
                      flush=True)
                launched = True

            # Once every slot is busy the scheduler stops printing entirely and
            # only the workers' epoch lines appear -- and a 4M-row epoch can take
            # many minutes. Mirror the HP search heartbeat so the console always
            # shows what is running and for how long.
            now_hb = time.time()
            if args.progress_secs > 0 and now_hb - last_heartbeat >= args.progress_secs:
                done_n = total_jobs - len(pending) - len(active)
                act = ", ".join(
                    f"{r['stage']} k={r['k']} gpu={r['gpu']} "
                    f"{(now_hb - r['launched_at']) / 60:.0f}m"
                    for r in sorted(active.values(),
                                    key=lambda d: d["launched_at"]))
                print(f"[scheduler] HEARTBEAT done={done_n}/{total_jobs} "
                      f"running={len(active)} pending={len(pending)} "
                      f"elapsed={(now_hb - sched_t0) / 60:.1f}m"
                      + (f" active=[{act}]" if act else ""), flush=True)
                last_heartbeat = now_hb

            if pending or active:
                time.sleep(args.poll_seconds if not launched else min(2.0, args.poll_seconds))
    finally:
        # On Ctrl-C or a worker failure, stop remaining children. Completed
        # atomic .pkl files stay resumable; partial temporary files are ignored.
        for record in active.values():
            proc = record["process"]
            if proc.is_alive():
                proc.terminate()
        for record in active.values():
            record["process"].join(timeout=10)

    incomplete = [output_dir / model_name(stage, arch) for stage, arch in jobs
                  if not is_complete(output_dir / model_name(stage, arch))]
    if incomplete:
        raise RuntimeError("Training ended but models are missing:\n" + "\n".join(map(str, incomplete)))
    write_manifest(manifest_path, source, jobs, tfidf_dir, tfidf_models, inputs, signature, statuses)
    print(f"All seven models complete -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

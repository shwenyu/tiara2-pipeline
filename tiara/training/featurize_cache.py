"""Shared, parallel, cached k-mer feature builder for the GPU HP search.

This module factors out the single most expensive part of the HP search --
"read FASTA -> 2-bit k-mer count -> TF-IDF weight -> L2 normalize" -- and makes
it fast in three ways AT ONCE:

  1. On-disk reuse (skip-if-exists).  Features are written to a PERSISTENT
     cache keyed by (stage, k, split) + an input fingerprint (per-file size).
     A --resume / restart never recomputes a valid (stage, k, split).

  2. Parallel read + featurize.  Each input FASTA is split into byte-range
     shards (many per worker) and featurized across a process Pool, so the
     multi-TB read/parse/count is spread over many CPUs instead of one.

  3. One pass, many k.  A single call can build several k in the SAME pass over
     the data, so the huge inputs are read once per split rather than once per
     (split, k).

The on-disk byte layout is IDENTICAL to the old `build_split_to_memmap` output
(row-major float32 X of shape (n, dim); int64 y of shape (n,), rows in group /
file order), so the GPU workers open the cached files unchanged.

Design constraints (so this can be unit-tested without the ML stack):
  * numba is OPTIONAL -- if missing, a pure-Python counter with identical
    semantics is used.
  * No torch / Biopython import at module load; the FASTA reader is a small
    self-contained byte-range parser matching SimpleFastaParser semantics for
    ACGT data. `TfidfWeighter` is imported lazily only when loading IDF.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

# The 2-bit pack + subsample helpers (numpy-only; lazy-imports this module).
try:
    from . import seqpack
except ImportError:  # top-level import in the sandbox/test harness
    import seqpack

# --------------------------------------------------------------------------- #
# numba is optional. Fall back to a pure-Python njit no-op decorator so the
# exact same counter source runs (slowly) when numba is unavailable.
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - depends on environment
    from numba import njit as _njit
    _HAVE_NUMBA = True
except Exception:  # pragma: no cover
    _HAVE_NUMBA = False

    def _njit(*args, **kwargs):
        if args and callable(args[0]) and not kwargs:
            return args[0]

        def _deco(fn):
            return fn

        return _deco


# --------------------------------------------------------------------------- #
# Stage definitions -- copied VERBATIM from hyperparameter_search_gpu so the
# cached features/labels are bit-identical to the streaming path they replace.
# --------------------------------------------------------------------------- #
STAGE_SPEC = {
    "first": {
        "files": ["organelle", "bacteria", "archaea", "eukarya"],
        "labels": [0, 1, 3, 4],
        "idf": "first-stage",
    },
    "second": {
        "files": ["plastids", "mitochondria"],
        "labels": [0, 2],
        "idf": "second-stage",
    },
}

_SPLIT_TAG = {"train": "train", "validation": "val"}


@_njit(cache=True)
def count_kmers_into(seq: np.ndarray, k: int, out: np.ndarray) -> None:
    """Rolling 2-bit ACGT k-mer counter (identical to the original njit fn).

    Index order == product('ACGT', repeat=k), matching the stored IDF ordering.
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


def featurize_block(seqs, k: int, idf: np.ndarray, dim: int) -> np.ndarray:
    """L2-normalized TF-IDF features for a block (identical to the original).

    Every row is independent, so splitting a split into shards/chunks does not
    change any row's value -- the concatenation is bit-identical to featurizing
    the whole split at once.
    """
    X = np.zeros((len(seqs), dim), dtype=np.float32)
    for i, seq in enumerate(seqs):
        if isinstance(seq, bytes):
            raw = np.frombuffer(seq, dtype=np.uint8)
        else:
            raw = np.frombuffer(seq.encode("ascii", errors="ignore"), dtype=np.uint8)
        count_kmers_into(raw, k, X[i])
    X *= idf
    norms = np.linalg.norm(X, axis=1)
    nz = norms > 0
    X[nz] /= norms[nz, None]
    return X


# --------------------------------------------------------------------------- #
# Input resolution (group -> file paths + label), matching group_paths.
# --------------------------------------------------------------------------- #
def group_paths(input_dir, split: str, name: str):
    split_dir = Path(input_dir) / split
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


def stage_split_inputs(input_dir, stage: str, split: str):
    """Ordered list of (path, label) for a (stage, split), in the SAME group /
    file order the streaming builder used, so rows line up with labels."""
    spec = STAGE_SPEC[stage]
    inputs = []
    for name, label in zip(spec["files"], spec["labels"]):
        for p in group_paths(input_dir, split, name):
            if Path(p).is_file():
                inputs.append((str(p), int(label)))
    return inputs


def _inputs_sig(inputs):
    """Cheap fingerprint: ordered [basename, size] per input file."""
    return [[os.path.basename(p), os.path.getsize(p)] for p, _ in inputs]


# --------------------------------------------------------------------------- #
# Byte-range FASTA shard reader.
#
# A record belongs to a shard [start, end) iff the byte offset of its '>' header
# line, o, satisfies start <= o < end. This partitions every record into exactly
# one shard with no gaps and no overlaps for any set of contiguous [start, end)
# ranges covering [0, filesize].
# --------------------------------------------------------------------------- #
def _iter_fasta_shard_raw(path, start: int, end: int):
    """Yield (sequence, raw_bytes) for records whose header offset is in
    [start, end). raw_bytes is the on-disk size of that record (its header line
    plus its sequence lines), so summing raw_bytes as records are consumed
    tracks real bytes scanned -- this drives a smooth, byte-based progress bar
    even while a single big shard is still mid-flight."""
    with open(path, "rb") as fh:
        if start != 0:
            # Land exactly on a line boundary: if the byte before `start` is not
            # a newline we are mid-line, so drop that partial line (it belongs to
            # a record owned by the previous shard).
            fh.seek(start - 1)
            if fh.read(1) != b"\n":
                fh.readline()
        header_seen = False
        parts = []
        rec_bytes = 0
        while True:
            line_start = fh.tell()
            line = fh.readline()
            if not line:
                break
            if line[:1] == b">":
                if line_start >= end:
                    # This header starts the next shard's territory.
                    break
                if header_seen:
                    yield "".join(parts).replace(" ", "").replace("\r", ""), rec_bytes
                header_seen = True
                parts = []
                rec_bytes = len(line)
            elif header_seen:
                parts.append(line.decode("ascii", errors="ignore").rstrip())
                rec_bytes += len(line)
        if header_seen:
            yield "".join(parts).replace(" ", "").replace("\r", ""), rec_bytes


def iter_fasta_shard(path, start: int, end: int):
    """Yield sequence strings whose header offset is in [start, end)."""
    for seq, _nb in _iter_fasta_shard_raw(path, start, end):
        yield seq


def plan_shards(entries, target_shards: int):
    """Split each file into byte sub-ranges roughly proportional to its size.

    Returns a list of (sid, path, label, start, end). `sid` increases with file
    order then byte order, so sorting results by sid reproduces the exact
    group/file/record order of the streaming builder.
    """
    sizes = [(p, l, os.path.getsize(p)) for p, l in entries]
    total = sum(s for _, _, s in sizes) or 1
    shards = []
    sid = 0
    for p, l, sz in sizes:
        if sz == 0:
            shards.append((sid, p, l, 0, 0))
            sid += 1
            continue
        n = max(1, int(round(target_shards * sz / total)))
        step = max(1, sz // n)
        bounds = sorted(set(list(range(0, sz, step))[:n] + [0, sz]))
        for a, b in zip(bounds, bounds[1:]):
            shards.append((sid, p, l, a, b))
            sid += 1
    return shards


# --------------------------------------------------------------------------- #
# Worker: featurize one shard for ALL requested k in a single read pass.
# --------------------------------------------------------------------------- #
_W = {}


def _init_worker(ks, idf_by_k, dims, chunk, tmp, pq=None, spec=None):
    _W["ks"] = ks
    _W["idf"] = idf_by_k
    _W["dims"] = dims
    _W["chunk"] = chunk
    _W["tmp"] = tmp
    _W["pq"] = pq
    _W["spec"] = spec


def _featurize_shard(task):
    sid, path, label, start, end = task
    ks = _W["ks"]
    idf = _W["idf"]
    dims = _W["dims"]
    chunk = _W["chunk"]
    tmp = _W["tmp"]

    xfiles = {k: open(os.path.join(tmp, f"part{sid}_k{k}_X.f32"), "wb") for k in ks}
    yfile = open(os.path.join(tmp, f"part{sid}_y.i64"), "wb")
    pq = _W.get("pq")
    count = 0
    buf = []
    bytes_acc = 0  # raw bytes read since the last flush (for progress)

    def _flush():
        nonlocal count, bytes_acc
        if not buf:
            return
        nseq = len(buf)
        for k in ks:
            block = np.ascontiguousarray(
                featurize_block(buf, k, idf[k], dims[k]), dtype=np.float32)
            xfiles[k].write(block.tobytes())
        yfile.write(np.full(nseq, label, dtype=np.int64).tobytes())
        count += nseq
        if pq is not None:
            try:
                # (bytes_delta, seqs_delta): bytes drive %/rate/ETA; seqs proves
                # liveness within a single big shard.
                pq.put((bytes_acc, nseq))
            except Exception:
                pass
        bytes_acc = 0
        buf.clear()

    spec = _W.get("spec")
    # Per-class rate, resolved once per shard from its path. Must match the
    # pack path (seqpack._pack_shard) and the final-model path
    # (train_models_gpu._subsample_seqs) exactly, or the three would disagree
    # about which records exist.
    rate = seqpack.rate_for_path(spec, path) if spec is not None else 1.0
    try:
        for seq, nb in _iter_fasta_shard_raw(path, start, end):
            bytes_acc += nb
            if spec is not None and not seqpack.keep_sequence(
                    seq, rate=rate, min_purity=spec["min_purity"],
                    min_len=spec["min_len"], seed=spec["seed"]):
                continue
            buf.append(seq)
            if len(buf) >= chunk:
                _flush()
        _flush()
    finally:
        for f in xfiles.values():
            f.close()
        yfile.close()
    return sid, count


# --------------------------------------------------------------------------- #
# Cache metadata + validity.
# --------------------------------------------------------------------------- #
def _kdir(feat_root, stage, k):
    return Path(feat_root) / stage / f"k{k}"


def _meta_path(kdir):
    return Path(kdir) / "meta.json"


def _read_meta(kdir):
    p = _meta_path(kdir)
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _update_meta(kdir, tag, n, dim, inputs_sig):
    meta = _read_meta(kdir)
    meta[tag] = {"n": int(n), "dim": int(dim), "inputs_sig": inputs_sig}
    _meta_path(kdir).write_text(json.dumps(meta))


def _split_valid(kdir, tag, dim, inputs_sig):
    meta = _read_meta(kdir).get(tag)
    if not meta:
        return None
    if int(meta.get("dim", -1)) != int(dim):
        return None
    if meta.get("inputs_sig") != inputs_sig:
        return None
    n = int(meta.get("n", -1))
    if n < 0:
        return None
    xp = Path(kdir) / f"{tag}_X.f32"
    yp = Path(kdir) / f"{tag}_y.i64"
    if not xp.is_file() or not yp.is_file():
        return None
    if xp.stat().st_size != n * dim * 4:
        return None
    if yp.stat().st_size != n * 8:
        return None
    return n


def _fmt_bytes(n):
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024:
            return f"{int(n)}B" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TiB"


def _fmt_dur(s):
    s = int(max(0, s))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{sec:02d}" if h else f"{m:d}:{sec:02d}"


def _bar(frac, width=12):
    frac = 0.0 if frac < 0 else (1.0 if frac > 1 else frac)
    fill = int(round(frac * width))
    return "#" * fill + "-" * (width - fill)


# --------------------------------------------------------------------------- #
# Build one split for a set of k in a single parallel pass.
#
# Progress: the total bytes to scan is known up front. Workers STREAM partial
# (bytes, seqs) deltas as they read, so % / rate / ETA advance smoothly even
# while a single big shard is still mid-flight; completed-shard bytes act as a
# monotonic FLOOR so the two accounting sources never double-count and the bar
# hits exactly 100% when every shard is done. Lines are throttled to
# progress_secs.
# --------------------------------------------------------------------------- #
def _build_split(feat_root, input_dir, stage, split, ks, idf_by_k, inputs,
                 inputs_sig, workers, chunk, log, progress_secs=20.0,
                 spec=None):
    tag = _SPLIT_TAG[split]
    entries = [(p, l) for p, l in inputs]
    dims = {k: 4 ** k for k in ks}

    # Aim for reasonably small shards (~1 GiB) so the progress bar AND worker
    # load-balancing stay smooth on multi-TB inputs, with a floor tied to the
    # worker count and a hard cap on the shard/temp-file count.
    total_in = sum(os.path.getsize(p) for p, _l in entries) or 1
    target_shards = max(workers * 4, min(4096, max(1, total_in // (1 << 30))))
    shards = plan_shards(entries, target_shards)
    shard_bytes = {sid: max(0, e - s) for sid, _p, _l, s, e in shards}
    total_bytes = sum(shard_bytes.values())
    total_shards = len(shards)

    Path(feat_root).mkdir(parents=True, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=f"featbuild_{stage}_{tag}_", dir=str(feat_root))

    t0 = time.time()
    state = {"bytes": 0, "shards": 0, "seqs": 0}
    lock = threading.Lock()

    def _emit():
        # Hold the lock across the log() call too, so lines from the main thread
        # (shard completions) and the drainer thread (heartbeats) never
        # interleave into a garbled console line.
        with lock:
            db, ds, sq = state["bytes"], state["shards"], state["seqs"]
            el = max(1e-6, time.time() - t0)
            if total_bytes:
                db = min(db, total_bytes)
                frac = db / total_bytes
            else:
                frac = ds / total_shards if total_shards else 1.0
            rate = db / el
            eta = ((total_bytes - db) / rate) if (rate > 0 and total_bytes) else 0
            # Field order matters: the most useful fields (%, ETA, bytes, rate)
            # come first so that if the console clamps this line to the terminal
            # width, the tail (seqs / shards / elapsed) is what gets dropped --
            # never the ETA. Keep the line compact to avoid wrapping.
            log(
                f"[{stage}/{split}] read+featurize [{_bar(frac)}] {frac * 100:5.1f}% | "
                f"ETA {_fmt_dur(eta)} | {_fmt_bytes(db)}/{_fmt_bytes(total_bytes)} | "
                f"{_fmt_bytes(rate)}/s | {sq:,} seqs | "
                f"shards {ds}/{total_shards} | elapsed {_fmt_dur(el)}"
            )

    try:
        if workers <= 1 or len(shards) <= 1:
            _init_worker(ks, idf_by_k, dims, chunk, tmp, None, spec)
            results = []
            for task in shards:
                sid, cnt = _featurize_shard(task)
                results.append((sid, cnt))
                with lock:
                    state["bytes"] += shard_bytes.get(sid, 0)
                    state["shards"] += 1
                    state["seqs"] += cnt
                _emit()
        else:
            import multiprocessing as mp
            ctx = mp.get_context("fork")
            pq = ctx.Queue()
            stop = threading.Event()

            def _drain():
                # Emit at least every progress_secs even when NO new sequences
                # arrived (a single huge shard can be mid-flight for a while),
                # so the elapsed clock / rate keep moving and the run never
                # looks hung.
                last = 0.0
                while True:
                    try:
                        msg = pq.get(timeout=0.5)
                    except Exception:
                        msg = None
                        if stop.is_set():
                            break
                    if msg is not None:
                        nbytes, nseq = msg
                        with lock:
                            state["bytes"] += nbytes
                            state["seqs"] += nseq
                    now = time.time()
                    if now - last >= progress_secs:
                        last = now
                        _emit()

            drainer = threading.Thread(target=_drain, daemon=True)
            drainer.start()

            results = []
            completed_bytes = 0
            with ctx.Pool(processes=workers, initializer=_init_worker,
                          initargs=(ks, idf_by_k, dims, chunk, tmp, pq, spec)) as pool:
                for sid, cnt in pool.imap_unordered(_featurize_shard, shards):
                    results.append((sid, cnt))
                    with lock:
                        # Workers stream partial byte deltas (smooth bar); use
                        # completed-shard bytes only as a monotonic FLOOR so the
                        # two sources never double-count and 100% is reached
                        # exactly when every shard is done.
                        completed_bytes += shard_bytes.get(sid, 0)
                        if completed_bytes > state["bytes"]:
                            state["bytes"] = completed_bytes
                        state["shards"] += 1
                    _emit()
            stop.set()
            drainer.join(timeout=2.0)
            _emit()

        counts = dict(results)
        order = sorted(counts)

        shapes = {}
        for k in ks:
            dim = dims[k]
            kdir = _kdir(feat_root, stage, k)
            kdir.mkdir(parents=True, exist_ok=True)
            # X
            x_tmp = kdir / f"{tag}_X.f32.tmp"
            with open(x_tmp, "wb") as out:
                for sid in order:
                    part = os.path.join(tmp, f"part{sid}_k{k}_X.f32")
                    with open(part, "rb") as pf:
                        shutil.copyfileobj(pf, out, length=1 << 20)
            os.replace(x_tmp, kdir / f"{tag}_X.f32")
            # y (identical across k, but written per-kdir for self-containment)
            y_tmp = kdir / f"{tag}_y.i64.tmp"
            with open(y_tmp, "wb") as out:
                for sid in order:
                    part = os.path.join(tmp, f"part{sid}_y.i64")
                    with open(part, "rb") as pf:
                        shutil.copyfileobj(pf, out, length=1 << 20)
            os.replace(y_tmp, kdir / f"{tag}_y.i64")

            n = sum(counts[sid] for sid in order)
            xsz = (kdir / f"{tag}_X.f32").stat().st_size
            ysz = (kdir / f"{tag}_y.i64").stat().st_size
            if xsz != n * dim * 4:
                raise RuntimeError(
                    f"feature size mismatch {stage}/{tag} k{k}: {xsz} != {n}*{dim}*4")
            if ysz != n * 8:
                raise RuntimeError(
                    f"label size mismatch {stage}/{tag} k{k}: {ysz} != {n}*8")
            _update_meta(kdir, tag, n, dim, inputs_sig)
            shapes[k] = (n, dim)
        return shapes
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Featurize FROM the 2-bit pack (D2). Output byte layout is IDENTICAL to
# _build_split (row-major float32 X, int64 y, group/file order), but the
# sequence source is the compact, already-subsampled, shared-across-k pack -- so
# the multi-TB FASTA is read ONCE at pack time instead of once per stage build,
# and the durable resumable artifact is the small pack rather than tens of TB of
# dense features.
# --------------------------------------------------------------------------- #
_PW = {}


def _init_pack_worker(ks, idf_by_k, dims, chunk, tmp, pdir, pq=None):
    _PW["ks"] = ks
    _PW["idf"] = idf_by_k
    _PW["dims"] = dims
    _PW["chunk"] = chunk
    _PW["tmp"] = tmp
    _PW["pdir"] = pdir
    _PW["pq"] = pq
    _PW["pack"] = None


def _featurize_pack_shard(task):
    sid, rstart, rend = task
    ks = _PW["ks"]
    idf = _PW["idf"]
    dims = _PW["dims"]
    chunk = _PW["chunk"]
    tmp = _PW["tmp"]
    pq = _PW.get("pq")
    if _PW.get("pack") is None:
        _PW["pack"] = seqpack.open_pack(_PW["pdir"])
    index, codes, resets, _n = _PW["pack"]

    xfiles = {k: open(os.path.join(tmp, f"part{sid}_k{k}_X.f32"), "wb") for k in ks}
    yfile = open(os.path.join(tmp, f"part{sid}_y.i64"), "wb")
    count = 0
    buf = []
    labs = []

    def _flush():
        nonlocal count
        if not buf:
            return
        nseq = len(buf)
        for k in ks:
            block = np.ascontiguousarray(
                featurize_block(buf, k, idf[k], dims[k]), dtype=np.float32)
            xfiles[k].write(block.tobytes())
        yfile.write(np.asarray(labs, dtype=np.int64).tobytes())
        count += nseq
        if pq is not None:
            try:
                pq.put((nseq,))
            except Exception:
                pass
        buf.clear()
        labs.clear()

    try:
        for i in range(rstart, rend):
            ascii_arr, lab = seqpack.record_ascii(index, codes, resets, i)
            buf.append(ascii_arr.tobytes())
            labs.append(lab)
            if len(buf) >= chunk:
                _flush()
        _flush()
    finally:
        for f in xfiles.values():
            f.close()
        yfile.close()
    return sid, count


def _build_split_from_pack(feat_root, pdir, stage, split, ks, idf_by_k,
                           inputs_sig, workers, chunk, log, progress_secs=20.0,
                           n_records=None):
    tag = _SPLIT_TAG[split]
    dims = {k: 4 ** k for k in ks}
    total = n_records
    if total is None:
        total = int(np.load(Path(pdir) / "index.npy", mmap_mode="r").size)

    # Record-range shards (sequential ranges -> concat reproduces pack order,
    # which is group/file order, so rows still line up with labels).
    if total:
        nshards = max(1, min(max(workers * 4, 1),
                             max(1, (total + chunk - 1) // max(1, chunk))))
        step = max(1, (total + nshards - 1) // nshards)
    else:
        step = 1
    shards = []
    sid = 0
    a = 0
    while a < total:
        b = min(total, a + step)
        shards.append((sid, a, b))
        a = b
        sid += 1
    if not shards:
        shards = [(0, 0, 0)]
    total_shards = len(shards)

    Path(feat_root).mkdir(parents=True, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=f"featpack_{stage}_{tag}_", dir=str(feat_root))
    t0 = time.time()
    state = {"seqs": 0, "shards": 0}
    lock = threading.Lock()

    def _emit():
        with lock:
            sq, ds = state["seqs"], state["shards"]
            el = max(1e-6, time.time() - t0)
            frac = sq / total if total else 1.0
            rate = sq / el
            eta = ((total - sq) / rate) if (rate > 0 and total) else 0
            log(
                f"[{stage}/{split}] read+featurize [{_bar(frac)}] {frac * 100:5.1f}% | "
                f"{sq:,}/{total:,} seqs | shards {ds}/{total_shards} | "
                f"{rate:,.0f} seq/s | elapsed {_fmt_dur(el)} | ETA {_fmt_dur(eta)}"
            )

    try:
        if workers <= 1 or len(shards) <= 1:
            _init_pack_worker(ks, idf_by_k, dims, chunk, tmp, str(pdir), None)
            results = []
            for task in shards:
                r_sid, cnt = _featurize_pack_shard(task)
                results.append((r_sid, cnt))
                with lock:
                    state["seqs"] += cnt
                    state["shards"] += 1
                _emit()
        else:
            import multiprocessing as mp
            ctx = mp.get_context("fork")
            pq = ctx.Queue()
            stop = threading.Event()

            def _drain():
                last = 0.0
                while True:
                    try:
                        msg = pq.get(timeout=0.5)
                    except Exception:
                        msg = None
                        if stop.is_set():
                            break
                    if msg is not None:
                        with lock:
                            state["seqs"] += msg[0]
                    now = time.time()
                    if now - last >= progress_secs:
                        last = now
                        _emit()

            drainer = threading.Thread(target=_drain, daemon=True)
            drainer.start()
            results = []
            completed = 0
            with ctx.Pool(processes=workers, initializer=_init_pack_worker,
                          initargs=(ks, idf_by_k, dims, chunk, tmp, str(pdir), pq)) as pool:
                for r_sid, cnt in pool.imap_unordered(_featurize_pack_shard, shards):
                    results.append((r_sid, cnt))
                    with lock:
                        completed += cnt
                        if completed > state["seqs"]:
                            state["seqs"] = completed
                        state["shards"] += 1
                    _emit()
            stop.set()
            drainer.join(timeout=2.0)
            _emit()

        counts = dict(results)
        order = sorted(counts)
        shapes = {}
        for k in ks:
            dim = dims[k]
            kdir = _kdir(feat_root, stage, k)
            kdir.mkdir(parents=True, exist_ok=True)
            x_tmp = kdir / f"{tag}_X.f32.tmp"
            with open(x_tmp, "wb") as out:
                for s in order:
                    part = os.path.join(tmp, f"part{s}_k{k}_X.f32")
                    with open(part, "rb") as pf:
                        shutil.copyfileobj(pf, out, length=1 << 20)
            os.replace(x_tmp, kdir / f"{tag}_X.f32")
            y_tmp = kdir / f"{tag}_y.i64.tmp"
            with open(y_tmp, "wb") as out:
                for s in order:
                    part = os.path.join(tmp, f"part{s}_y.i64")
                    with open(part, "rb") as pf:
                        shutil.copyfileobj(pf, out, length=1 << 20)
            os.replace(y_tmp, kdir / f"{tag}_y.i64")
            n = sum(counts[s] for s in order)
            xsz = (kdir / f"{tag}_X.f32").stat().st_size
            ysz = (kdir / f"{tag}_y.i64").stat().st_size
            if xsz != n * dim * 4:
                raise RuntimeError(
                    f"feature size mismatch {stage}/{tag} k{k}: {xsz} != {n}*{dim}*4")
            if ysz != n * 8:
                raise RuntimeError(
                    f"label size mismatch {stage}/{tag} k{k}: {ysz} != {n}*8")
            _update_meta(kdir, tag, n, dim, inputs_sig)
            shapes[k] = (n, dim)
        return shapes
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Public entry point.
# --------------------------------------------------------------------------- #
def ensure_features(feat_root, input_dir, stage, ks, *, tfidf_dir=None,
                    idf_map=None, workers=8, chunk=200_000,
                    splits=("train", "validation"), log=None,
                    progress_secs=20.0, seq_pack=None, subsample=None):
    """Ensure cached features exist for every (split, k); build only what's
    missing/stale, all needed k for a split in ONE pass. Returns
    {"train": {k: (n, dim)}, "val": {k: (n, dim)}} for the requested splits.

    seq_pack: optional root for the shared 2-bit sequence pack (D2). When set,
    each split is packed ONCE (compact, resumable, shared across k) and features
    are derived from the pack instead of re-reading the multi-TB FASTAs.
    subsample: optional raw subsample config (see seqpack.normalize_subsample)
    that deterministically downsamples + quality-filters records, without mmseqs.
    """
    log = log or (lambda _m: None)
    ks = [int(k) for k in ks]
    feat_root = Path(feat_root)
    subcfg = seqpack.normalize_subsample(subsample)
    if seq_pack is not None:
        seq_pack = Path(seq_pack)

    def _idf_for(k):
        if idf_map is not None:
            return np.asarray(idf_map[k], dtype=np.float32)
        return load_idf(stage, k, Path(tfidf_dir))

    result = {}
    for split in splits:
        tag = _SPLIT_TAG[split]
        result[tag] = {}
        inputs = stage_split_inputs(input_dir, stage, split)
        if not inputs:
            raise FileNotFoundError(
                f"No input FASTAs found for {stage}/{split} under {input_dir}")
        base_sig = _inputs_sig(inputs)
        spec = seqpack.split_spec(subcfg, split)
        # Subsampling changes which rows land in the cache, so it MUST be part of
        # the cache signature (else a rate change would silently reuse a stale
        # cache). No subsample -> keep the plain list sig for back-compat.
        sig = base_sig if subcfg is None else {"files": base_sig, "sub": spec}

        needed = []
        for k in ks:
            dim = 4 ** k
            n = _split_valid(_kdir(feat_root, stage, k), tag, dim, sig)
            if n is None:
                needed.append(k)
            else:
                result[tag][k] = (n, dim)

        if not needed:
            log(f"[{stage}/{split}] all k cached {ks}")
            continue

        log(f"[{stage}/{split}] building k={needed} "
            f"(workers={workers}, chunk={chunk})")
        idf_by_k = {k: _idf_for(k) for k in needed}
        if seq_pack is not None:
            pdir = seqpack.pack_dir(seq_pack, stage, split)
            n_rec = seqpack.build_pack(seq_pack, stage, split, inputs,
                                       base_sig, spec, workers=workers, log=log)
            shapes = _build_split_from_pack(
                feat_root, pdir, stage, split, needed, idf_by_k, sig,
                workers, chunk, log, progress_secs=progress_secs,
                n_records=n_rec)
        else:
            shapes = _build_split(feat_root, input_dir, stage, split, needed,
                                  idf_by_k, inputs, sig, workers, chunk, log,
                                  progress_secs=progress_secs,
                                  spec=(spec if subcfg is not None else None))
        for k in needed:
            result[tag][k] = shapes[k]
        log(f"[{stage}/{split}] done k={needed} -> "
            + ", ".join(f"k{k}:{shapes[k]}" for k in needed))

    return result


def load_idf(stage: str, k: int, tfidf_dir) -> np.ndarray:
    """Load the stored TF-IDF idf vector (lazy import of the ML dependency)."""
    from tiara.src.transformations import TfidfWeighter

    idf_dir = Path(tfidf_dir) / f"k{k}-{STAGE_SPEC[stage]['idf']}"
    if not idf_dir.exists():
        raise FileNotFoundError(f"Missing TF-IDF model: {idf_dir}")
    idf = np.asarray(TfidfWeighter.load_params(str(idf_dir)).idfs, dtype=np.float32)
    expected = 4 ** k
    if idf.shape[0] != expected:
        raise ValueError(
            f"IDF length {idf.shape[0]} != 4**{k} ({expected}) for {idf_dir}")
    return idf


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Build the shared, parallel, cached k-mer feature store "
                    "for the GPU hyperparameter search (all k in one pass).")
    ap.add_argument("input_dir", help="train_ready dir containing train/ and validation/")
    ap.add_argument("--stage", required=True, choices=["first", "second"])
    ap.add_argument("--ks", required=True, help="comma-separated k values, e.g. 4,5,6")
    ap.add_argument("--tfidf-dir", required=True, help="versioned TF-IDF root")
    ap.add_argument("--feature-cache", required=True, help="persistent cache root")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--feat-chunk", type=int, default=200_000)
    ap.add_argument("--progress-secs", type=float, default=2.0,
                    help="seconds between progress-bar refreshes emitted to "
                         "stdout; the console renders these in place")
    ap.add_argument("--splits", default="train,validation")
    ap.add_argument("--seq-pack", default=None,
                    help="root for the shared 2-bit sequence pack (D2); when "
                         "set, features are derived from the compact pack so the "
                         "multi-TB FASTA is read only once")
    ap.add_argument("--subsample", default=None,
                    help="JSON subsample config (see seqpack.normalize_subsample)")
    args = ap.parse_args(argv)

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    subsample = json.loads(args.subsample) if args.subsample else None
    shapes = ensure_features(
        Path(args.feature_cache).expanduser().resolve(),
        args.input_dir, args.stage, ks,
        tfidf_dir=args.tfidf_dir, idf_map=None,
        workers=args.workers, chunk=args.feat_chunk,
        splits=splits, progress_secs=args.progress_secs,
        seq_pack=(Path(args.seq_pack).expanduser().resolve()
                  if args.seq_pack else None),
        subsample=subsample,
        log=lambda m: print(m, flush=True))
    print(f"feature cache ready ({'numba' if _HAVE_NUMBA else 'python'}): {shapes}",
          flush=True)


if __name__ == "__main__":
    main()

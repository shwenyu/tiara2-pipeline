"""2-bit packed sequence store + deterministic quality-aware subsampling.

WHY THIS EXISTS
---------------
The old feature cache materialized the DENSE TF-IDF matrices (4**k float32 per
sequence). For ~1e9 fragments that is tens of TB of writes on a single spinning
disk -- the true bottleneck (I/O bound, tasks piling into D state), and it does
not even fit on /data. This module attacks the ROOT of the I/O volume in two
mathematically-lossless ways:

  D2. Cache the *input sequences* in ~2 bits/base instead of the dense features.
      A 2-bit pack is ~4x smaller than the FASTA and ~26-60x smaller than the
      dense feature cache, it is SHARED across every k (one store serves
      k=4..7), and it is the durable artifact for staged resume -- features are
      re-derived from it on demand (cheap CPU work; the box sits ~80% idle).
      Non-ACGT bases (which the k-mer counter treats as window RESETS) are
      preserved EXACTLY via per-record reset points, so features derived from
      the pack are bit-identical to features computed straight from the FASTA.

  B.  Stratified, quality-aware SUBSAMPLING applied AT PACK TIME so the store
      only holds the chosen fragments. Selection is deterministic (content
      hashed), so a uniform rate within every class PRESERVES class proportions
      and a --resume reselects exactly the same set. Train can additionally
      demand high ACGT purity (drop junk / high-N fragments) while val/test stay
      representative. No clustering, no mmseqs, no GPU -- pure arithmetic.

The module is numpy-only (no numba/torch/Bio at import) so it is unit-testable
in the CI sandbox; the heavy k-mer counting still reuses
``featurize_cache.count_kmers_into`` (numba when available), imported lazily to
avoid an import cycle.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import time
import zlib
from pathlib import Path

import numpy as np

# base byte -> 2-bit code (matches count_kmers_into: only UPPER-CASE ACGT count;
# everything else is a window reset).
_CODE_LUT = np.full(256, 255, dtype=np.uint8)
_CODE_LUT[ord("A")] = 0
_CODE_LUT[ord("C")] = 1
_CODE_LUT[ord("G")] = 2
_CODE_LUT[ord("T")] = 3
# 2-bit code -> ACGT ASCII byte (for reconstruction).
_ASCII_LUT = np.array([65, 67, 71, 84], dtype=np.uint8)
_RESET_BYTE = 78  # 'N' -> a single non-ACGT byte forces a window reset.

# One row per kept record. Offsets index codes.bin (bytes) and resets.bin
# (uint32 elements). Labels are small ints.
INDEX_DTYPE = np.dtype([
    ("code_off", "<u8"), ("nbytes", "<u4"), ("ncodes", "<u4"),
    ("reset_off", "<u8"), ("nresets", "<u4"), ("label", "<i2"),
])

_SUBSAMPLE_DENOM = 1_000_000


# --------------------------------------------------------------------------- #
# Subsampling (B): deterministic, content-hashed, ratio-preserving.
# --------------------------------------------------------------------------- #
def _raw_bytes(seq):
    if isinstance(seq, (bytes, bytearray)):
        return bytes(seq)
    return seq.encode("ascii", errors="ignore")


def _to_uint8(seq) -> np.ndarray:
    return np.frombuffer(_raw_bytes(seq), dtype=np.uint8)


def acgt_purity(seq) -> float:
    """Fraction of bytes that are upper-case ACGT (1.0 == perfectly clean)."""
    arr = _to_uint8(seq)
    if arr.size == 0:
        return 0.0
    return float(np.count_nonzero(_CODE_LUT[arr] != 255)) / float(arr.size)


def normalize_subsample(cfg):
    """Coerce a raw config dict into a canonical per-split spec dict, or return
    None when subsampling is disabled. Canonical form is what gets embedded in
    the cache signature, so it MUST be stable/JSON-serializable."""
    if not cfg or not cfg.get("enabled", False):
        return None
    seed = int(cfg.get("seed", 0))
    out = {"seed": seed}
    for split in ("train", "validation", "test"):
        raw = cfg.get(split, {}) or {}
        out[split] = {
            "rate": float(raw.get("rate", 1.0)),
            "min_purity": float(raw.get("min_acgt_purity", raw.get("min_purity", 0.0))),
            "min_len": int(raw.get("min_len", 0)),
        }
        # Only emit class_rates when actually used. The spec is embedded in the
        # feature-cache/pack signature, so unconditionally adding the key would
        # invalidate the existing ~6 TB of val/test cache for no reason.
        rates = raw.get("class_rates") or {}
        if rates:
            out[split]["class_rates"] = {
                str(k).lower(): float(v) for k, v in sorted(rates.items())}
    return out


def split_spec(subcfg, split):
    """Per-split {rate,min_purity,min_len,seed} from a normalized subsample dict.
    `split` is a train_ready split name ('train'/'validation'/'test')."""
    if subcfg is None:
        return {"rate": 1.0, "min_purity": 0.0, "min_len": 0, "seed": 0}
    spec = dict(subcfg.get(split, subcfg.get("train")))
    spec["seed"] = int(subcfg.get("seed", 0))
    return spec


def keep_sequence(seq, *, rate=1.0, min_purity=0.0, min_len=0, seed=0) -> bool:
    """Deterministic subsample decision for ONE sequence.

    * Quality gate first: drop fragments shorter than `min_len` or below
      `min_purity` ACGT fraction (junk / high-N).
    * Then a content-hashed uniform keep at probability `rate`. Because the same
      rate is applied within every class, CLASS PROPORTIONS are preserved; and
      because the decision is a pure function of the bytes, a restart reselects
      the identical subset (staged resume safe).
    """
    raw = _raw_bytes(seq)
    if len(raw) < int(min_len):
        return False
    # Decide on the UPPER-CASED bytes so a soft-masked (lowercase) fragment from
    # the FASTA/pack path and its upper-cased form on the train_models path get
    # the SAME purity score AND the SAME keep/drop draw. Otherwise the two paths
    # would disagree on which records exist. bytes.upper() is ASCII-only, O(n).
    upper = raw.upper()
    if min_purity > 0.0 and acgt_purity(upper) < float(min_purity):
        return False
    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    h = zlib.crc32(upper, int(seed) & 0xFFFFFFFF) % _SUBSAMPLE_DENOM
    return h < int(rate * _SUBSAMPLE_DENOM)


# --------------------------------------------------------------------------- #
# Per-class rates + class rebalancing.
#
# A single uniform rate preserves the corpus's class proportions, which for this
# corpus means archaea stays at ~0.11% of train while eukarya takes ~64%. That
# is fine for val/test (they must keep the TRUE prior or the reported metrics
# are meaningless) but starves the small classes during training. So train gets
# per-class rates; val/test keep the single uniform rate.
# --------------------------------------------------------------------------- #
_CLASS_SUFFIXES = ("_fr",)
_SEQ_EXTS = (".fasta", ".fa", ".fna", ".fasta.gz", ".fa.gz", ".fna.gz")
# The flat training layout abbreviates some class names ('plast_fr.fasta').
# Without this map that file would key as 'plast', silently miss its configured
# rate, and fall back to the uniform one -- a wrong-but-quiet rebalance.
_CLASS_ALIASES = {
    "plast": "plastids",
    "plastid": "plastids",
    "mito": "mitochondria",
    "mitochondrion": "mitochondria",
}


def class_of_path(path) -> str:
    """Canonical class key for a corpus FASTA path.

    The three subsample call sites see different filenames for the same class
    ('bacteria.fasta' in the corpus, 'bacteria_fr.fasta' in the flat layout).
    They MUST agree on the class key or they would apply different rates and
    disagree on which records exist.
    """
    name = os.path.basename(str(path))
    low = name.lower()
    for ext in sorted(_SEQ_EXTS, key=len, reverse=True):
        if low.endswith(ext):
            name = name[: -len(ext)]
            break
    for suf in _CLASS_SUFFIXES:
        if name.lower().endswith(suf):
            name = name[: -len(suf)]
            break
    low = name.lower()
    return _CLASS_ALIASES.get(low, low)


def rate_for(spec, cls) -> float:
    """Per-class rate, falling back to the split's uniform rate."""
    if not spec:
        return 1.0
    rates = spec.get("class_rates") or {}
    if cls in rates:
        return float(rates[cls])
    return float(spec.get("rate", 1.0))


def rate_for_path(spec, path) -> float:
    return rate_for(spec, class_of_path(path))


def balance_rates(counts, *, mode="none", target_total=None, base_rate=1.0,
                  max_rate=1.0, min_rows=0):
    """Per-class keep rates that reshape the class mix toward `mode`.

    counts: {class: available_rows}.
    mode:
      'none'   -- uniform base_rate (proportions preserved, current behaviour).
      'sqrt'   -- target share proportional to sqrt(count): a compromise that
                  lifts rare classes a lot while still letting abundant ones
                  dominate somewhat. This is the recommended setting.
      'linear' -- equal target rows per class (fully balanced, most aggressive).

    A rate is never allowed above `max_rate`: we can only DROP rows, never
    invent them, so a rare class simply keeps everything it has rather than
    being oversampled into duplicates.
    """
    counts = {str(k): int(v) for k, v in (counts or {}).items() if int(v) > 0}
    if not counts:
        return {}
    if mode == "none":
        return {c: float(base_rate) for c in counts}

    if mode == "sqrt":
        weights = {c: math.sqrt(n) for c, n in counts.items()}
    elif mode == "linear":
        weights = {c: 1.0 for c in counts}
    else:
        raise ValueError(f"unknown balance mode: {mode!r}")

    if target_total is None:
        target_total = int(sum(counts.values()) * float(base_rate))
    target_total = max(0, int(target_total))
    wsum = sum(weights.values()) or 1.0

    rates = {}
    for cls, n in counts.items():
        want = target_total * (weights[cls] / wsum)
        if min_rows:
            want = max(want, float(min_rows))
        # Cannot exceed what exists.
        rates[cls] = round(min(float(max_rate), want / float(n)), 6)
    return rates


def expected_rows(counts, rates):
    """Rows each class contributes under `rates` (for plan reporting)."""
    out = {}
    for cls, n in (counts or {}).items():
        out[str(cls)] = int(int(n) * float(rates.get(str(cls), 1.0)))
    return out


# --------------------------------------------------------------------------- #
# 2-bit encode / decode (D2). Lossless w.r.t. k-mer counting.
# --------------------------------------------------------------------------- #
def _pack2(codes: np.ndarray) -> bytes:
    n = int(codes.size)
    pad = (-n) % 4
    if pad:
        codes = np.concatenate([codes, np.zeros(pad, dtype=np.uint8)])
    v = codes.reshape(-1, 4)
    packed = (v[:, 0] << 6) | (v[:, 1] << 4) | (v[:, 2] << 2) | v[:, 3]
    return packed.astype(np.uint8).tobytes()


def _unpack2(packed, n: int) -> np.ndarray:
    b = np.frombuffer(packed, dtype=np.uint8)
    out = np.empty((b.size, 4), dtype=np.uint8)
    out[:, 0] = (b >> 6) & 3
    out[:, 1] = (b >> 4) & 3
    out[:, 2] = (b >> 2) & 3
    out[:, 3] = b & 3
    return out.reshape(-1)[:n]


def encode_seq(seq):
    """Return (packed_bytes, n_codes, resets_uint32).

    Only ACGT bases become 2-bit codes; each maximal run of non-ACGT bases
    BETWEEN two ACGT bases collapses to one reset point (index into the code
    stream where the rolling window must reset). Leading/trailing non-ACGT need
    no marker (the window is already empty). This mirrors count_kmers_into
    exactly, so decode->count == count on the original sequence.
    """
    arr = _to_uint8(seq)
    vals = _CODE_LUT[arr]
    valid = vals != 255
    vi = np.nonzero(valid)[0]
    codes = vals[valid].astype(np.uint8)
    n = int(codes.size)
    if n > 1:
        d = np.diff(vi)
        resets = (np.nonzero(d > 1)[0] + 1).astype(np.uint32)
    else:
        resets = np.zeros(0, dtype=np.uint32)
    return _pack2(codes), n, resets


def decode_to_ascii(packed, n_codes: int, resets) -> np.ndarray:
    """Reconstruct an ACGT byte array with a single 'N' inserted at every reset,
    ready for count_kmers_into (identical counts to the raw sequence)."""
    codes = _unpack2(packed, int(n_codes))
    ascii_arr = _ASCII_LUT[codes]
    resets = np.asarray(resets, dtype=np.int64)
    if resets.size:
        ascii_arr = np.insert(ascii_arr, resets, _RESET_BYTE)
    return ascii_arr


# --------------------------------------------------------------------------- #
# On-disk pack store (per stage/split): codes.bin + resets.bin + index.npy.
# --------------------------------------------------------------------------- #
def pack_dir(pack_root, stage, split) -> Path:
    return Path(pack_root) / stage / split


def _meta_path(pdir) -> Path:
    return Path(pdir) / "meta.json"


def pack_valid(pdir, inputs_sig, subcfg_split):
    """Return record count if a finished pack matches (inputs_sig, subcfg_split),
    else None. subcfg_split is the per-split spec (rate/min_purity/min_len/seed).
    """
    mp = _meta_path(pdir)
    if not mp.is_file():
        return None
    try:
        meta = json.loads(mp.read_text())
    except Exception:
        return None
    if meta.get("inputs_sig") != inputs_sig:
        return None
    if meta.get("subsample") != subcfg_split:
        return None
    for name in ("codes.bin", "resets.bin", "index.npy"):
        if not (Path(pdir) / name).is_file():
            return None
    return int(meta.get("n", -1))


def _fc():
    """Lazy import of the sibling module (avoids an import cycle)."""
    try:
        from . import featurize_cache as fc  # type: ignore
    except ImportError:  # top-level import in the sandbox/test harness
        import featurize_cache as fc  # type: ignore
    return fc


# ---- pack builder worker (one byte-range shard) --------------------------- #
def _pack_shard(task):
    sid, path, label, start, end, spec = task
    fc = _fc()
    # Per-class rate: the shard's path identifies its class, so rebalancing
    # needs no change to the task tuple or the resume manifest.
    rate = rate_for_path(spec, path)
    min_purity = spec["min_purity"]
    min_len = spec["min_len"]
    seed = spec["seed"]
    tmp = spec["tmp"]

    codes_buf = bytearray()
    resets_buf = bytearray()
    rows = []
    code_off = 0
    reset_off = 0
    kept = 0
    scanned_bytes = 0
    for seq, nb in fc._iter_fasta_shard_raw(path, start, end):
        scanned_bytes += nb
        if not keep_sequence(seq, rate=rate, min_purity=min_purity,
                             min_len=min_len, seed=seed):
            continue
        packed, ncodes, resets = encode_seq(seq)
        codes_buf += packed
        rr = np.asarray(resets, dtype=np.uint32)
        resets_buf += rr.tobytes()
        rows.append((code_off, len(packed), ncodes, reset_off, rr.size, label))
        code_off += len(packed)
        reset_off += int(rr.size)
        kept += 1

    idx = np.array(rows, dtype=INDEX_DTYPE) if rows else np.zeros(0, dtype=INDEX_DTYPE)
    with open(os.path.join(tmp, f"part{sid}.codes"), "wb") as f:
        f.write(bytes(codes_buf))
    with open(os.path.join(tmp, f"part{sid}.resets"), "wb") as f:
        f.write(bytes(resets_buf))
    np.save(os.path.join(tmp, f"part{sid}.index.npy"), idx)
    return sid, kept, scanned_bytes


def _read_manifest(mpath, inputs_sig, spec):
    if not Path(mpath).is_file():
        return None
    try:
        m = json.loads(Path(mpath).read_text())
    except Exception:
        return None
    if m.get("inputs_sig") != inputs_sig or m.get("subsample") != spec:
        return None
    return {int(s) for s in m.get("done", [])}


def _write_manifest(mpath, inputs_sig, spec, done):
    tmp = Path(mpath).with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"inputs_sig": inputs_sig, "subsample": spec,
                               "done": sorted(int(s) for s in done)}))
    os.replace(tmp, mpath)


def build_pack(pack_root, stage, split, inputs, inputs_sig, spec, *,
               workers=8, log=None, target_shards=None, progress_secs=20.0):
    """Build (or resume) the 2-bit pack for one (stage, split).

    `inputs` is the ordered [(path, label), ...] for the stage/split; `spec` is
    the per-split subsample spec (rate/min_purity/min_len/seed). Returns the
    kept record count. Resume is at BYTE-SHARD granularity and PERSISTS across
    process restarts: completed shards are recorded in .build/manifest.json, so
    a killed run continues from where it stopped -- exactly the staged resume
    the dense cache lacked.
    """
    log = log or (lambda _m: None)
    pdir = pack_dir(pack_root, stage, split)
    have = pack_valid(pdir, inputs_sig, spec)
    if have is not None:
        log(f"[pack {stage}/{split}] cached n={have:,}")
        return have

    pdir.mkdir(parents=True, exist_ok=True)
    build = pdir / ".build"
    manifest = build / "manifest.json"

    fc = _fc()
    entries = [(p, l) for p, l in inputs]
    total_in = sum(os.path.getsize(p) for p, _l in entries) or 1
    if target_shards is None:
        target_shards = max(workers * 4, min(4096, max(1, total_in // (1 << 30))))
    shards = fc.plan_shards(entries, target_shards)
    total_shards = len(shards)

    done = _read_manifest(manifest, inputs_sig, spec)
    if done is None:
        shutil.rmtree(build, ignore_errors=True)
        done = set()
    build.mkdir(parents=True, exist_ok=True)

    spec_w = dict(spec)
    spec_w["tmp"] = str(build)
    todo = []
    for sid, path, label, s, e in shards:
        part_idx = build / f"part{sid}.index.npy"
        if sid in done and part_idx.is_file():
            continue
        todo.append((sid, path, label, s, e, spec_w))

    if todo:
        log(f"[pack {stage}/{split}] need {len(todo)}/{total_shards} shards "
            f"(workers={workers}, resumed={len(done)} done)")
        shard_nbytes = {sid: max(0, e - s) for sid, _p, _l, s, e in shards}
        total_planned_bytes = sum(shard_nbytes.values()) or 1
        resumed_bytes = sum(shard_nbytes.get(sid, 0) for sid in done)
        scanned_bytes = resumed_bytes
        kept_this_run = 0
        started = time.monotonic()
        last_report = started

        def record_result(result):
            nonlocal scanned_bytes, kept_this_run, last_report
            sid, kept, shard_scanned = result
            done.add(sid)
            scanned_bytes += int(shard_scanned)
            kept_this_run += int(kept)
            _write_manifest(manifest, inputs_sig, spec, done)

            now = time.monotonic()
            if now - last_report >= float(progress_secs) or len(done) == total_shards:
                elapsed = max(now - started, 1e-9)
                new_scanned = max(0, scanned_bytes - resumed_bytes)
                speed = new_scanned / elapsed
                remaining = max(0, total_planned_bytes - scanned_bytes)
                eta = remaining / speed if speed > 0 else float("inf")
                pct = 100.0 * len(done) / max(1, total_shards)
                eta_text = f"{eta / 60:.1f}m" if eta != float("inf") else "?"
                log(
                    f"[pack {stage}/{split}] {len(done)}/{total_shards} shards "
                    f"({pct:.1f}%), scanned={scanned_bytes / (1 << 30):.1f}/"
                    f"{total_planned_bytes / (1 << 30):.1f} GiB, "
                    f"kept(this run)={kept_this_run:,}, "
                    f"speed={speed / (1 << 20):.1f} MiB/s, "
                    f"elapsed={elapsed / 60:.1f}m, ETA={eta_text}"
                )
                last_report = now

        if workers <= 1 or len(todo) <= 1:
            for task in todo:
                record_result(_pack_shard(task))
        else:
            import multiprocessing as mp
            ctx = mp.get_context("fork")
            with ctx.Pool(processes=workers) as pool:
                for result in pool.imap_unordered(_pack_shard, todo):
                    record_result(result)

    # Finalize: concat parts in sid order, fixing offsets into global space.
    order = sorted(sid for sid, *_ in shards)
    idx_parts = []
    code_running = 0
    reset_running = 0
    codes_tmp = pdir / "codes.bin.tmp"
    resets_tmp = pdir / "resets.bin.tmp"
    merge_total_bytes = sum(
        (build / f"part{sid}.codes").stat().st_size
        + (build / f"part{sid}.resets").stat().st_size
        for sid in order
    ) or 1
    merge_written = 0
    merge_started = time.monotonic()
    merge_last_report = merge_started
    log(f"[pack {stage}/{split}] finalizing {len(order)} shards, "
        f"packed={merge_total_bytes / (1 << 30):.1f} GiB")
    with open(codes_tmp, "wb") as cf, open(resets_tmp, "wb") as rf:
        for merge_i, sid in enumerate(order, 1):
            pc = build / f"part{sid}.codes"
            pr = build / f"part{sid}.resets"
            pi = build / f"part{sid}.index.npy"
            pc_size = pc.stat().st_size
            pr_size = pr.stat().st_size
            idx = np.load(pi)
            if idx.size:
                idx = idx.copy()
                idx["code_off"] = idx["code_off"] + code_running
                idx["reset_off"] = idx["reset_off"] + reset_running
                idx_parts.append(idx)
            with open(pc, "rb") as f:
                shutil.copyfileobj(f, cf, length=1 << 20)
            with open(pr, "rb") as f:
                shutil.copyfileobj(f, rf, length=1 << 20)
            code_running += pc_size
            reset_running += pr_size // 4
            merge_written += pc_size + pr_size
            now = time.monotonic()
            if now - merge_last_report >= float(progress_secs) or merge_i == len(order):
                elapsed = max(now - merge_started, 1e-9)
                speed = merge_written / elapsed
                remaining = max(0, merge_total_bytes - merge_written)
                eta = remaining / speed if speed > 0 else 0.0
                log(
                    f"[pack {stage}/{split}] finalize {merge_i}/{len(order)} "
                    f"({100.0 * merge_i / max(1, len(order)):.1f}%), "
                    f"written={merge_written / (1 << 30):.1f}/"
                    f"{merge_total_bytes / (1 << 30):.1f} GiB, "
                    f"speed={speed / (1 << 20):.1f} MiB/s, ETA={eta / 60:.1f}m"
                )
                merge_last_report = now
    os.replace(codes_tmp, pdir / "codes.bin")
    os.replace(resets_tmp, pdir / "resets.bin")
    index = (np.concatenate(idx_parts) if idx_parts
             else np.zeros(0, dtype=INDEX_DTYPE))
    np.save(pdir / "index.npy", index)
    n = int(index.size)
    _meta_path(pdir).write_text(json.dumps({
        "n": n, "inputs_sig": inputs_sig, "subsample": spec,
        "stage": stage, "split": split,
    }))
    shutil.rmtree(build, ignore_errors=True)
    log(f"[pack {stage}/{split}] done n={n:,} "
        f"(codes={code_running} B, resets={reset_running})")
    return n


# --------------------------------------------------------------------------- #
# Reading records back for featurization.
# --------------------------------------------------------------------------- #
def open_pack(pdir):
    """Open a finished pack read-only. Returns (index, codes, resets, n)."""
    pdir = Path(pdir)
    index = np.load(pdir / "index.npy", mmap_mode="r")
    n = int(index.size)
    cpath = pdir / "codes.bin"
    rpath = pdir / "resets.bin"
    codes = (np.memmap(cpath, dtype=np.uint8, mode="r")
             if cpath.stat().st_size else np.zeros(0, dtype=np.uint8))
    resets = (np.memmap(rpath, dtype=np.uint32, mode="r")
              if rpath.stat().st_size else np.zeros(0, dtype=np.uint32))
    return index, codes, resets, n


def record_ascii(index, codes, resets, i):
    """Reconstruct record i as an ACGT(+N) byte array; return (ascii, label)."""
    row = index[i]
    off = int(row["code_off"])
    nb = int(row["nbytes"])
    nc = int(row["ncodes"])
    ro = int(row["reset_off"])
    nr = int(row["nresets"])
    packed = bytes(codes[off:off + nb])
    rs = np.asarray(resets[ro:ro + nr])
    return decode_to_ascii(packed, nc, rs), int(row["label"])


def iter_records(pdir, start=0, end=None):
    """Yield (ascii_uint8, label) for records [start, end)."""
    index, codes, resets, n = open_pack(pdir)
    if end is None or end > n:
        end = n
    for i in range(start, end):
        yield record_ascii(index, codes, resets, i)


def labels_array(pdir) -> np.ndarray:
    index = np.load(Path(pdir) / "index.npy", mmap_mode="r")
    return np.asarray(index["label"], dtype=np.int64)

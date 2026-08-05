#!/usr/bin/env python3
"""Deterministically sample an existing fragment corpus by BASE-PAIR budget.

This is the data-only balancing checkpoint between regroup and training:

    corpus_ready_<corpus_tag>
      -> sample_fragments_by_bp.py
      -> corpus_ready_<version_tag>_bpbalanced
      -> TF-IDF / seqpack / feature cache / train

Why bp rather than record counts
--------------------------------
T6 mixes 1/2/3/5/10 kb fragments. Equal record counts therefore do not mean
equal sequence mass. This script plans against the SUM OF FRAGMENT LENGTHS and
then applies a deterministic uniform keep probability inside each stratum.
Expected retained bp, not row count, follows the requested priors.

WHAT CHANGED IN v2.1.3 (and why)
--------------------------------
v2.1.2 balanced ONLY the train split and symlinked validation straight to the
raw corpus. That validation split was 66.4 M records, 98.6% eukarya. Every
downstream decision that used validation -- the hyper-parameter ranking, and
(with final_include_validation) the final fit itself -- was therefore optimised
against a distribution where "always answer eukarya" scores 0.988 accuracy. The
published v2.1.2 pack did exactly that: 6,015,853 of 6,015,856 benchmark
fragments came back `eukarya` with p=1.000000.

So:

  * `--balance-splits train,validation` (default) balances EACH listed split
    independently, from its OWN available bp. No records cross a split
    boundary, so this cannot leak train data into validation.
  * `--eval-mode` now only decides how the REMAINING splits (test) are
    mirrored. Test deliberately keeps the natural prior: it is the honest
    operating point.
  * `--require-clades` fails loudly when a weighted eukaryotic clade has zero
    available bp instead of silently redistributing its budget. In v2.1.2
    alveolata, stramenopiles, land_plant and eukarya_other were all empty --
    35% of the eukaryotic budget quietly went to the clades that remained.
  * `--taxdump-dir` / `--accession-taxid-tsv` resolve eukaryotic clades from
    taxids. The `sg=` header field collapses Opisthokonta-Metazoa,
    Archaeplastida and Protist(...) into one bucket each, which is where those
    empty clades came from.

The dominant eukarya FASTA is split into ordered byte ranges, processed in
parallel, and merged in original record order without changing deterministic
selection results.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
import shutil
import sys
import zlib
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tiara2 import labels  # noqa: E402

CLASSES = ("archaea", "bacteria", "eukarya", "mitochondria", "plastids")
ACGT_TABLE = bytes(1 if i in b"ACGTacgt" else 0 for i in range(256))

# Splits that may be balanced. `test` is deliberately NOT balanceable: it is
# the only split that still carries the real-world class prior.
BALANCEABLE_SPLITS = ("train", "validation")

# Tiara Supplementary Table S3, normalized.
TIARA_PRIORS = {
    "archaea": 0.173368670115289,
    "bacteria": 0.4243864390266533,
    "eukarya": 0.31667691201373266,
    "mitochondria": 0.027914704586049317,
    "plastids": 0.05765327425827554,
}

# Broader than Tiara1 while protecting fungi and microbial eukaryotes.
# These weights only split the fixed Eukarya budget; they do not change the
# top-level 31.67% Eukarya prior.
DEFAULT_EUK_WEIGHTS = {
    "fungi": 0.30,
    "land_plant": 0.12,
    "algae": 0.08,
    "metazoa_vertebrate": 0.08,
    "metazoa_invertebrate": 0.08,
    "alveolata": 0.10,
    "stramenopiles": 0.10,
    "other_protist": 0.10,
    "eukarya_other": 0.04,
}


def parse_map(text: str | None, defaults: dict[str, float]) -> dict[str, float]:
    if not text:
        out = dict(defaults)
    else:
        p = Path(text)
        raw = json.loads(p.read_text()) if p.is_file() else json.loads(text)
        out = {str(k).strip().lower(): float(v) for k, v in raw.items()}
    if any((not math.isfinite(v) or v < 0) for v in out.values()):
        raise SystemExit("weights/priors must be finite and >= 0")
    total = sum(out.values())
    if total <= 0:
        raise SystemExit("weights/priors sum to zero")
    return {k: v / total for k, v in out.items()}


def parse_splits(text: str) -> list[str]:
    """Parse --balance-splits into a canonical, de-duplicated list.

    Rejects `test` explicitly: balancing the held-out set would destroy the
    only unbiased estimate of real-world behaviour we have.
    """
    wanted = [x.strip().lower() for x in str(text).split(",") if x.strip()]
    if not wanted:
        raise SystemExit("--balance-splits must name at least one split")
    bad = [x for x in wanted if x not in BALANCEABLE_SPLITS]
    if bad:
        raise SystemExit(
            f"--balance-splits accepts only {', '.join(BALANCEABLE_SPLITS)}; "
            f"got {', '.join(bad)}. `test` must keep the natural class prior.")
    return [s for s in BALANCEABLE_SPLITS if s in wanted]


def purity(seq_lines: list[bytes]) -> float:
    n = good = 0
    for line in seq_lines:
        n += len(line)
        # One C-level translation instead of four complete str.count scans.
        good += sum(line.translate(ACGT_TABLE))
    return good / n if n else 0.0


def seq_len(seq_lines: list[bytes]) -> int:
    return sum(len(x) for x in seq_lines)


def iter_fasta_range(path: Path, start: int, end: int, buffer_bytes: int):
    """Yield records whose header offset is in [start, end).

    Chunks may start in the middle of a record.  Each worker scans forward to
    the next FASTA header, while the previous worker finishes the record that
    began before its end boundary.  Therefore every record is emitted exactly
    once and concatenating chunk outputs by chunk index preserves serial order.
    Sequence lines stay as bytes to avoid decoding terabytes of bases.
    """
    size = path.stat().st_size
    start = max(0, min(int(start), size))
    end = max(start, min(int(end), size))
    with open(path, "rb", buffering=buffer_bytes) as fh:
        fh.seek(start)
        if start > 0:
            # If start is inside a line, finish that line first.
            fh.seek(start - 1)
            if fh.read(1) != b"\n":
                fh.readline()
            # We may now be in sequence data; find the next record header.
            while True:
                pos = fh.tell()
                line = fh.readline()
                if not line:
                    return
                if line.startswith(b">"):
                    header_pos = pos
                    header_line = line
                    break
        else:
            while True:
                pos = fh.tell()
                line = fh.readline()
                if not line:
                    return
                if line.startswith(b">"):
                    header_pos = pos
                    header_line = line
                    break

        while header_line:
            if header_pos >= end:
                return
            header_b = header_line[1:].rstrip(b"\r\n")
            seq = []
            next_header = b""
            next_pos = size
            while True:
                pos = fh.tell()
                line = fh.readline()
                if not line:
                    break
                if line.startswith(b">"):
                    next_header = line
                    next_pos = pos
                    break
                line = line.strip()
                if line:
                    seq.append(line)
            header = header_b.decode("utf-8", "surrogateescape")
            yield header.split(None, 1)[0], header, seq
            header_line = next_header
            header_pos = next_pos


def make_tasks(files: dict[str, Path], euk_workers: int):
    """Create byte-range tasks; only the dominant Eukarya file is split."""
    tasks = []
    for cls, path in files.items():
        size = path.stat().st_size
        chunks = euk_workers if cls == "eukarya" else 1
        chunks = max(1, min(chunks, max(1, size)))
        for index in range(chunks):
            start = size * index // chunks
            end = size * (index + 1) // chunks
            tasks.append((cls, index, chunks, str(path), start, end))
    return tasks


# --------------------------------------------------------------------------- #
# Eukaryotic clade resolution
# --------------------------------------------------------------------------- #
def build_acc2clade(taxdump_dir: str, acc_tsv: str, log=print) -> dict[str, str]:
    """Build `accession -> clade` ONCE, in the parent process.

    Loading nodes.dmp costs ~200 MiB and several seconds; doing it inside every
    worker would multiply that by --workers. Instead the parent resolves the
    accession table once and ships a plain dict to the workers.

    Degrades to an empty mapping (header `sg=` resolution only) whenever the
    taxdump or the accession table is missing or unusable -- and SAYS SO, so a
    silent fallback can never be mistaken for taxid-grade resolution again.
    """
    if not taxdump_dir and not acc_tsv:
        return {}
    if not acc_tsv:
        log("[clade] taxdump given but no --accession-taxid-tsv: cannot map "
            "accessions to taxids; falling back to header sg=")
        return {}
    tsv = Path(acc_tsv).expanduser()
    if not tsv.is_file():
        log(f"[clade] accession table not found: {tsv}; falling back to "
            f"header sg=")
        return {}

    try:
        from tiara2 import taxonomy as _tax
        found = _tax.find_taxdump(taxdump_dir) if taxdump_dir else None
        if found is None:
            log(f"[clade] no nodes.dmp/names.dmp under {taxdump_dir!r}; "
                f"falling back to header sg=")
            return {}
        tax = _tax.Taxonomy.from_taxdump(found)
    except Exception as exc:                      # never block the pipeline
        log(f"[clade] taxonomy unavailable ({exc!r}); falling back to header sg=")
        return {}

    acc_keys = ("accession", "assembly_accession", "asm_accession",
                "genome", "gca", "gcf")
    tax_keys = ("taxid", "species_taxid", "tax_id")
    mapping: dict[str, str] = {}
    rows = 0
    try:
        with open(tsv, newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            names = {(n or "").strip().lower(): n for n in (reader.fieldnames or [])}
            acc_col = next((names[k] for k in acc_keys if k in names), None)
            tax_col = next((names[k] for k in tax_keys if k in names), None)
            if not acc_col or not tax_col:
                log(f"[clade] {tsv} has no accession/taxid columns "
                    f"(saw {list(names)}); falling back to header sg=")
                return {}
            for row in reader:
                acc = (row.get(acc_col) or "").strip()
                raw = (row.get(tax_col) or "").strip()
                if not acc or not raw:
                    continue
                rows += 1
                info = tax.info(raw)
                if getattr(info, "ok", False) and getattr(info, "clade", None):
                    mapping[acc] = str(info.clade)
    except OSError as exc:
        log(f"[clade] cannot read {tsv} ({exc}); falling back to header sg=")
        return {}
    log(f"[clade] resolved {len(mapping):,}/{rows:,} accessions to clades "
        f"via taxdump")
    return mapping


def euk_bucket(fragment_id: str, header: str,
               acc2clade: dict[str, str] | None = None) -> tuple[str, str]:
    """Return (clade, how) for a eukaryotic fragment.

    Order of preference: accession -> taxid -> clade (exact), then the header
    `sg=`/`group=` fields, then the documented ambiguous collapses.
    """
    if acc2clade:
        acc = labels.accession_from_frag_id(fragment_id)
        if acc:
            clade = acc2clade.get(acc)
            if clade:
                return str(clade), "taxid"
    meta = labels.parse_header(header)
    clade, how = labels.euk_subclass(
        meta.get("label", ""), meta.get("sg", ""),
        group=meta.get("group", ""), taxid=meta.get("taxid"), taxonomy=None,
    )
    # Old v2.1 headers contain broad sg values that are intentionally ambiguous.
    # Keep those visible instead of silently pretending they were resolved --
    # `ambiguous_sg_collapsed` in the report is the signal that the clade
    # budgets below are only as good as the header text.
    if clade is None:
        sg = meta.get("sg", "").strip().lower()
        if sg == "opisthokonta-metazoa":
            clade = "metazoa_invertebrate"
            how = "ambiguous_sg_collapsed"
        elif sg == "archaeplastida":
            clade = "algae"
            how = "ambiguous_sg_collapsed"
        elif sg == "protist(sar/excavata/amoebozoa)":
            clade = "other_protist"
            how = "ambiguous_sg_collapsed"
        else:
            clade = "eukarya_other"
            how = "fallback"
    return str(clade), str(how)


def stratum_for(class_name: str, fragment_id: str, header: str,
                acc2clade: dict[str, str] | None = None) -> tuple[str, str]:
    if class_name != "eukarya":
        return class_name, "class"
    clade, how = euk_bucket(fragment_id, header, acc2clade)
    return f"eukarya/{clade}", how


def source_files(root: Path, split: str) -> dict[str, Path]:
    out = {}
    aliases = {"plastids": ("plastids.fasta", "plastid.fasta", "plast.fasta"),
               "mitochondria": ("mitochondria.fasta", "mito.fasta")}
    for cls in CLASSES:
        candidates = aliases.get(cls, (f"{cls}.fasta",))
        for name in candidates:
            p = root / split / name
            # .is_file() follows symlinks, so a symlinked corpus works, but a
            # BROKEN symlink is correctly reported as missing.
            if p.is_file():
                out[cls] = p
                break
    missing = [c for c in CLASSES if c not in out]
    if missing:
        raise SystemExit(
            f"missing {split} FASTA(s) under {root/split}: {', '.join(missing)} "
            f"(a broken symlink counts as missing)")
    return out


def _census_one(payload):
    (cls, index, chunks, path_s, start, end, min_len, min_purity,
     buffer_bytes, acc2clade) = payload
    path = Path(path_s)
    stats = defaultdict(lambda: {"records": 0, "bp": 0})
    resolution = defaultdict(int)
    for fid, header, seq in iter_fasta_range(path, start, end, buffer_bytes):
        n = seq_len(seq)
        if n < min_len or (min_purity > 0 and purity(seq) < min_purity):
            continue
        bucket, how = stratum_for(cls, fid, header, acc2clade)
        stats[bucket]["records"] += 1
        stats[bucket]["bp"] += n
        resolution[f"{bucket}\t{how}"] += 1
    return cls, index, chunks, dict(stats), dict(resolution)


def census(files: dict[str, Path], *, min_len: int, min_purity: float,
           workers: int = 1, euk_workers: int = 1,
           buffer_bytes: int = 32 << 20,
           acc2clade: dict[str, str] | None = None,
           label: str = "census"):
    """Census FASTA byte ranges in parallel."""
    payloads = [(*task, min_len, min_purity, buffer_bytes, acc2clade)
                for task in make_tasks(files, euk_workers)]
    results = []
    if workers <= 1:
        for payload in payloads:
            print(f"[{label}] {payload[0]} chunk {payload[1]+1}/{payload[2]}",
                  file=sys.stderr)
            results.append(_census_one(payload))
    else:
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=min(workers, len(payloads))) as pool:
            futures = {pool.submit(_census_one, p): (p[0], p[1], p[2])
                       for p in payloads}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                print(f"[{label}] completed {result[0]} "
                      f"chunk {result[1]+1}/{result[2]}",
                      file=sys.stderr, flush=True)
    merged_stats = defaultdict(lambda: {"records": 0, "bp": 0})
    merged_resolution = defaultdict(int)
    for _cls, _index, _chunks, local_stats, local_resolution in results:
        for bucket, values in local_stats.items():
            merged_stats[bucket]["records"] += values["records"]
            merged_stats[bucket]["bp"] += values["bp"]
        for key, value in local_resolution.items():
            merged_resolution[key] += value
    return dict(merged_stats), dict(merged_resolution)


def allocate_capped(total: int, weights: dict[str, float], capacities: dict[str, int]):
    """Weighted water-filling: redistribute deficits from capacity-limited bins."""
    total = max(0, int(total))
    keys = [k for k, w in weights.items() if w > 0]
    alloc = {k: 0.0 for k in keys}
    active = set(keys)
    remaining = float(total)
    while active and remaining > 0.5:
        wsum = sum(weights[k] for k in active)
        if wsum <= 0:
            break
        capped = []
        proposals = {k: remaining * weights[k] / wsum for k in active}
        for k, want in proposals.items():
            room = max(0.0, float(capacities.get(k, 0)) - alloc[k])
            if want >= room:
                alloc[k] += room
                remaining -= room
                capped.append(k)
        if not capped:
            for k, want in proposals.items():
                alloc[k] += want
            remaining = 0.0
            break
        active.difference_update(capped)
    rounded = {k: min(int(capacities.get(k, 0)), int(round(v))) for k, v in alloc.items()}
    # Correct harmless rounding drift without exceeding capacities.
    drift = min(total, sum(capacities.get(k, 0) for k in keys)) - sum(rounded.values())
    if drift > 0:
        for k in sorted(keys, key=lambda x: capacities.get(x, 0) - rounded[x], reverse=True):
            add = min(drift, capacities.get(k, 0) - rounded[k])
            rounded[k] += add
            drift -= add
            if drift == 0:
                break
    return rounded


def clade_coverage(stats: dict, euk_weights: dict[str, float]) -> dict:
    """Which weighted eukaryotic clades actually have data?

    v2.1.2 silently redistributed the budget of four empty clades (35% of the
    eukaryotic mass). This makes that visible, and with --require-clades fatal.
    """
    missing, present = [], []
    redistributed = 0.0
    for clade, weight in euk_weights.items():
        if weight <= 0:
            continue
        have = stats.get(f"eukarya/{clade}", {}).get("bp", 0)
        if have > 0:
            present.append(clade)
        else:
            missing.append(clade)
            redistributed += float(weight)
    return {"missing": sorted(missing), "present": sorted(present),
            "redistributed_weight": round(redistributed, 6)}


def make_plan(stats, priors, euk_weights, target_total_bp: int | None):
    top_available = {}
    for cls in CLASSES:
        if cls == "eukarya":
            top_available[cls] = sum(v["bp"] for k, v in stats.items() if k.startswith("eukarya/"))
        else:
            top_available[cls] = stats.get(cls, {}).get("bp", 0)
    if target_total_bp is None:
        feasible = [top_available[c] / p for c, p in priors.items() if p > 0]
        target_total_bp = int(min(feasible))
    top_target = {c: int(round(target_total_bp * priors[c])) for c in priors}
    for cls in top_target:
        if top_target[cls] > top_available.get(cls, 0):
            raise SystemExit(
                f"target total is infeasible without oversampling: {cls} needs "
                f"{top_target[cls]:,} bp, has {top_available.get(cls,0):,}; "
                "omit --target-total-bp to use the maximum strict budget")

    plan = {}
    for cls in CLASSES:
        if cls != "eukarya":
            have = stats.get(cls, {}).get("bp", 0)
            want = top_target[cls]
            plan[cls] = {"available_bp": have, "target_bp": want,
                         "rate": want / have if have else 0.0}

    euk_caps = {f"eukarya/{k}": stats.get(f"eukarya/{k}", {}).get("bp", 0)
                for k in euk_weights}
    euk_w = {f"eukarya/{k}": v for k, v in euk_weights.items()}
    euk_alloc = allocate_capped(top_target["eukarya"], euk_w, euk_caps)
    if sum(euk_alloc.values()) < top_target["eukarya"]:
        raise SystemExit("eukaryotic subclasses cannot fill the requested Eukarya bp budget")
    for bucket, want in euk_alloc.items():
        have = stats.get(bucket, {}).get("bp", 0)
        plan[bucket] = {"available_bp": have, "target_bp": want,
                        "rate": want / have if have else 0.0}
    return target_total_bp, top_available, top_target, plan


def keep_id(fragment_id: str, rate: float, seed: int) -> bool:
    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    token = f"{seed}:{fragment_id}".encode("utf-8", "surrogatepass")
    return (zlib.crc32(token) & 0xFFFFFFFF) < int(rate * (1 << 32))


def _sample_one(payload):
    (cls, index, chunks, src_s, start, end, shard_root_s, plan, seed,
     min_len, min_purity, buffer_bytes, acc2clade) = payload
    src = Path(src_s)
    shard_root = Path(shard_root_s)
    actual = defaultdict(lambda: {"records": 0, "bp": 0})
    class_dir = shard_root / cls
    class_dir.mkdir(parents=True, exist_ok=True)
    dst = class_dir / f"{index:05d}.fasta"
    tmp = class_dir / f".{index:05d}.tmp.{os.getpid()}"
    try:
        with open(tmp, "wb", buffering=buffer_bytes) as w:
            for fid, header, seq in iter_fasta_range(
                    src, start, end, buffer_bytes):
                n = seq_len(seq)
                if n < min_len or (min_purity > 0 and purity(seq) < min_purity):
                    continue
                bucket, _how = stratum_for(cls, fid, header, acc2clade)
                rate = plan.get(bucket, {"rate": 0.0})["rate"]
                if not keep_id(fid, rate, seed):
                    continue
                w.write(b">" + header.encode("utf-8", "surrogateescape") + b"\n")
                for line in seq:
                    w.write(line + b"\n")
                actual[bucket]["records"] += 1
                actual[bucket]["bp"] += n
        os.replace(tmp, dst)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    return cls, index, chunks, str(dst), dict(actual)


def sample_split(files, out_root, plan, *, split, seed, min_len, min_purity,
                 workers: int = 1, euk_workers: int = 1,
                 buffer_bytes: int = 32 << 20,
                 acc2clade: dict[str, str] | None = None):
    """Sample one split's chunks concurrently, then merge in byte order.

    The shard directory is namespaced per split, so balancing train and
    validation in the same run can never mix or clobber each other's shards.
    """
    shard_root = out_root / f".sampling_shards_{split}"
    if shard_root.exists():
        shutil.rmtree(shard_root)
    shard_root.mkdir(parents=True)
    payloads = [(*task, str(shard_root), plan, seed, min_len, min_purity,
                 buffer_bytes, acc2clade)
                for task in make_tasks(files, euk_workers)]
    results = []
    out_dir = out_root / split
    try:
        if workers <= 1:
            for payload in payloads:
                results.append(_sample_one(payload))
        else:
            with concurrent.futures.ProcessPoolExecutor(
                    max_workers=min(workers, len(payloads))) as pool:
                futures = {pool.submit(_sample_one, p): (p[0], p[1], p[2])
                           for p in payloads}
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    results.append(result)
                    print(f"[sample:{split}] completed {result[0]} "
                          f"chunk {result[1]+1}/{result[2]}",
                          file=sys.stderr, flush=True)

        actual = defaultdict(lambda: {"records": 0, "bp": 0})
        by_class = defaultdict(list)
        for cls, index, chunks, shard_s, local in results:
            by_class[cls].append((index, chunks, Path(shard_s)))
            for bucket, values in local.items():
                actual[bucket]["records"] += values["records"]
                actual[bucket]["bp"] += values["bp"]

        # CRITICAL: if a previous run left this split as a SYMLINK to the raw
        # corpus, writing through it would overwrite the source data. Break the
        # link first, then create a real directory.
        if out_dir.is_symlink() or (out_dir.exists() and not out_dir.is_dir()):
            out_dir.unlink()
        out_dir.mkdir(parents=True, exist_ok=True)
        for cls in CLASSES:
            parts = sorted(by_class[cls])
            if not parts:
                raise RuntimeError(f"no sampling shards produced for {cls}")
            expected = parts[0][1]
            if len(parts) != expected or [x[0] for x in parts] != list(range(expected)):
                raise RuntimeError(f"incomplete sampling shards for {cls}")
            dst = out_dir / f"{cls}.fasta"
            tmp = out_dir / f".{cls}.fasta.merge.tmp"
            print(f"[merge:{split}] {cls}: {len(parts)} ordered shard(s)",
                  file=sys.stderr, flush=True)
            with open(tmp, "wb", buffering=buffer_bytes) as w:
                for _index, _chunks, shard in parts:
                    with open(shard, "rb", buffering=buffer_bytes) as r:
                        shutil.copyfileobj(r, w, length=buffer_bytes)
                    shard.unlink()
            os.replace(tmp, dst)
        return dict(actual)
    except BaseException:
        if out_dir.is_dir() and not out_dir.is_symlink():
            for tmp in out_dir.glob(".*.merge.tmp"):
                try:
                    tmp.unlink()
                except OSError:
                    pass
        raise
    finally:
        shutil.rmtree(shard_root, ignore_errors=True)


def link_eval(src_root: Path, out_root: Path, mode: str, splits) -> list[str]:
    """Mirror the splits that were NOT balanced (normally just `test`).

    Every symlink targets `src.resolve()`, so the result is always a direct
    link to the real directory -- never a link to another link, which is how
    stale corpora kept resurfacing in earlier rounds.
    """
    mirrored = []
    for split in splits:
        src = src_root / split
        if not src.exists():
            continue
        dst = out_root / split
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        elif dst.is_dir():
            shutil.rmtree(dst)
        if mode == "symlink":
            dst.symlink_to(src.resolve(), target_is_directory=True)
        elif mode == "copy":
            shutil.copytree(src, dst)
        else:
            continue
        mirrored.append(split)
    return mirrored


def write_reports(out_root: Path, per_split: dict, args, extra: dict) -> None:
    """Write sampling_plan.json + sampling_report.tsv.

    Schema 2 is per-split. For backwards compatibility with `tiara2 status`
    and `scripts/build_run_report.py`, the TRAIN numbers are also mirrored at
    the top level exactly where schema 1 put them.
    """
    report = {
        "schema": 2,
        "source_root": str(Path(args.source_root).resolve()),
        "seed": args.seed,
        "balanced_splits": list(per_split),
        "eval_mode": args.eval_mode,
        "parallel": {
            "workers": args.workers,
            "euk_workers": args.euk_workers,
            "io_buffer_mb": args.io_buffer_mb,
            "strategy": "ordered_byte_ranges",
        },
        "quality": {"min_len": args.min_len,
                    "min_acgt_purity": args.min_acgt_purity},
        "splits": {},
    }
    report.update(extra or {})

    tsv_rows = []
    for split, data in per_split.items():
        stats = data["stats"]
        plan = data["plan"]
        actual = data["actual"]
        block = {
            "target_total_bp": sum(data["top_target"].values()),
            "top_available_bp": data["top_available"],
            "top_target_bp": data["top_target"],
            "euk_resolution": data["resolution"],
            "clade_coverage": data["coverage"],
            "strata": {},
        }
        for k in sorted(set(stats) | set(plan) | set(actual)):
            row = {
                "available_records": stats.get(k, {}).get("records", 0),
                "available_bp": stats.get(k, {}).get("bp", 0),
                "target_bp": plan.get(k, {}).get("target_bp", 0),
                "rate": plan.get(k, {}).get("rate", 0.0),
                "kept_records": actual.get(k, {}).get("records", 0),
                "kept_bp": actual.get(k, {}).get("bp", 0),
            }
            block["strata"][k] = row
            target = row["target_bp"]
            err = 100.0 * (row["kept_bp"] - target) / target if target else 0.0
            tsv_rows.append({"split": split, "stratum": k, **row,
                             "bp_error_pct": f"{err:.4f}"})
        report["splits"][split] = block

    # ---- schema-1 compatibility mirror (train) ---------------------------
    primary = "train" if "train" in report["splits"] else next(
        iter(report["splits"]), None)
    if primary:
        mirror = report["splits"][primary]
        report["target_total_bp"] = mirror["target_total_bp"]
        report["top_available_bp"] = mirror["top_available_bp"]
        report["top_target_bp"] = mirror["top_target_bp"]
        report["strata"] = mirror["strata"]
        report["euk_resolution"] = mirror["euk_resolution"]

    plan_path = out_root / args.plan_file
    report_path = out_root / args.report_file
    plan_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    with open(report_path, "w", newline="") as fh:
        fields = ["split", "stratum", "available_records", "available_bp",
                  "target_bp", "rate", "kept_records", "kept_bp",
                  "bp_error_pct"]
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for row in tsv_rows:
            w.writerow(row)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-root", required=True,
                    help="input corpus root containing <split>/{class}.fasta")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--balance-splits", default="train,validation",
                    help="splits to balance independently (default: "
                         "train,validation). `test` is never balanced.")
    ap.add_argument("--target-total-bp", type=int, default=None,
                    help="strict total bp budget PER SPLIT; "
                         "default=max feasible without oversampling")
    ap.add_argument("--class-priors",
                    help="JSON string/file; default=Tiara S3 top-level priors")
    ap.add_argument("--euk-weights",
                    help="JSON string/file; default=broadened Tiara2 euk weights")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-len", type=int, default=1000)
    ap.add_argument("--min-acgt-purity", type=float, default=0.9)
    ap.add_argument("--eval-mode", choices=("symlink", "copy", "none"),
                    default="symlink",
                    help="how to mirror the splits that are NOT balanced "
                         "(default: symlink)")
    ap.add_argument("--taxdump-dir", default="",
                    help="NCBI taxdump (nodes.dmp/names.dmp) for taxid-grade "
                         "eukaryotic clade resolution")
    ap.add_argument("--accession-taxid-tsv", default="",
                    help="TSV mapping accession -> taxid; required for "
                         "--taxdump-dir to have any effect")
    ap.add_argument("--require-clades", action="store_true",
                    help="fail if a weighted eukaryotic clade has no data "
                         "instead of redistributing its budget")
    ap.add_argument("--plan-file", default="sampling_plan.json")
    ap.add_argument("--report-file", default="sampling_report.tsv")
    ap.add_argument("--workers", type=int, default=16,
                    help="maximum concurrent worker processes (default: 16)")
    ap.add_argument("--euk-workers", type=int, default=12,
                    help="ordered byte-range chunks for eukarya.fasta (default: 12)")
    ap.add_argument("--io-buffer-mb", type=int, default=32,
                    help="sequential read/write buffer per worker (default: 32 MiB)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.euk_workers < 1:
        raise SystemExit("--euk-workers must be >= 1")
    if args.io_buffer_mb < 1:
        raise SystemExit("--io-buffer-mb must be >= 1")
    if "/" in args.plan_file or "/" in args.report_file:
        raise SystemExit("--plan-file/--report-file must be bare file names")
    buffer_bytes = args.io_buffer_mb * (1 << 20)
    splits = parse_splits(args.balance_splits)

    src = Path(args.source_root).expanduser()
    out = Path(args.out_root).expanduser()
    if not src.is_dir():
        raise SystemExit(f"--source-root is not a directory: {src}")
    # Never let the output alias the input, directly or through a symlink.
    if out.exists() and out.resolve() == src.resolve():
        raise SystemExit(f"--out-root resolves to --source-root ({src.resolve()})")
    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(f"output is not empty: {out}; use --force")
    if args.force and out.exists():
        # rmtree does NOT follow symlinked directories: it unlinks them, so a
        # symlinked validation/test from a previous run cannot cause the source
        # corpus to be deleted.
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    priors = parse_map(args.class_priors, TIARA_PRIORS)
    if set(priors) != set(CLASSES):
        raise SystemExit(f"class priors must contain exactly: {', '.join(CLASSES)}")
    euk_weights = parse_map(args.euk_weights, DEFAULT_EUK_WEIGHTS)

    def log(message):
        print(message, file=sys.stderr, flush=True)

    acc2clade = build_acc2clade(args.taxdump_dir, args.accession_taxid_tsv,
                                log=log)
    backend = "taxid" if acc2clade else "header_sg"
    log(f"[run] balancing splits : {', '.join(splits)}")
    log(f"[run] other splits     : {args.eval_mode}")
    log(f"[run] clade resolution : {backend}")

    per_split: dict[str, dict] = {}
    clade_problems: list[str] = []

    for split in splits:
        files = source_files(src, split)
        task_count = len(make_tasks(files, args.euk_workers))
        log(f"[run:{split}] workers={min(args.workers, task_count)} "
            f"euk_chunks={args.euk_workers} tasks={task_count} "
            f"io_buffer={args.io_buffer_mb} MiB/worker")
        stats, resolution = census(
            files, min_len=args.min_len, min_purity=args.min_acgt_purity,
            workers=args.workers, euk_workers=args.euk_workers,
            buffer_bytes=buffer_bytes, acc2clade=acc2clade,
            label=f"census:{split}")

        coverage = clade_coverage(stats, euk_weights)
        if coverage["missing"]:
            log(f"[warn:{split}] eukaryotic clades with NO data: "
                f"{', '.join(coverage['missing'])} "
                f"({coverage['redistributed_weight']:.0%} of the eukaryotic "
                f"budget would be redistributed)")
            if args.require_clades:
                clade_problems.append(
                    f"{split}: {', '.join(coverage['missing'])}")

        target_total, top_available, top_target, plan = make_plan(
            stats, priors, euk_weights, args.target_total_bp)
        log(f"[plan:{split}] target total = {target_total:,} bp")
        for k in sorted(plan):
            p = plan[k]
            log(f"[plan:{split}] {k:<32} have={p['available_bp']:>15,} "
                f"target={p['target_bp']:>15,} rate={p['rate']:.8f}")

        actual = sample_split(
            files, out, plan, split=split, seed=args.seed,
            min_len=args.min_len, min_purity=args.min_acgt_purity,
            workers=args.workers, euk_workers=args.euk_workers,
            buffer_bytes=buffer_bytes, acc2clade=acc2clade)

        per_split[split] = {"stats": stats, "resolution": resolution,
                            "top_available": top_available,
                            "top_target": top_target, "plan": plan,
                            "actual": actual, "coverage": coverage}

    mirrored = []
    if args.eval_mode != "none":
        others = [s for s in ("train", "validation", "test") if s not in splits]
        mirrored = link_eval(src, out, args.eval_mode, others)
        if mirrored:
            log(f"[mirror] {args.eval_mode}: {', '.join(mirrored)} "
                f"(unbalanced by design)")

    # Reports are written BEFORE any --require-clades abort, so the evidence
    # needed to fix the corpus is always on disk.
    write_reports(out, per_split, args,
                  {"clade_resolution_backend": backend,
                   "mirrored_splits": mirrored})

    print(f"[done] sampled corpus -> {out}")
    print(f"[done] plan/report -> {out/args.plan_file} ; {out/args.report_file}")

    if clade_problems:
        raise SystemExit(
            "[FATAL] --require-clades: weighted eukaryotic clades with no "
            "data:\n  - " + "\n  - ".join(clade_problems) +
            "\nTheir budget would have been silently redistributed (this is "
            "what happened in v2.1.2). Either supply --taxdump-dir plus "
            "--accession-taxid-tsv so the clades can be resolved, or remove "
            "them from euk_weights on purpose. Reports were still written.")


if __name__ == "__main__":
    main()
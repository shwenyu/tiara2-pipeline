#!/usr/bin/env python3
"""Build deterministic continuous short crops without changing frozen splits."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


FILES = (
    "bacteria.fasta",
    "archaea.fasta",
    "eukarya.fasta",
    "mitochondria.fasta",
    "plastids.fasta",
)
LENGTH_BINS = (
    (800, 999),
    (1000, 1249),
    (1250, 1499),
    (1500, 1750),
    (1751, 1999),
    (2000, 2249),
    (2250, 2499),
)


def fasta(path: Path):
    header = None
    parts = []
    with path.open() as handle:
        for line in handle:
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(parts).upper()
                header = line[1:].strip()
                parts = []
            else:
                parts.append(line.strip())
    if header is not None:
        yield header, "".join(parts).upper()


def stable_u64(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def choose_crop(header: str, sequence: str, seed: int, focus_fraction: float):
    maximum = min(2499, len(sequence))
    if maximum < 800:
        return None
    token = stable_u64(f"{seed}|{header}")
    focus = maximum >= 1250 and (token % 1_000_000) < int(focus_fraction * 1_000_000)
    low = 1250 if focus else 800
    high = min(maximum, 1750 if focus else 2499)
    if high < low:
        low = 800
        high = maximum
    length = low + ((token >> 20) % (high - low + 1))
    max_start = len(sequence) - length
    start = 0 if max_start == 0 else (token >> 40) % (max_start + 1)
    return int(start), int(length), focus


def bin_name(length: int) -> str:
    for low, high in LENGTH_BINS:
        if low <= length <= high:
            return f"{low}-{high}"
    raise ValueError(length)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=24101)
    parser.add_argument("--focus-fraction", type=float, default=0.5)
    parser.add_argument("--smoke-limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if not 0 <= args.focus_fraction <= 1:
        raise SystemExit("--focus-fraction must be in [0,1]")

    source = Path(args.source).resolve()
    out = Path(args.out).resolve()
    if out.exists():
        manifest_path = out / "crop_manifest.json"
        if manifest_path.is_file() and not args.force:
            existing = json.loads(manifest_path.read_text())
            if existing.get("status") == "complete":
                print(f"[resume] crops already complete: {out}")
                return 0
        if not args.force:
            raise FileExistsError(f"incomplete output exists: {out}; use --force")
        backup = out.with_name(out.name + ".backup_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.replace(out, backup)

    tmp = out.with_name(out.name + f".tmp.{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    manifest = {
        "format": "tiara2-v2.4.1b-continuous-crops-v1",
        "status": "building",
        "source": str(source),
        "seed": args.seed,
        "length_bp": [800, 2499],
        "focus_length_bp": [1250, 1750],
        "focus_fraction": args.focus_fraction,
        "smoke_limit_per_file": args.smoke_limit,
        "splits": {},
    }
    try:
        for split in ("train", "validation"):
            split_out = tmp / split
            split_out.mkdir()
            split_stats = {"files": {}, "length_bins": Counter(), "records": 0, "bp": 0}
            for filename in FILES:
                src = source / split / filename
                dst = split_out / filename
                if not src.is_file():
                    raise FileNotFoundError(src)
                stats = Counter()
                with dst.open("w") as writer:
                    for index, (header, sequence) in enumerate(fasta(src)):
                        if args.smoke_limit and index >= args.smoke_limit:
                            break
                        chosen = choose_crop(
                            header,
                            sequence,
                            args.seed + (0 if split == "train" else 1),
                            args.focus_fraction,
                        )
                        if chosen is None:
                            stats["skipped_under_800"] += 1
                            continue
                        start, length, focus = chosen
                        crop = sequence[start : start + length]
                        writer.write(
                            f">{header} v241b_start={start} v241b_length={length} "
                            f"v241b_focus={int(focus)}\n{crop}\n"
                        )
                        stats["records"] += 1
                        stats["bp"] += length
                        stats["focus_records"] += int(focus)
                        split_stats["length_bins"][bin_name(length)] += 1
                split_stats["files"][filename] = dict(stats)
                split_stats["records"] += stats["records"]
                split_stats["bp"] += stats["bp"]
            split_stats["length_bins"] = dict(sorted(split_stats["length_bins"].items()))
            manifest["splits"][split] = split_stats

        files = {}
        for path in sorted(tmp.glob("*/*.fasta")):
            files[str(path.relative_to(tmp))] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        manifest["files"] = files
        manifest["status"] = "complete"
        manifest["generated_at"] = datetime.now(timezone.utc).isoformat()
        (tmp / "crop_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        os.replace(tmp, out)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    except Exception:
        (tmp / "crop_manifest.failed.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())

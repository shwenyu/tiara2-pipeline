#!/usr/bin/env python3
"""Create read-only sequence/label caches for exactly the v2.4.0 short rows."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

FILES = ("bacteria.fasta", "archaea.fasta", "eukarya.fasta", "mitochondria.fasta", "plastids.fasta")
HEADS = ("root", "euk", "prok", "organelle")
TABLE = np.full(256, 4, dtype=np.uint8)
for chars, value in ((b"Aa", 0), (b"Cc", 1), (b"Gg", 2), (b"TtUu", 3)):
    for char in chars:
        TABLE[char] = value


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fasta(path):
    seq = []
    with Path(path).open() as handle:
        for line in handle:
            if line.startswith(">"):
                if seq:
                    yield "".join(seq)
                seq = []
            else:
                seq.append(line.strip())
    if seq:
        yield "".join(seq)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-ready", required=True)
    p.add_argument("--short-features", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-bp", type=int, default=2499)
    args = p.parse_args()
    source = json.loads((Path(args.short_features) / "composite_features.json").read_text())
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    result = {"format": "tiara2-v250-rc-sequence-cache-v1", "source_manifest": str(Path(args.short_features).resolve()), "splits": {}}
    for split in ("train", "validation"):
        rows = int(source["splits"][split]["rows"])
        selected = np.load(source["splits"][split]["shards"][0]["indices"], mmap_mode="r")
        if len(selected) != rows:
            raise ValueError(f"{split} index length mismatch")
        split_out = out / split; split_out.mkdir(exist_ok=True)
        token_path = split_out / "tokens.u8"
        length_path = split_out / "lengths.i16"
        tokens = np.memmap(token_path, dtype=np.uint8, mode="w+", shape=(rows, args.max_bp))
        tokens[:] = 4
        lengths = np.memmap(length_path, dtype=np.int16, mode="w+", shape=(rows,))
        cursor = source_row = output_row = 0
        for filename in FILES:
            for seq in fasta(Path(args.train_ready) / split / filename):
                if cursor < rows and source_row == int(selected[cursor]):
                    raw = np.frombuffer(seq.encode("ascii", "replace"), dtype=np.uint8)
                    n = min(len(raw), args.max_bp)
                    tokens[output_row, :n] = TABLE[raw[:n]]
                    lengths[output_row] = n
                    output_row += 1; cursor += 1
                    if output_row % 50000 == 0:
                        print(f"[{split}] {output_row:,}/{rows:,}", flush=True)
                source_row += 1
        if output_row != rows:
            raise ValueError(f"{split} wrote {output_row}, expected {rows}")
        tokens.flush(); lengths.flush()
        shard = source["splits"][split]["shards"][0]
        base_rows = int(shard["base_rows"])
        labels = {}
        for head in HEADS:
            full = np.memmap(shard["files"][head], dtype=np.int64, mode="r", shape=(base_rows,))
            path = split_out / f"{head}.i64"
            chosen = np.memmap(path, dtype=np.int64, mode="w+", shape=(rows,))
            chosen[:] = full[selected]
            chosen.flush(); labels[head] = str(path)
        result["splits"][split] = {"rows": rows, "max_bp": args.max_bp, "tokens": str(token_path), "lengths": str(length_path), "labels": labels, "source_indices": str(Path(shard["indices"]).resolve()), "source_indices_sha256": sha256(shard["indices"])}
    result["schema"] = source["schema"]
    result["immutable_sources"] = {"train_ready": str(Path(args.train_ready).resolve()), "short_features": str(Path(args.short_features).resolve())}
    (out / "sequence_cache.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

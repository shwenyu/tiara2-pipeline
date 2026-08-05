#!/usr/bin/env python3
"""Shard source FASTAs into (partition, length_bin, split) shards + metadata.

Run this ONCE (ideally folded into the chop stage so no extra full pass is
needed). It reads each source split/class FASTA a single time and writes:

  <out>/shards/<partition>/<bin>/<split>.fasta
  <out>/metadata/<partition>__<split>.tsv   (fragment_id, split, partition,
                                              length, length_bin, source)

Because double-95 duplicates require length ratio >= min_cov, only same-bin
(plus geometric neighbours) fragments can collide, so downstream searches run
per bin instead of against the whole corpus.
"""
import argparse
import json
from pathlib import Path

from common import LengthBinner, iter_fasta
from progress import (
    add_progress_args, file_size, progress_from_args, record_bytes,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    p.add_argument("--classes", nargs="+", required=True)
    p.add_argument("--partition-by", choices=["class", "global"], default="class")
    p.add_argument("--mode", choices=["discrete", "geometric"], default="discrete")
    p.add_argument("--min-cov", type=float, default=0.95)
    p.add_argument("--discrete-lengths", type=int, nargs="+",
                   default=[1000, 2000, 3000, 5000, 10000])
    p.add_argument("--tolerance", type=int, default=0)
    add_progress_args(p)
    return p.parse_args()


def main():
    a = parse_args()
    # Progress goes to stderr so the shard manifest / any piped stdout stays
    # clean; the workload is stated up front from st_size.
    prog = progress_from_args(a, label="bin")
    units = []
    for split in a.splits:
        for cls in a.classes:
            src = Path(a.source_root) / split / f"{cls}.fasta"
            if src.is_file():
                units.append((f"{split}/{cls}.fasta", file_size(src)))
    prog.plan(units)
    binner = LengthBinner(mode=a.mode, min_cov=a.min_cov,
                          discrete_lengths=tuple(a.discrete_lengths),
                          tolerance=a.tolerance)
    out = Path(a.out)
    (out / "shards").mkdir(parents=True, exist_ok=True)
    (out / "metadata").mkdir(parents=True, exist_ok=True)

    handles = {}
    meta_handles = {}
    counts = {}

    def shard_handle(partition, binkey, split):
        key = (partition, binkey, split)
        h = handles.get(key)
        if h is None:
            d = out / "shards" / partition / binkey
            d.mkdir(parents=True, exist_ok=True)
            h = open(d / f"{split}.fasta", "w")
            handles[key] = h
        return h

    def meta_handle(partition, split):
        key = (partition, split)
        h = meta_handles.get(key)
        if h is None:
            h = open(out / "metadata" / f"{partition}__{split}.tsv", "w")
            h.write("fragment_id\tsplit\tpartition\tlength\tlength_bin\tsource\n")
            meta_handles[key] = h
        return h

    try:
        for split in a.splits:
            for cls in a.classes:
                src = Path(a.source_root) / split / f"{cls}.fasta"
                partition = cls if a.partition_by == "class" else "all"
                prog.start_unit(f"{split}/{cls}.fasta", file_size(src))
                shards_before = len(handles)
                for frag_id, header, seq in iter_fasta(src):
                    prog.tick(nbytes=record_bytes(header, seq))
                    length = sum(len(s) for s in seq)
                    binkey = binner.bin_key(length)
                    sh = shard_handle(partition, binkey, split)
                    sh.write(">" + header + "\n")
                    for s in seq:
                        sh.write(s + "\n")
                    meta_handle(partition, split).write(
                        f"{frag_id}\t{split}\t{partition}\t{length}\t{binkey}\t{src}\n")
                    counts[(partition, binkey, split)] = counts.get(
                        (partition, binkey, split), 0) + 1
                # Per-file verdict: how many length bins this file opened. A
                # preset that silently dumps everything into VAR is visible
                # here, not three hours later in bin_manifest.json.
                prog.finish_unit(f"+{len(handles) - shards_before} shard(s)")
    finally:
        for h in handles.values():
            h.close()
        for h in meta_handles.values():
            h.close()

    manifest = {
        "partition_by": a.partition_by,
        "mode": a.mode,
        "min_cov": a.min_cov,
        "discrete_lengths": a.discrete_lengths,
        "shard_counts": {"/".join(k): v for k, v in sorted(counts.items())},
    }
    (out / "bin_manifest.json").write_text(json.dumps(manifest, indent=2))
    prog.finish(f"{len(counts)} shard(s) -> {out / 'bin_manifest.json'}")


if __name__ == "__main__":
    main()

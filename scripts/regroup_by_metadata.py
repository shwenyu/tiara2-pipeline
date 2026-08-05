#!/usr/bin/env python3
"""Rebuild per-species / per-split FASTAs from surviving fragment IDs.

Dedup produces JSON files of removed IDs per (partition, bin, split). This stage
combines them into a global removed-ID set, then streams the original source
FASTAs once, dropping removed fragments and routing survivors to output layout
keyed by metadata (species/class + split). Sequences are written exactly once.
"""
import argparse
import json
from pathlib import Path

from common import iter_fasta
from progress import (
    add_progress_args, file_size, progress_from_args, record_bytes,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", required=True)
    p.add_argument("--removed-json", nargs="+", default=[])
    p.add_argument("--out", required=True)
    p.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    p.add_argument("--classes", nargs="+", required=True)
    p.add_argument("--summary", required=True)
    add_progress_args(p)
    return p.parse_args()


def main():
    a = parse_args()
    # Progress on stderr; the summary JSON is the result.
    prog = progress_from_args(a, label="regroup")
    # removed IDs are namespaced by split to avoid cross-split id clashes
    removed = {s: set() for s in a.splits}
    for jf in a.removed_json:
        obj = json.loads(Path(jf).read_text())
        split = obj["query_split"]
        removed.setdefault(split, set()).update(obj.get("removed_ids", []))
    prog.write(f"removed-id sets: "
               + ", ".join(f"{s}={len(ids):,}" for s, ids in removed.items()))

    units = []
    for split in a.splits:
        for cls in a.classes:
            src = Path(a.source_root) / split / f"{cls}.fasta"
            if src.is_file():
                units.append((f"{split}/{cls}.fasta", file_size(src)))
    prog.plan(units)

    out = Path(a.out)
    stats = {}
    for split in a.splits:
        for cls in a.classes:
            src = Path(a.source_root) / split / f"{cls}.fasta"
            dst_dir = out / split
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / f"{cls}.fasta"
            prog.start_unit(f"{split}/{cls}.fasta", file_size(src))
            n_in = n_out = 0
            with open(dst.with_suffix(".fasta.tmp"), "w") as w:
                for frag_id, header, seq in iter_fasta(src):
                    prog.tick(nbytes=record_bytes(header, seq))
                    n_in += 1
                    if frag_id in removed[split]:
                        continue
                    n_out += 1
                    w.write(">" + header + "\n")
                    for s in seq:
                        w.write(s + "\n")
            dst.with_suffix(".fasta.tmp").replace(dst)
            stats[f"{split}/{cls}"] = {
                "input": n_in, "kept": n_out, "removed": n_in - n_out,
                "retention": (n_out / n_in if n_in else 1.0),
            }
            # Retention per file, immediately: a dedup run that eats a whole
            # class shows up on the first file instead of at the end.
            prog.finish_unit(f"kept {n_out:,}/{n_in:,} "
                             f"({(n_out / n_in if n_in else 1.0):.2%})")
    Path(a.summary).write_text(json.dumps(stats, indent=2))
    prog.finish(f"summary -> {a.summary}")


if __name__ == "__main__":
    main()

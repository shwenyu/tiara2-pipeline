#!/usr/bin/env python3
"""Re-route an existing train_ready corpus into the correct class files using
the per-record header metadata, WITHOUT re-chopping any genome.

The chopper already stamped every fragment with its true provenance::

    >GCA_000411095.1|25 sg=Archaeplastida label=euk epoch=train

Only the *file* a record landed in was wrong.  This script streams every class
FASTA, recomputes the class with :func:`tiara2.labels.class_from_header`, and
writes a corrected tree.  Sequences are copied byte-for-byte; nothing is
re-sampled, so the corpus stays reproducible.

Usage::

    python3 scripts/repair_corpus_labels.py --root <train_ready> --out <train_ready_relabelled>
    python3 scripts/repair_corpus_labels.py --root <train_ready> --out <...> --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from progress import (  # noqa: E402
    NullProgress, add_progress_args, file_size, progress_from_args, record_bytes,
)
from tiara2.labels import FINE_CLASSES as CLASSES, OTHER_MEMBERS, class_from_header  # noqa: E402

DEFAULT_SPLITS = ("train", "validation", "test")
BUFFER_SIZE = 4 * 1024 * 1024
ORGANELLE_CLASSES = ("mitochondria", "plastids")
# stage-1 groups the organelles and viruses together; materialise that union so
# training can read one file instead of concatenating three.
STAGE1_OTHER = OTHER_MEMBERS


def iter_records(path: Path):
    """Yield ``(header, sequence_lines)`` for a FASTA file."""
    header = None
    chunk: list[str] = []
    with path.open() as handle:
        for line in handle:
            if line.startswith(">"):
                if header is not None:
                    yield header, chunk
                header = line.rstrip("\n")
                chunk = []
            elif header is not None:
                chunk.append(line.rstrip("\n"))
    if header is not None:
        yield header, chunk


def repair_split(split_dir: Path, out_dir: Path, dry_run: bool = False,
                 prog=None) -> dict:
    """Repair one split directory.  Returns a movement matrix.

    ``prog`` is a :class:`progress.Progress`; each source FASTA is announced
    before it is opened and its own relocation count is reported the moment it
    closes, so a wrong routing rule surfaces in minutes instead of after the
    whole corpus has been rewritten.
    """
    prog = prog or NullProgress("relabel")
    moved: Counter = Counter()      # (from_class, to_class) -> n
    unknown = 0
    out_dir.mkdir(parents=True, exist_ok=True)

    handles = {}
    organelle = None
    other = None
    unclassified = None
    if not dry_run:
        for class_name in CLASSES:
            handles[class_name] = (out_dir / f"{class_name}.fasta").open("w", buffering=BUFFER_SIZE)
        organelle = (out_dir / "organelle.fasta").open("w", buffering=BUFFER_SIZE)
        other = (out_dir / "other.fasta").open("w", buffering=BUFFER_SIZE)
        unclassified = (out_dir / "unclassified.fasta").open("w", buffering=BUFFER_SIZE)
    try:
        for file_class in CLASSES:
            src = split_dir / f"{file_class}.fasta"
            if not src.is_file():
                continue
            prog.start_unit(f"{split_dir.name}/{src.name}", file_size(src))
            unit_relocated = 0
            for header, chunk in iter_records(src):
                prog.tick(nbytes=record_bytes(header, chunk))
                target = class_from_header(header)
                if target is None:
                    unknown += 1
                    moved[(file_class, "unknown")] += 1
                    if unclassified is not None:
                        unclassified.write(header + "\n" + "\n".join(chunk) + "\n")
                    continue
                moved[(file_class, target)] += 1
                if target != file_class:
                    unit_relocated += 1
                if handles:
                    record = header + "\n" + "\n".join(chunk) + "\n"
                    handles[target].write(record)
                    if target in ORGANELLE_CLASSES:
                        organelle.write(record)
                    if target in STAGE1_OTHER:
                        other.write(record)
            prog.finish_unit(f"relocated {unit_relocated:,}")
    finally:
        for handle in handles.values():
            handle.close()
        if organelle is not None:
            organelle.close()
        if other is not None:
            other.close()
        if unclassified is not None:
            unclassified.close()

    if not dry_run:
        legacy = out_dir / "archea.fasta"
        if legacy.exists() or legacy.is_symlink():
            legacy.unlink()
        legacy.symlink_to("archaea.fasta")
        if unknown == 0:
            (out_dir / "unclassified.fasta").unlink()

    before = Counter()
    after = Counter()
    for (src_class, dst_class), n in moved.items():
        before[src_class] += n
        after[dst_class] += n
    return {
        "split": split_dir.name,
        "before": dict(before),
        "after": dict(after),
        "moves": {f"{a} -> {b}": n for (a, b), n in sorted(moved.items()) if a != b},
        "unknown": unknown,
        "records": sum(moved.values()),
    }


def plan_units(root: Path, splits=DEFAULT_SPLITS):
    """(name, bytes) for every source FASTA this run will read.

    Sizes come from ``st_size``, which is known before the first read, so the
    total workload can be stated up front.
    """
    units = []
    for split in splits:
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        for class_name in CLASSES:
            src = split_dir / f"{class_name}.fasta"
            if src.is_file():
                units.append((f"{split}/{src.name}", file_size(src)))
    return units


def repair_corpus(root: Path, out: Path, splits=DEFAULT_SPLITS,
                  dry_run: bool = False, copy_sidecars: bool = True,
                  prog=None) -> dict:
    prog = prog or NullProgress("relabel")
    prog.plan(plan_units(root, splits))
    results = []
    for split in splits:
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        results.append(repair_split(split_dir, out / split, dry_run, prog))

    if not dry_run and copy_sidecars:
        for name in ("fragments.tsv", "chop_manifest.json", "prepare_manifest.json"):
            src = root / name
            if src.is_file():
                shutil.copy2(src, out / name)

    before_total = {}
    after_total = {}
    for result in results:
        for class_name, n in result["before"].items():
            before_total[class_name] = before_total.get(class_name, 0) + n
        for class_name, n in result["after"].items():
            after_total[class_name] = after_total.get(class_name, 0) + n
    emptied = [c for c in CLASSES if before_total.get(c, 0) > 0 and after_total.get(c, 0) == 0]

    report = {
        "root": str(root),
        "out": str(out),
        "dry_run": dry_run,
        "splits": results,
        "records": sum(r["records"] for r in results),
        "relocated": sum(sum(r["moves"].values()) for r in results),
        "unknown": sum(r["unknown"] for r in results),
        "before": before_total,
        "after": after_total,
        "emptied_classes": emptied,
    }
    if not dry_run:
        out.mkdir(parents=True, exist_ok=True)
        (out / "relabel_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    prog.finish(f"relocated {report['relocated']:,}")
    return report


def render(report: dict) -> str:
    lines = [("DRY-RUN " if report["dry_run"] else "") + f"relabel: {report['root']} -> {report['out']}", ""]
    for split in report["splits"]:
        lines.append(f"  [{split['split']}] {split['records']:,} records")
        lines.append(f"      {'class':16} {'before':>16} {'after':>16} {'delta':>16}")
        for class_name in CLASSES:
            before = split["before"].get(class_name, 0)
            after = split["after"].get(class_name, 0)
            lines.append(f"      {class_name:16} {before:>16,} {after:>16,} {after - before:>+16,}")
        if split["moves"]:
            lines.append("      relocations:")
            for move, n in sorted(split["moves"].items(), key=lambda kv: -kv[1]):
                lines.append(f"          {move:36} {n:>16,}")
        if split["unknown"]:
            lines.append(f"      !! {split['unknown']:,} records had unmappable metadata -> unclassified.fasta")
        lines.append("")
    lines.append(f"  total records   {report['records']:,}")
    lines.append(f"  relocated       {report['relocated']:,}")
    if report["unknown"]:
        lines.append(f"  UNKNOWN         {report['unknown']:,}  <== inspect before training")
    if report["emptied_classes"]:
        lines.append("")
        lines.append("  !! REFUSING: these classes would be emptied by the relabel:")
        for class_name in report["emptied_classes"]:
            lines.append(f"        {class_name:16} {report['before'].get(class_name, 0):>16,} -> 0")
        lines.append("")
        lines.append("  A class going to zero means the routing rules disagree with the corpus,")
        lines.append("  not that the corpus lacks the class.  Inspect the headers first:")
        lines.append("      grep -m5 '^>' <root>/train/<class>.fasta")
        lines.append("  Override only if you genuinely intend to drop the class: --allow-class-drop")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    ap.add_argument("--allow-class-drop", action="store_true",
                    help="proceed even if a class would end up with zero records")
    ap.add_argument("--dry-run", action="store_true",
                    help="count relocations without writing any FASTA")
    add_progress_args(ap)
    args = ap.parse_args(argv)

    root = Path(args.root)
    out = Path(args.out)
    if not args.dry_run and out.resolve() == root.resolve():
        raise SystemExit("refusing to write in place; choose a different --out")
    prog = progress_from_args(args, label="relabel")
    report = repair_corpus(root, out, tuple(args.splits), args.dry_run, prog=prog)
    # Progress went to stderr; the report is the RESULT and goes to stdout so
    # `> report.txt` and `| grep` stay clean.
    print(render(report))
    if report["emptied_classes"] and not args.allow_class_drop:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

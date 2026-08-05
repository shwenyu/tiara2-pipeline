#!/usr/bin/env python3
"""Audit a train_ready corpus for records whose FASTA file disagrees with the
class implied by their own header metadata.

Every record written by the chopper carries its provenance::

    >GCA_000411095.1|25 sg=Archaeplastida label=euk epoch=train

so the corpus can be checked -- and repaired -- without re-chopping anything.

Usage::

    python3 scripts/audit_corpus_labels.py --root /path/train_ready_v2_0_hybrid
    python3 scripts/audit_corpus_labels.py --root ... --limit 200000 --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from progress import (  # noqa: E402
    NullProgress, add_progress_args, file_size, progress_from_args,
)
from tiara2.labels import FINE_CLASSES as CLASSES, class_from_header, parse_header  # noqa: E402

DEFAULT_SPLITS = ("train", "validation", "test")


def iter_headers(path: Path, limit: int = 0):
    """Yield header lines from a FASTA file, stopping after ``limit`` (0 = all)."""
    seen = 0
    with path.open() as handle:
        for line in handle:
            if not line.startswith(">"):
                continue
            yield line.rstrip("\n")
            seen += 1
            if limit and seen >= limit:
                return


def audit_file(path: Path, file_class: str, limit: int = 0, prog=None) -> dict:
    """Tally header-derived classes for one class FASTA file."""
    prog = prog or NullProgress("audit")
    derived = Counter()
    supergroups = Counter()
    mismatch_sg = Counter()
    total = 0
    for header in iter_headers(path, limit):
        total += 1
        # Only headers are read, but the file is still traversed in full, so
        # progress is accounted in BYTES for a percentage that exists on the
        # first pass. Sequence lines are charged to the header that owns them.
        prog.tick(nbytes=len(header) + 1)
        meta = parse_header(header)
        cls = class_from_header(header)
        derived[cls or "unknown"] += 1
        supergroups[meta["sg"] or "(none)"] += 1
        if cls != file_class:
            mismatch_sg[(meta["sg"] or "(none)", meta["label"] or "(none)")] += 1
    matched = derived.get(file_class, 0)
    return {
        "path": str(path),
        "file_class": file_class,
        "records": total,
        "matched": matched,
        "mismatched": total - matched,
        "derived": dict(derived),
        "supergroups": dict(supergroups),
        "mismatch_detail": {f"sg={k[0]} label={k[1]}": v for k, v in mismatch_sg.most_common()},
    }


def plan_units(root: Path, splits=DEFAULT_SPLITS):
    """(name, bytes) for every class FASTA this audit will traverse."""
    units = []
    for split in splits:
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        for class_name in CLASSES:
            path = split_dir / f"{class_name}.fasta"
            if path.is_file():
                units.append((f"{split}/{path.name}", file_size(path)))
    return units


def audit_corpus(root: Path, splits=DEFAULT_SPLITS, limit: int = 0,
                 prog=None) -> dict:
    prog = prog or NullProgress("audit")
    prog.plan(plan_units(root, splits))
    files = []
    for split in splits:
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        for class_name in CLASSES:
            path = split_dir / f"{class_name}.fasta"
            if path.is_file():
                prog.start_unit(f"{split}/{path.name}", file_size(path))
                entry = audit_file(path, class_name, limit, prog)
                rate = ((entry["mismatched"] / entry["records"])
                        if entry["records"] else 0.0)
                # Per-file verdict right here: a mislabelled class is visible
                # immediately, not only in the summary table at the end.
                prog.finish_unit(f"mismatched {entry['mismatched']:,} "
                                 f"({rate:.2%})")
                files.append(entry)
    total = sum(f["records"] for f in files)
    bad = sum(f["mismatched"] for f in files)
    prog.finish(f"mismatched {bad:,} of {total:,}")
    return {
        "root": str(root),
        "limit_per_file": limit,
        "files": files,
        "records": total,
        "mismatched": bad,
        "mismatch_rate": (bad / total) if total else 0.0,
        "clean": bad == 0,
    }


def render(report: dict) -> str:
    lines = [f"label audit: {report['root']}"]
    if report["limit_per_file"]:
        lines.append(f"  (sampled: first {report['limit_per_file']:,} records per file)")
    lines.append("")
    lines.append(f"  {'file':44} {'records':>14} {'mismatched':>14} {'rate':>8}")
    lines.append("  " + "-" * 84)
    for entry in report["files"]:
        short = "/".join(Path(entry["path"]).parts[-2:])
        rate = (entry["mismatched"] / entry["records"]) if entry["records"] else 0.0
        flag = "  <== " if entry["mismatched"] else ""
        lines.append(f"  {short:44} {entry['records']:>14,} {entry['mismatched']:>14,} {rate:>7.2%}{flag}")
    lines.append("  " + "-" * 84)
    lines.append(f"  {'TOTAL':44} {report['records']:>14,} {report['mismatched']:>14,} {report['mismatch_rate']:>7.2%}")
    for entry in report["files"]:
        if not entry["mismatched"]:
            continue
        short = "/".join(Path(entry["path"]).parts[-2:])
        lines.append("")
        lines.append(f"  {short} -- records actually belong to:")
        for cls, n in sorted(entry["derived"].items(), key=lambda kv: -kv[1]):
            mark = "ok " if cls == entry["file_class"] else "BAD"
            lines.append(f"      {mark} {cls:16} {n:>14,}")
        for detail, n in list(entry["mismatch_detail"].items())[:8]:
            lines.append(f"          {detail:50} {n:>14,}")
    lines.append("")
    if report["clean"]:
        lines.append("  RESULT: clean -- every record sits in the right file.")
    else:
        lines.append("  RESULT: MISLABELLED records present.")
        lines.append("  Repair without re-chopping:")
        lines.append("      python3 scripts/repair_corpus_labels.py --root <root> --out <root>_relabelled")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="train_ready corpus root")
    ap.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    ap.add_argument("--limit", type=int, default=0,
                    help="only read the first N records per file (0 = all)")
    ap.add_argument("--json", default="", help="also write the report as JSON")
    add_progress_args(ap)
    args = ap.parse_args(argv)

    prog = progress_from_args(args, label="audit")
    report = audit_corpus(Path(args.root), tuple(args.splits), args.limit,
                          prog=prog)
    # Progress on stderr, report on stdout.
    print(render(report))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["clean"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

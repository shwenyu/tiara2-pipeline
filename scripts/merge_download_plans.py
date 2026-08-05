#!/usr/bin/env python3
"""Join the three genome plans into ONE download manifest.

WHY THIS EXISTS
---------------
Three independent tables decide which genomes v2.2 should train on, and nothing
in the pipeline joins them:

1. ``download_all.tsv``      -- acquire/plan-downloads. Everything NCBI offers
                                that passed QC + one-isolate-per-species. It is
                                produced by ncbi_pipeline.py (outside this
                                package), so its exact columns are treated as
                                unknown and preserved verbatim.
2. ``curation_selection.tsv`` -- the curate stage. One genome per genus plus a
                                per-genome ``target_bp`` cap. This is a PLAN:
                                curate moves no data and, critically, acquire
                                never reads it back (grep the package: only
                                curate.py mentions the file).
3. ``tiara1_selection.tsv``  -- import_tiara_corpus.py. Tiara1's 8,198 rows,
                                each tagged ``new`` / ``already_held`` / ... and
                                ``is_anchor`` for the 69 euk nuclear genomes.

Without the join, running acquire end-to-end downloads the full NCBI selection
and silently ignores the genus cap; running curate alone downloads nothing.

WHAT IT DOES
------------
output = (download_all INTERSECT curate) UNION (Tiara1 rows we must not lose)

* Rows keep their original ``download_all`` columns and order, so whatever
  consumes the manifest keeps working. Five columns are appended:
  ``plan_source``, ``target_bp``, ``target_fragments``, ``curate_anchor``,
  ``tiara1_status``.
* ``target_bp`` is carried over from curate so the per-genome cap survives into
  the download step -- that cap is the whole point of the v2.2 round.
* Accession matching is version-tolerant: ``GCA_000008085.1`` matches
  ``GCA_000008085.2`` on the base accession, because RefSeq/GenBank version
  drift is the normal case and an exact-only match would silently drop anchors.
  Exact hits always win over base hits.
* Anchors are non-negotiable. If an anchor accession is in no output row the
  script exits 2, because a missing anchor means the euk set is NOT a superset
  of Tiara1's and the comparison against Tiara1 stops being meaningful.
* Anything selected by curate (or required by Tiara1) that ``download_all``
  does not contain is written to ``missing_from_downloads.tsv`` with its
  ``ftp_path`` where known, instead of being dropped in silence.

Usage
-----
    python3 scripts/merge_download_plans.py \\
        --downloads /data/shouhanyu/Tiara2/metadata/download_all.tsv \\
        --curation  /ssd/shouhanyu/Tiara2/.work_pipeline_v2_2_breadth/curate/curation_selection.tsv \\
        --tiara1    /data/shouhanyu/Tiara2/import/tiara1/tiara1_selection.tsv \\
        --anchors   config/tiara1_anchor_accessions.txt \\
        --out       /data/shouhanyu/Tiara2/metadata/download_curated.tsv

Exit codes: 0 ok, 2 anchors missing / empty result, 1 bad input.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from progress import add_progress_args, counter_from_args  # noqa: E402

# Candidate accession column names, best first. download_all.tsv comes from
# ncbi_pipeline.py, whose header we do not control.
ACCESSION_KEYS = ("assembly_accession", "accession", "asm_accession",
                  "assembly", "gca", "genome_accession")
FTP_KEYS = ("ftp_path", "ftp", "url", "download_url")
ORGANISM_KEYS = ("organism_name", "organism", "species", "name")
APPENDED = ("plan_source", "target_bp", "target_fragments", "curate_anchor",
            "tiara1_status")
MISSING_COLUMNS = ("accession", "required_by", "reason", "target_bp",
                   "organism_name", "ftp_path")


def base_accession(acc: str) -> str:
    """``GCA_000008085.1`` -> ``GCA_000008085`` (version-tolerant matching)."""
    return str(acc or "").strip().split(".")[0]


def pick_column(fieldnames, candidates):
    """First candidate present, compared case-insensitively and '#'-stripped."""
    if not fieldnames:
        return ""
    norm = {str(name or "").lstrip("#").strip().lower(): name
            for name in fieldnames}
    for cand in candidates:
        if cand in norm:
            return norm[cand]
    return ""


def read_tsv(path, prog=None, label=""):
    """Read a TSV/CSV into (rows, fieldnames).

    Tolerates the assembly_summary habit of putting the header on a ``#`` line.
    """
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"not a file: {path}")
    delim = "," if path.suffix.lower() == ".csv" else "\t"
    with open(path, newline="", encoding="utf-8", errors="replace") as handle:
        lines = [ln for ln in handle.read().splitlines() if ln.strip()]
    if not lines:
        return [], []
    header_idx = 0
    for i, line in enumerate(lines[:5]):
        probe = line.lstrip("#").lower()
        if any(key in probe for key in ("accession", "assembly")):
            header_idx = i
            break
    fieldnames = [h.lstrip("#").strip() for h in lines[header_idx].split(delim)]
    rows = []
    for line in lines[header_idx + 1:]:
        if line.startswith("#"):
            continue
        rows.append(dict(zip(fieldnames, line.split(delim))))
        if prog is not None:
            prog.count()
    if prog is not None:
        prog.write(f"{label or path.name}: {len(rows):,} row(s), "
                   f"{len(fieldnames)} column(s)")
    return rows, fieldnames


def read_anchors(path):
    """One accession per line; ``#`` comments and blanks ignored."""
    if not path:
        return []
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"anchor file not found: {p}")
    out = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line.split()[0])
    return out


def index_by_accession(rows, key):
    """Two lookups: exact accession, and base accession -> first row.

    Exact is authoritative; base is the fallback that survives version drift.
    """
    exact, base = {}, {}
    for row in rows:
        acc = str(row.get(key, "") or "").strip()
        if not acc:
            continue
        exact.setdefault(acc, row)
        base.setdefault(base_accession(acc), row)
    return exact, base


def tiara1_required(rows, statuses):
    """Tiara1 rows we refuse to lose: anchors always, plus selected statuses.

    Anchors are kept regardless of status. An anchor already in our index does
    not need downloading, but it must still be represented in the plan or the
    'our euk set is a superset of Tiara1's' claim goes unchecked.
    """
    wanted = {s.strip().lower() for s in statuses if s.strip()}
    out = []
    for row in rows:
        status = str(row.get("status", "") or "").strip().lower()
        anchor = str(row.get("is_anchor", "") or "").strip() == "1"
        if anchor or (status in wanted):
            out.append(row)
    return out


def merge(downloads, dl_fields, curation, tiara1, anchors, prog=None):
    """Do the join. Returns (out_rows, missing_rows, report)."""
    dl_key = pick_column(dl_fields, ACCESSION_KEYS)
    if not dl_key:
        raise SystemExit(
            "cannot find an accession column in the download manifest; "
            f"columns were: {', '.join(map(str, dl_fields))}")
    dl_exact, dl_base = index_by_accession(downloads, dl_key)

    cur_key = pick_column(list(curation[0].keys()) if curation else [],
                          ACCESSION_KEYS) or "assembly_accession"
    t1_key = pick_column(list(tiara1[0].keys()) if tiara1 else [],
                         ACCESSION_KEYS) or "accession"

    out_rows: list[dict] = []
    missing: list[dict] = []
    claimed: set[int] = set()          # id() of download rows already emitted
    kept_base: set[str] = set()
    counts = {"curate_exact": 0, "curate_base": 0, "curate_missing": 0,
              "tiara1_added": 0, "tiara1_missing": 0}

    def emit(dl_row, source, cur_row=None, t1_row=None):
        row = dict(dl_row)
        row["plan_source"] = source
        row["target_bp"] = (cur_row or {}).get("target_bp", "")
        row["target_fragments"] = (cur_row or {}).get("target_fragments", "")
        row["curate_anchor"] = (cur_row or {}).get("anchor", "")
        row["tiara1_status"] = (t1_row or {}).get("status", "")
        out_rows.append(row)
        claimed.add(id(dl_row))
        kept_base.add(base_accession(row.get(dl_key, "")))

    # --- pass 1: curate decides the shape of the corpus --------------------
    for cur_row in curation:
        acc = str(cur_row.get(cur_key, "") or "").strip()
        if not acc:
            continue
        if prog is not None:
            prog.count()
        hit = dl_exact.get(acc)
        if hit is not None:
            counts["curate_exact"] += 1
            emit(hit, "curate", cur_row=cur_row)
            continue
        hit = dl_base.get(base_accession(acc))
        if hit is not None:
            counts["curate_base"] += 1
            emit(hit, "curate_other_version", cur_row=cur_row)
            continue
        counts["curate_missing"] += 1
        missing.append({
            "accession": acc,
            "required_by": "curate",
            "reason": "NOT_IN_DOWNLOAD_MANIFEST",
            "target_bp": cur_row.get("target_bp", ""),
            "organism_name": cur_row.get(
                pick_column(list(cur_row.keys()), ORGANISM_KEYS), ""),
            "ftp_path": cur_row.get(
                pick_column(list(cur_row.keys()), FTP_KEYS), ""),
        })

    # --- pass 2: Tiara1 rows curate did not already bring in ---------------
    for t1_row in tiara1:
        acc = str(t1_row.get(t1_key, "") or "").strip()
        if not acc:
            continue
        if prog is not None:
            prog.count()
        if base_accession(acc) in kept_base:
            continue
        hit = dl_exact.get(acc) or dl_base.get(base_accession(acc))
        if hit is not None and id(hit) not in claimed:
            counts["tiara1_added"] += 1
            emit(hit, "tiara1", t1_row=t1_row)
            continue
        if hit is None:
            counts["tiara1_missing"] += 1
            missing.append({
                "accession": acc,
                "required_by": "tiara1_anchor" if str(
                    t1_row.get("is_anchor", "")).strip() == "1" else "tiara1",
                "reason": "NOT_IN_DOWNLOAD_MANIFEST",
                "target_bp": "",
                "organism_name": t1_row.get("organism_name", ""),
                "ftp_path": "",
            })

    # --- anchors: the one hard gate ----------------------------------------
    anchors_missing = sorted({a for a in anchors
                              if base_accession(a) not in kept_base})

    report = {
        "download_manifest_rows": len(downloads),
        "download_accession_column": dl_key,
        "curation_rows": len(curation),
        "tiara1_required_rows": len(tiara1),
        "output_rows": len(out_rows),
        "dropped_from_manifest": len(downloads) - len(
            {id(r) for r in downloads if id(r) in claimed}),
        "missing_rows": len(missing),
        "anchors_total": len(anchors),
        "anchors_missing": anchors_missing,
        "counts": counts,
    }
    return out_rows, missing, report


def write_tsv(path, rows, columns):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns),
                                delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def render(report) -> str:
    c = report["counts"]
    lines = ["merge download plans", ""]
    lines.append(f"  download manifest rows  {report['download_manifest_rows']:>9,}"
                 f"   (accession column: {report['download_accession_column']})")
    lines.append(f"  curate plan rows        {report['curation_rows']:>9,}")
    lines.append(f"  tiara1 required rows    {report['tiara1_required_rows']:>9,}")
    lines.append("")
    lines.append(f"  matched by curate       {c['curate_exact']:>9,}")
    lines.append(f"    other version         {c['curate_base']:>9,}")
    lines.append(f"    not in manifest       {c['curate_missing']:>9,}")
    lines.append(f"  added for tiara1        {c['tiara1_added']:>9,}")
    lines.append(f"    not in manifest       {c['tiara1_missing']:>9,}")
    lines.append("")
    lines.append(f"  OUTPUT rows             {report['output_rows']:>9,}")
    lines.append(f"  dropped from manifest   {report['dropped_from_manifest']:>9,}")
    lines.append(f"  anchors                 {report['anchors_total']:>9,}"
                 f"   missing: {len(report['anchors_missing'])}")
    if report["anchors_missing"]:
        shown = ", ".join(report["anchors_missing"][:10])
        lines.append(f"    MISSING ANCHORS: {shown}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--downloads", required=True,
                    help="download_all.tsv from acquire/plan-downloads")
    ap.add_argument("--curation", required=True,
                    help="curation_selection.tsv from the curate stage")
    ap.add_argument("--tiara1", default="",
                    help="tiara1_selection.tsv from import_tiara_corpus.py")
    ap.add_argument("--anchors", default="",
                    help="anchor accession list (config/tiara1_anchor_accessions.txt)")
    ap.add_argument("--out", default="",
                    help="output manifest (default: <downloads dir>/download_curated.tsv)")
    ap.add_argument("--report", default="",
                    help="also write the report as JSON")
    ap.add_argument("--tiara1-statuses", default="new",
                    help="comma-separated import statuses to force-keep "
                         "(default: new; anchors are always kept)")
    ap.add_argument("--allow-missing-anchors", action="store_true",
                    help="exit 0 even when anchor accessions are absent "
                         "(default: exit 2 -- a missing anchor invalidates the "
                         "Tiara1 comparison)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report only, write nothing")
    add_progress_args(ap)
    args = ap.parse_args(argv)

    prog = counter_from_args(args, label="merge")

    downloads, dl_fields = read_tsv(args.downloads, prog, "download_all")
    curation, _ = read_tsv(args.curation, prog, "curation_selection")
    tiara1_rows, _ = ([], []) if not args.tiara1 else read_tsv(
        args.tiara1, prog, "tiara1_selection")
    required = tiara1_required(tiara1_rows,
                              str(args.tiara1_statuses).split(","))
    if tiara1_rows:
        prog.write(f"tiara1: {len(required):,} of {len(tiara1_rows):,} row(s) "
                   f"are anchors or {args.tiara1_statuses}")
    anchors = read_anchors(args.anchors)

    out_rows, missing, report = merge(downloads, dl_fields, curation,
                                     required, anchors, prog=prog)
    prog.finish()

    out_path = Path(args.out) if args.out else (
        Path(args.downloads).parent / "download_curated.tsv")
    columns = list(dl_fields) + [c for c in APPENDED if c not in dl_fields]

    print(render(report))
    print("")
    if args.dry_run:
        print(f"dry run: would write {len(out_rows):,} row(s) -> {out_path}")
    else:
        write_tsv(out_path, out_rows, columns)
        print(f"manifest -> {out_path}")
        if missing:
            miss_path = out_path.parent / "missing_from_downloads.tsv"
            write_tsv(miss_path, missing, MISSING_COLUMNS)
            print(f"missing  -> {miss_path}  ({len(missing):,} row(s))")
        if args.report:
            Path(args.report).parent.mkdir(parents=True, exist_ok=True)
            Path(args.report).write_text(
                json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
            print(f"report   -> {args.report}")

    if not out_rows:
        print("\nERROR: nothing matched. Check that the two plans describe the "
              "same round (a stale download_all.tsv from the previous corpus "
              "tag is the usual cause).", file=sys.stderr)
        return 2
    if report["anchors_missing"] and not args.allow_missing_anchors:
        print(f"\nERROR: {len(report['anchors_missing'])} anchor accession(s) "
              "are not in the merged manifest. Our euk set is therefore NOT a "
              "superset of Tiara1's, so the head-to-head comparison would be "
              "invalid. Fix the input tables, or override with "
              "--allow-missing-anchors if you genuinely intend to drop them.",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Subset an existing corpus down to a curation plan, without re-downloading.

WHY THIS EXISTS
---------------
``curate`` writes a PLAN (``curation_selection.tsv``): which assemblies to keep
and how many base pairs to draw from each one.  Nothing downstream reads that
plan -- ``chop_bin`` takes a single input, ``source_ready``, and chops whatever
it finds there.  So a plan only becomes real at the moment ``source_ready`` is
built.

When the genomes are already on disk from a previous corpus, rebuilding
``source_ready`` from ``store/`` is pure I/O for no new information.  This
script instead filters the previous corpus in one streaming pass:

    old source_ready/<split>/<class>.fasta   (read-only, usually on the HDD)
        -> keep fragments whose accession is in the plan
        -> stop at that accession's target_bp
    new source_ready/<split>/<class>.fasta   (written to the NVMe tier)

TWO THINGS THAT WILL BITE YOU, BOTH HANDLED HERE
------------------------------------------------
1. THE CLASS IS IN ``sg``, NOT IN ``label``.  An organelle record is routinely
   tagged ``label=euk sg=Organelle-Mito``, because its host really is a
   eukaryote.  ``label`` answers "whose genome is this", ``sg`` answers "which
   compartment".  Trusting ``label`` collapses every mitochondrion and plastid
   into ``eukarya`` and deletes two classes.  Every record is therefore routed
   by ``tiara2.labels.class_from_header``, which checks the compartment first
   and only ever compares supergroups with ``==``.  The class in the *file
   name* is treated as a hint, never as the answer, and every disagreement is
   counted and reported (``--on-class-mismatch`` decides what happens).

2. THE SOURCE AND THE DESTINATION ARE ON DIFFERENT TIERS.  The old corpus sits
   on the cold array; the new one belongs on the NVMe.  Both roots are
   therefore explicit arguments, the script refuses to write into its own
   input, and it reports the device of each side so a mistake is visible before
   the multi-hour pass rather than after it.

The pass is streaming and single-threaded: it never holds a fragment set in
memory, only per-accession byte counters.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for _p in (str(HERE), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import iter_fasta  # noqa: E402
from progress import add_progress_args, counter_from_args  # noqa: E402
from tiara2 import labels  # noqa: E402

ACCESSION_KEYS = ("assembly_accession", "accession", "entity_id", "acc")
TARGET_KEYS = ("target_bp",)
ANCHOR_KEYS = ("anchor", "is_anchor", "curate_anchor")

MISSING_COLUMNS = ("accession", "reason", "target_bp", "anchor",
                   "organism_name", "clade")
SHORTFALL_COLUMNS = ("accession", "target_bp", "available_bp", "deficit_bp",
                     "fragments", "anchor", "class")


# --------------------------------------------------------------------------- #
# plan side
# --------------------------------------------------------------------------- #

def base_accession(acc: str) -> str:
    """``GCA_000411095.2`` -> ``GCA_000411095``.  Version-tolerant matching."""
    acc = (acc or "").strip()
    return acc.rsplit(".", 1)[0] if "." in acc else acc


def pick_column(fieldnames, candidates, *, what: str, path: Path) -> str:
    lowered = {(f or "").strip().lower(): f for f in (fieldnames or [])}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    raise SystemExit(
        f"cannot find the {what} column in {path}\n"
        f"  looked for : {', '.join(candidates)}\n"
        f"  columns are: {', '.join(str(f) for f in (fieldnames or []))}")


def read_tsv(path: Path):
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        return list(reader), list(reader.fieldnames or [])


def read_anchors(path: Path) -> set:
    out = set()
    if not path:
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if line:
                out.add(line)
    return out


def as_int(value) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def build_plan(rows, columns, path: Path, anchors: set, *, default_target=None):
    """Return ``(exact, by_base)`` lookups of accession -> plan entry."""
    acc_col = pick_column(columns, ACCESSION_KEYS, what="accession", path=path)
    target_col = next((c for c in columns
                       if (c or "").strip().lower() in TARGET_KEYS), None)
    anchor_col = next((c for c in columns
                       if (c or "").strip().lower() in ANCHOR_KEYS), None)

    exact: dict = {}
    by_base: dict = {}
    for row in rows:
        acc = str(row.get(acc_col, "")).strip()
        if not acc:
            continue
        target = as_int(row.get(target_col)) if target_col else None
        if target is None:
            target = default_target
        anchor = acc in anchors or base_accession(acc) in {
            base_accession(a) for a in anchors}
        if not anchor and anchor_col:
            anchor = str(row.get(anchor_col, "")).strip().lower() in (
                "1", "true", "yes", "y")
        entry = {
            "accession": acc,
            "target_bp": target,
            "anchor": anchor,
            "organism_name": str(row.get("organism_name", "") or ""),
            "clade": str(row.get("clade", "") or ""),
            "kept_bp": 0,
            "kept_fragments": 0,
            "available_bp": 0,
            "class": "",
        }
        exact[acc] = entry
        by_base.setdefault(base_accession(acc), entry)
    return exact, by_base, acc_col


# --------------------------------------------------------------------------- #
# corpus side
# --------------------------------------------------------------------------- #

def discover_inputs(source_root: Path, splits):
    """Yield ``(split, class_hint, path)`` for every FASTA under the corpus."""
    found = []
    for split_dir in sorted(p for p in source_root.iterdir() if p.is_dir()):
        if splits and split_dir.name not in splits:
            continue
        for fasta in sorted(split_dir.glob("*.fasta")):
            found.append((split_dir.name, fasta.stem, fasta))
    return found


def device_of(path: Path) -> str:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return str(os.stat(probe).st_dev)
    except OSError:
        return "?"


class Writers:
    """Lazily opened ``<dest>/<split>/<class>.fasta`` handles.

    Files are written as ``.part`` and renamed only once the whole pass
    succeeded, so an interrupted run can never leave a corpus that looks
    complete to ``chop_bin``.
    """

    def __init__(self, dest_root: Path, enabled: bool = True):
        self.dest_root = dest_root
        self.enabled = enabled
        self.handles: dict = {}
        self.counts: dict = {}

    def write(self, split: str, klass: str, header: str, seq_lines):
        key = (split, klass)
        stat = self.counts.setdefault(key, {"fragments": 0, "bp": 0})
        stat["fragments"] += 1
        stat["bp"] += sum(len(s) for s in seq_lines)
        if not self.enabled:
            return
        fh = self.handles.get(key)
        if fh is None:
            out_dir = self.dest_root / split
            out_dir.mkdir(parents=True, exist_ok=True)
            fh = open(out_dir / f"{klass}.fasta.part", "w", encoding="utf-8")
            self.handles[key] = fh
        fh.write(">" + header + "\n")
        for line in seq_lines:
            fh.write(line + "\n")

    def commit(self):
        for fh in self.handles.values():
            fh.close()
        for (split, klass) in list(self.handles):
            part = self.dest_root / split / f"{klass}.fasta.part"
            part.replace(self.dest_root / split / f"{klass}.fasta")
        self.handles.clear()

    def abort(self):
        for fh in self.handles.values():
            try:
                fh.close()
            except OSError:
                pass
        self.handles.clear()


def subset(args, plan_exact, plan_base, inputs, writers, counter):
    """Apply a genome plan to selected classes and re-route by header metadata."""
    plan_classes=set(args.plan_classes)
    canonical=set(labels.FINE_CLASSES)
    stats={"records_read":0,"records_kept":0,"bp_kept":0,"not_in_plan":0,
           "over_budget":0,"passthrough_records":0,"matched_other_version":0,
           "class_mismatch":0,"legacy_container_rerouted":0,"unclassifiable":0,
           "per_file":[],"mismatch_examples":[]}
    for split,class_hint,path in inputs:
        per={"split":split,"class_hint":class_hint,"path":str(path),"read":0,"kept":0,"bp":0}
        for frag_id,header,seq_lines in iter_fasta(path):
            stats["records_read"]+=1; per["read"]+=1; counter.count()
            acc=labels.accession_from_frag_id(frag_id)
            klass=labels.class_from_header(header)
            if klass is None:
                stats["unclassifiable"]+=1
                raise SystemExit(f"cannot classify {frag_id!r} from header: {header}")
            if class_hint in canonical and klass!=class_hint:
                stats["class_mismatch"]+=1
                if len(stats["mismatch_examples"])<20:
                    stats["mismatch_examples"].append({"frag_id":frag_id,"file_class":class_hint,
                        "header_class":klass,"header":header})
                if args.on_class_mismatch=="fail":
                    raise SystemExit(f"class mismatch for {frag_id}: file={class_hint}, header={klass}")
                if args.on_class_mismatch=="hint": klass=class_hint
            elif class_hint not in canonical:
                stats["legacy_container_rerouted"]+=1
            entry=None
            if klass in plan_classes:
                entry=plan_exact.get(acc) or plan_base.get(base_accession(acc))
                if entry is None:
                    stats["not_in_plan"]+=1; continue
                if acc not in plan_exact: stats["matched_other_version"]+=1
            else:
                stats["passthrough_records"]+=1
            length=sum(len(s) for s in seq_lines)
            if entry is not None:
                entry["available_bp"]+=length
                target=None if args.ignore_target_bp else entry["target_bp"]
                if target is not None and entry["kept_bp"]>=target:
                    stats["over_budget"]+=1; continue
                entry["kept_bp"]+=length; entry["kept_fragments"]+=1
                entry["class"]=entry["class"] or klass
            stats["records_kept"]+=1; stats["bp_kept"]+=length
            per["kept"]+=1; per["bp"]+=length; writers.write(split,klass,header,seq_lines)
        stats["per_file"].append(per)
    return stats


# --------------------------------------------------------------------------- #
# parallel file workers
# --------------------------------------------------------------------------- #

class _NullCounter:
    def count(self, *args, **kwargs):
        return None

    def finish(self):
        return None


def _plan_updates(plan_exact):
    """Return only mutable counters changed by one worker."""
    out = {}
    for acc, entry in plan_exact.items():
        if entry["available_bp"] or entry["kept_fragments"]:
            out[acc] = {
                "available_bp": entry["available_bp"],
                "kept_bp": entry["kept_bp"],
                "kept_fragments": entry["kept_fragments"],
                "class": entry["class"],
            }
    return out


def _subset_file_worker(payload):
    """Filter one input FASTA into deterministic per-worker shards."""
    index, args, plan_exact, plan_base, input_item, worker_root = payload
    root = Path(worker_root)
    writers = Writers(root, enabled=not args.dry_run)
    try:
        stats = subset(args, plan_exact, plan_base, [input_item], writers,
                       _NullCounter())
        if not args.dry_run:
            writers.commit()
        return {
            "index": index,
            "root": str(root),
            "stats": stats,
            "counts": writers.counts,
            "updates": _plan_updates(plan_exact),
        }
    except BaseException:
        writers.abort()
        raise


def _merge_worker_stats(results):
    numeric = (
        "records_read", "records_kept", "bp_kept", "not_in_plan",
        "over_budget", "passthrough_records", "matched_other_version",
        "class_mismatch", "legacy_container_rerouted", "unclassifiable",
    )
    merged = {key: 0 for key in numeric}
    merged.update({"per_file": [], "mismatch_examples": []})
    counts = {}
    for result in sorted(results, key=lambda r: r["index"]):
        stats = result["stats"]
        for key in numeric:
            merged[key] += stats[key]
        merged["per_file"].extend(stats["per_file"])
        room = 20 - len(merged["mismatch_examples"])
        if room > 0:
            merged["mismatch_examples"].extend(
                stats["mismatch_examples"][:room])
        for key, value in result["counts"].items():
            total = counts.setdefault(key, {"fragments": 0, "bp": 0})
            total["fragments"] += value["fragments"]
            total["bp"] += value["bp"]
    return merged, counts


def _apply_worker_updates(plan_exact, results):
    for result in results:
        for acc, update in result["updates"].items():
            entry = plan_exact[acc]
            entry["available_bp"] += update["available_bp"]
            entry["kept_bp"] += update["kept_bp"]
            entry["kept_fragments"] += update["kept_fragments"]
            if not entry["class"] and update["class"]:
                entry["class"] = update["class"]


def _merge_shards(results, dest_root: Path, buffer_mb: int):
    """Merge in source-file order, deleting each shard immediately.

    Peak temporary SSD use is the selected shards plus at most the currently
    merged output class; it is never a second complete corpus copy.
    """
    ordered = sorted(results, key=lambda r: r["index"])
    keys = sorted({key for result in ordered for key in result["counts"]})
    buf = max(1, int(buffer_mb)) * (1 << 20)
    for split, klass in keys:
        out_dir = dest_root / split
        out_dir.mkdir(parents=True, exist_ok=True)
        part = out_dir / f"{klass}.fasta.part"
        with open(part, "wb", buffering=buf) as dst:
            for result in ordered:
                shard = Path(result["root"]) / split / f"{klass}.fasta"
                if not shard.is_file():
                    continue
                with open(shard, "rb", buffering=buf) as src:
                    shutil.copyfileobj(src, dst, length=buf)
                shard.unlink()
        part.replace(out_dir / f"{klass}.fasta")


def parallel_subset(args, plan_exact, plan_base, inputs, dest_root: Path,
                    temp_root: Path):
    if not args.ignore_target_bp:
        raise SystemExit(
            "--workers > 1 requires --ignore-target-bp: a shared target_bp "
            "counter cannot be applied deterministically across workers")
    if temp_root == dest_root or temp_root in dest_root.parents:
        raise SystemExit("parallel temp root must not contain the destination")
    temp_root.mkdir(parents=True, exist_ok=False)
    payloads = [
        (i, args, plan_exact, plan_base, item,
         str(temp_root / f"worker_{i:03d}"))
        for i, item in enumerate(inputs)
    ]
    results = []
    try:
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=min(args.workers, len(inputs))) as pool:
            futures = {pool.submit(_subset_file_worker, p): p[0]
                       for p in payloads}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                item = inputs[result["index"]]
                print(f"[subset] completed {result['index'] + 1}/{len(inputs)}: "
                      f"{item[0]}/{item[1]} ({result['stats']['records_read']:,} records)",
                      flush=True)
        stats, counts = _merge_worker_stats(results)
        _apply_worker_updates(plan_exact, results)
        if not args.dry_run:
            _merge_shards(results, dest_root, args.merge_buffer_mb)
        return stats, counts
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)

# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

def write_tsv(path: Path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns), delimiter="\t",
                               extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def render(report: dict) -> str:
    lines = []
    lines.append("corpus subset")
    lines.append(f"  source        : {report['source_root']} (dev {report['source_device']})")
    lines.append(f"  destination   : {report['dest_root']} (dev {report['dest_device']})")
    if report["source_device"] == report["dest_device"]:
        lines.append("    !! same device: the new corpus is NOT on a separate tier")
    lines.append(f"  plan rows     : {report['plan_rows']:,}")
    lines.append(f"  records read  : {report['records_read']:,}")
    lines.append(f"  records kept  : {report['records_kept']:,}  ({report['bp_kept']:,} bp)")
    lines.append(f"  dropped       : {report['not_in_plan']:,} not in plan, "
                 f"{report['over_budget']:,} over target_bp")
    if report["matched_other_version"]:
        lines.append(f"  version drift : {report['matched_other_version']:,} "
                     "records matched the plan on base accession")
    lines.append(f"  class routing : {report['class_mismatch']:,} header/file "
                 f"disagreements, {report['unclassifiable']:,} unclassifiable "
                 f"(policy: {report['on_class_mismatch']})")
    for ex in report["mismatch_examples"][:5]:
        lines.append(f"      {ex['frag_id']}: file={ex['file_class']} "
                     f"header={ex['header_class']}")
    lines.append("  per split/class:")
    for row in report["per_class"]:
        lines.append("      %-12s %-14s %10s frags %16s bp"
                     % (row["split"], row["class"],
                        f"{row['fragments']:,}", f"{row['bp']:,}"))
    lines.append(f"  accessions    : {report['accessions_found']:,} found, "
                 f"{report['accessions_missing']:,} missing, "
                 f"{report['accessions_short']:,} short of target_bp")
    lines.append(f"  anchors       : {report['anchors_total']:,} required, "
                 f"{report['anchors_missing']:,} missing")
    if report["missing_path"]:
        lines.append(f"  missing list  -> {report['missing_path']}")
    if report["shortfall_path"]:
        lines.append(f"  shortfall list-> {report['shortfall_path']}")
    return "\n".join(lines)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Filter an existing corpus down to a curation plan.")
    ap.add_argument("--source-root", required=True,
                    help="existing corpus root: <split>/<class>.fasta "
                         "(read-only; typically the previous train_ready on /data)")
    ap.add_argument("--dest-root", required=True,
                    help="new source_ready root to write (typically on the NVMe tier)")
    ap.add_argument("--selection", required=True,
                    help="curation_selection.tsv from the curate stage")
    ap.add_argument("--anchors", default="",
                    help="anchor accession list; every anchor must survive")
    ap.add_argument("--splits", default="train,validation,test",
                    help="comma-separated splits; test_benchmark excluded by default")
    ap.add_argument("--plan-classes",
                    default="bacteria,archaea,eukarya,mitochondria,plastids",
                    help="classes governed by the genome/genus selection plan")
    ap.add_argument("--ignore-target-bp", action="store_true",
                    help="genome selection only; keep every fragment of selected genomes")
    ap.add_argument("--default-target-bp", type=int, default=0,
                    help="bp cap for plan rows without target_bp (0 = uncapped)")
    ap.add_argument("--on-class-mismatch", choices=("header", "hint", "fail"),
                    default="header",
                    help="header = trust sg (default), hint = trust the file "
                         "name, fail = stop on the first disagreement")
    ap.add_argument("--report", default="", help="write the JSON report here")
    ap.add_argument("--allow-missing-anchors", action="store_true")
    ap.add_argument("--defer-missing-anchors", action="store_true",
                    help="record undownloaded Tiara anchors without failing this round")
    ap.add_argument("--allow-same-device", action="store_true",
                    help="permit a destination on the same device as the source")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing non-empty destination")
    ap.add_argument("--dry-run", action="store_true",
                    help="count everything, write no FASTA")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel input-FASTA readers (recommended: 3-4 on "
                         "an HDD array; requires --ignore-target-bp)")
    ap.add_argument("--temp-root", default="",
                    help="worker shard directory; default is a hidden sibling "
                         "of dest-root on SSD and is removed after merging")
    ap.add_argument("--merge-buffer-mb", type=int, default=64,
                    help="buffer used while merging SSD shards (default: 64)")
    ap.add_argument("--min-free-gb", type=float, default=500.0,
                    help="refuse parallel writes when the temporary tier has "
                         "less free space than this reserve (default: 500 GiB)")
    add_progress_args(ap)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.merge_buffer_mb < 1:
        raise SystemExit("--merge-buffer-mb must be >= 1")
    if args.min_free_gb < 0:
        raise SystemExit("--min-free-gb must be >= 0")
    args.plan_classes=[x.strip() for x in args.plan_classes.split(",") if x.strip()]
    unknown=set(args.plan_classes)-set(labels.FINE_CLASSES)
    if unknown: raise SystemExit("unknown --plan-classes: "+", ".join(sorted(unknown)))
    source_root = Path(args.source_root).expanduser().resolve()
    dest_root = Path(args.dest_root).expanduser().resolve()
    selection = Path(args.selection).expanduser()

    if not source_root.is_dir():
        raise SystemExit(f"source root is not a directory: {source_root}")
    if dest_root == source_root or dest_root in source_root.parents:
        raise SystemExit(
            f"destination {dest_root} would overwrite its own input "
            f"{source_root}; pick a separate root")
    existing = list(dest_root.glob("*/*.fasta")) if dest_root.is_dir() else []
    if existing and not (args.force or args.dry_run):
        raise SystemExit(
            f"destination already holds {len(existing)} FASTA file(s): "
            f"{dest_root}\nRerun with --force to overwrite.")

    src_dev, dst_dev = device_of(source_root), device_of(dest_root)
    if src_dev == dst_dev and not (args.allow_same_device or args.dry_run):
        raise SystemExit(
            f"source and destination are on the same device (dev {src_dev}).\n"
            f"  source: {source_root}\n  dest  : {dest_root}\n"
            "The new corpus is meant to live on the NVMe tier. Rerun with "
            "--allow-same-device if the tiers really are collapsed.")

    anchors = read_anchors(Path(args.anchors).expanduser()) if args.anchors else set()
    rows, columns = read_tsv(selection)
    plan_exact, plan_base, acc_col = build_plan(
        rows, columns, selection, anchors,
        default_target=args.default_target_bp or None)
    if not plan_exact:
        raise SystemExit(f"no usable rows in {selection}")

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    inputs = discover_inputs(source_root, splits)
    if not inputs:
        raise SystemExit(
            f"no <split>/<class>.fasta found under {source_root}\n"
            "Check that this is a corpus root and not its parent.")

    parallel = args.workers > 1
    temp_root = None
    if parallel:
        if args.temp_root:
            temp_root = Path(args.temp_root).expanduser().resolve()
        else:
            temp_root = (dest_root.parent /
                         f".{dest_root.name}.subset_tmp.{os.getpid()}")
        probe = temp_root
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        free = shutil.disk_usage(probe).free
        reserve = int(args.min_free_gb * (1 << 30))
        if free < reserve and not args.dry_run:
            raise SystemExit(
                f"parallel temp tier has only {free / (1 << 40):.2f} TiB free; "
                f"reserve is {args.min_free_gb:g} GiB: {probe}")
        print(f"[subset] parallel workers={min(args.workers, len(inputs))}; "
              f"temporary shards={temp_root}; free={free / (1 << 40):.2f} TiB",
              flush=True)
        stats, parallel_counts = parallel_subset(
            args, plan_exact, plan_base, inputs, dest_root, temp_root)
        writers = Writers(dest_root, enabled=False)
        writers.counts = parallel_counts
    else:
        counter = counter_from_args(args, "subset", noun="record")
        writers = Writers(dest_root, enabled=not args.dry_run)
        try:
            stats = subset(args, plan_exact, plan_base, inputs, writers, counter)
        except BaseException:
            writers.abort()
            raise
        counter.finish()

    per_class = [
        {"split": split, "class": klass, "fragments": v["fragments"], "bp": v["bp"]}
        for (split, klass), v in sorted(writers.counts.items())
    ]

    missing, shortfall = [], []
    for entry in plan_exact.values():
        if entry["kept_fragments"] == 0:
            missing.append({
                "accession": entry["accession"],
                "reason": "absent_from_source_corpus",
                "target_bp": entry["target_bp"] or "",
                "anchor": "1" if entry["anchor"] else "0",
                "organism_name": entry["organism_name"],
                "clade": entry["clade"],
            })
            continue
        target = entry["target_bp"]
        if target and entry["kept_bp"] < target:
            shortfall.append({
                "accession": entry["accession"],
                "target_bp": target,
                "available_bp": entry["available_bp"],
                "deficit_bp": target - entry["kept_bp"],
                "fragments": entry["kept_fragments"],
                "anchor": "1" if entry["anchor"] else "0",
                "class": entry["class"],
            })

    anchors_missing = sorted(
        acc for acc in anchors
        if (plan_exact.get(acc) or plan_base.get(base_accession(acc)) or
            {"kept_fragments": 0})["kept_fragments"] == 0)

    missing_path = shortfall_path = ""
    if not args.dry_run:
        dest_root.mkdir(parents=True, exist_ok=True)
        if missing:
            missing_path = str(dest_root / "missing_from_source_corpus.tsv")
            write_tsv(Path(missing_path), missing, MISSING_COLUMNS)
        if shortfall:
            shortfall_path = str(dest_root / "shortfall_vs_target_bp.tsv")
            write_tsv(Path(shortfall_path), shortfall, SHORTFALL_COLUMNS)

    report = {
        "source_root": str(source_root),
        "dest_root": str(dest_root),
        "source_device": src_dev,
        "dest_device": dst_dev,
        "selection": str(selection),
        "accession_column": acc_col,
        "plan_rows": len(plan_exact),
        "on_class_mismatch": args.on_class_mismatch,
        "dry_run": bool(args.dry_run),
        "workers": int(args.workers),
        "parallel": bool(parallel),
        "temporary_shards": str(temp_root) if temp_root else "",
        "records_read": stats["records_read"],
        "records_kept": stats["records_kept"],
        "bp_kept": stats["bp_kept"],
        "not_in_plan": stats["not_in_plan"],
        "over_budget": stats["over_budget"],
        "passthrough_records": stats["passthrough_records"],
        "plan_classes": args.plan_classes,
        "ignore_target_bp": bool(args.ignore_target_bp),
        "legacy_container_rerouted": stats["legacy_container_rerouted"],
        "matched_other_version": stats["matched_other_version"],
        "class_mismatch": stats["class_mismatch"],
        "unclassifiable": stats["unclassifiable"],
        "mismatch_examples": stats["mismatch_examples"],
        "per_class": per_class,
        "per_file": stats["per_file"],
        "accessions_found": len(plan_exact) - len(missing),
        "accessions_missing": len(missing),
        "accessions_short": len(shortfall),
        "anchors_total": len(anchors),
        "anchors_missing": len(anchors_missing),
        "anchors_missing_list": anchors_missing,
        "missing_path": missing_path,
        "shortfall_path": shortfall_path,
    }

    if args.dry_run:
        writers.abort()
    elif not parallel:
        writers.commit()

    if args.report:
        out = Path(args.report).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(render(report))
    if args.dry_run:
        print("  DRY RUN: no FASTA written")

    if anchors_missing and not (args.allow_missing_anchors or args.defer_missing_anchors):
        print("\n!! %d anchor(s) produced no fragments: %s"
              % (len(anchors_missing), ", ".join(anchors_missing[:10])),
              file=sys.stderr)
        print("   These are the Tiara1 genomes the whole comparison rests on. "
              "Acquire them before training, or pass "
              "--allow-missing-anchors deliberately.", file=sys.stderr)
        return 2
    if stats["records_kept"] == 0:
        print("\n!! nothing was kept: no accession in the plan appears in the "
              "source corpus. A plan built for a different corpus tag is the "
              "usual cause.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

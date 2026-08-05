#!/usr/bin/env python3
"""Merge the downloaded v2 index with Tiara S1 at INDEX level.

No sequence is downloaded. Tiara-only rows are preserved in the merged index
and deferred list; only already downloaded v2 rows from the five training
classes enter ``current_available_candidates.tsv``.
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
from progress import add_progress_args, counter_from_args

FIVE = ("bacteria", "archaea", "eukarya", "mitochondria", "plastids")
OK = {"ok", "cached", "cached_no_md5"}
ACC_KEYS = ("assembly_accession", "accession", "entity_id", "sequence_accession")
EXTRA = ("training_class", "index_source", "available_current_round",
         "availability_reason", "v2_accession", "tiara1_accession",
         "tiara1_status", "tiara1_anchor", "accession_match")


def base_acc(x):
    x = str(x or "").strip()
    return x.rsplit(".", 1)[0] if "." in x else x


def read_tsv(path):
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"required TSV does not exist: {path}")
    with path.open(encoding="utf-8", newline="") as fh:
        r = csv.DictReader(fh, delimiter="\t")
        return list(r), list(r.fieldnames or [])


def write_tsv(path, rows, cols):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def first(row, keys):
    for key in keys:
        value = str(row.get(key, "") or "").strip()
        if value:
            return value
    return ""


def klass(row):
    raw = first(row, ("training_class", "klass", "fine_class")).lower()
    aliases = {
        "bacteria":"bacteria", "archaea":"archaea",
        "nuclear_euk":"eukarya", "euk":"eukarya", "eukarya":"eukarya",
        "eukaryota":"eukarya", "mitochondrion":"mitochondria",
        "mitochondria":"mitochondria", "mito":"mitochondria",
        "plastid":"plastids", "plastids":"plastids", "chloroplast":"plastids",
    }
    if raw in aliases:
        return aliases[raw]
    group = first(row, ("group_name", "group")).lower()
    groups = {
        "bacteria":"bacteria", "archaea":"archaea", "fungi":"eukarya",
        "protozoa":"eukarya", "plant":"eukarya", "invertebrate":"eukarya",
        "vertebrate_other":"eukarya", "vertebrate_mammalian":"eukarya",
        "jgi_eukaryota":"eukarya", "mitochondrion":"mitochondria",
        "mitochondria":"mitochondria", "plastid":"plastids", "plastids":"plastids",
    }
    return groups.get(group, "")


def truthy(x):
    return str(x or "").lower() in {"1", "true", "yes", "y"}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Full-union v2 and Tiara S1 indexes")
    ap.add_argument("--existing", required=True)
    ap.add_argument("--download-status", required=True)
    ap.add_argument("--tiara1", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--data-root", default="",
        help="Tiara2 data root containing store/organelle; default is inferred "
             "from <data-root>/index/selected.tsv")
    ap.add_argument("--dry-run", action="store_true")
    add_progress_args(ap); a = ap.parse_args(argv)

    old, old_cols = read_tsv(a.existing)
    status, status_cols = read_tsv(a.download_status)
    tiara, tiara_cols = read_tsv(a.tiara1)
    existing_path = Path(a.existing).expanduser().resolve()
    data_root = (Path(a.data_root).expanduser().resolve() if a.data_root else
                 existing_path.parent.parent)
    organelle_store = data_root / "store" / "organelle"
    good_exact, good_base = set(), set()
    for r in status:
        if first(r, ("result", "status", "download_status")).lower() in OK:
            acc = first(r, ("entity_id", "assembly_accession", "accession"))
            if acc: good_exact.add(acc); good_base.add(base_acc(acc))

    merged, exact, bases = [], {}, {}
    c = counter_from_args(a, "merge-indexes", total=len(old)+len(tiara), noun="row")
    organelle_local = 0
    for src in old:
        c.count(); row = dict(src); acc = first(row, ACC_KEYS)
        if not acc: continue
        entity_type = first(row, ("entity_type",)).lower()
        # Assembly downloads are recorded in download_status.tsv.  RefSeq
        # mitochondria/plastids take a different path: fetch-organelle writes
        # store/organelle/<sequence_accession>.fna.gz and intentionally never
        # creates a download_status row.  Requiring that row silently removed
        # every one of the 33k locally held organelles from the genus plan.
        organelle_path = organelle_store / f"{acc}.fna.gz"
        local_organelle = entity_type == "organelle" and organelle_path.is_file()
        available = (acc in good_exact or base_acc(acc) in good_base or
                     local_organelle)
        if local_organelle:
            organelle_local += 1
        reason = ("organelle_store" if local_organelle else
                  "download_status_ok" if available else "not_downloaded")
        row.update({"training_class":klass(row), "index_source":"v2",
                    "available_current_round":"1" if available else "0",
                    "availability_reason":reason,
                    "v2_accession":acc, "tiara1_accession":"", "tiara1_status":"",
                    "tiara1_anchor":"0", "accession_match":"v2_only"})
        merged.append(row); exact[acc]=row; bases.setdefault(base_acc(acc), row)
    counts = {"both_exact":0, "both_other_version":0, "tiara1_only":0}
    for tr in tiara:
        c.count(); ta = first(tr, ACC_KEYS)
        if not ta: continue
        hit, how = exact.get(ta), "exact"
        if hit is None: hit, how = bases.get(base_acc(ta)), "other_version"
        if hit is not None:
            counts["both_"+how] += 1
            hit.update({"index_source":"v2+tiara1", "tiara1_accession":ta,
                        "tiara1_status":tr.get("status", ""),
                        "tiara1_anchor":"1" if truthy(tr.get("is_anchor")) else "0",
                        "accession_match":how})
        else:
            row = {k:"" for k in old_cols}
            row.update(tr); row.update({"entity_id":ta, "assembly_accession":ta,
                "training_class":klass(tr), "index_source":"tiara1",
                "available_current_round":"0", "availability_reason":"tiara1_download_deferred",
                "v2_accession":"", "tiara1_accession":ta,
                "tiara1_status":tr.get("status", ""),
                "tiara1_anchor":"1" if truthy(tr.get("is_anchor")) else "0",
                "accession_match":"tiara1_only"})
            merged.append(row); exact[ta]=row; bases.setdefault(base_acc(ta), row)
            counts["tiara1_only"] += 1
    c.finish()

    current, deferred = [], []
    for row in merged:
        acc = first(row, ACC_KEYS); k = row.get("training_class", "")
        if row.get("available_current_round") == "1" and k in FIVE:
            r = dict(row); r["assembly_accession"] = acc; current.append(r)
        elif row.get("index_source") != "v2" or row.get("available_current_round") != "1":
            deferred.append(row)
    counts.update({"v2_rows":len(old), "tiara1_rows":len(tiara),
                   "merged_rows":len(merged), "current_candidates":len(current),
                   "deferred_rows":len(deferred),
                   "local_organelle_files_matched":organelle_local})
    cols = list(old_cols)
    for x in tiara_cols + list(EXTRA):
        if x not in cols: cols.append(x)
    if "assembly_accession" not in cols: cols.insert(0, "assembly_accession")
    out = Path(a.out_dir)
    if not a.dry_run:
        write_tsv(out/"merged_training_index.tsv", merged, cols)
        write_tsv(out/"current_available_candidates.tsv", current, cols)
        write_tsv(out/"deferred_downloads.tsv", deferred, cols)
        (out/"merge_training_indexes_report.json").write_text(json.dumps(counts, indent=2))
    print("training index union")
    for k,v in counts.items(): print(f"  {k:26s}: {v:,}")
    if a.dry_run: print("  DRY RUN: no files written")
    else:
        print(f"  current candidates -> {out/'current_available_candidates.tsv'}")
        print(f"  deferred downloads -> {out/'deferred_downloads.tsv'}")
    return 0

if __name__ == "__main__": sys.exit(main())

#!/usr/bin/env python3
"""Fill Tiara2 v2.3 hierarchical metadata from accession->taxid + NCBI taxdump.

Run inside the Tiara2 repository/environment. The input TSV is replaced
atomically after a backup and a resolution report are written.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path

from tiara2.taxonomy import Taxonomy, find_taxdump

ACCESSION_RE = re.compile(r"(GC[AF]_\d+\.\d+)")
VALID_EUK_GROUPS = {
    "fungi", "land_plant", "algae", "metazoa_vertebrate",
    "metazoa_invertebrate", "alveolata", "stramenopiles",
    "other_protist",
}
CLADE_REMAP = {"eukarya_other": "other_protist"}


def accession_of(text: str) -> str:
    match = ACCESSION_RE.search(text or "")
    return match.group(1) if match else ""


def load_accession_taxids(path: Path):
    mapping = {}
    malformed = 0
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 2:
                fields = line.split()
            if len(fields) < 2:
                malformed += 1
                continue
            accession = accession_of(fields[0])
            raw_taxid = fields[1].strip()
            if not accession or not raw_taxid.isdigit():
                malformed += 1
                continue
            mapping[accession] = int(raw_taxid)
    return mapping, malformed


def nearest_species_taxid(taxonomy: Taxonomy, taxid: int) -> int:
    for node, rank in taxonomy.ancestors(taxid):
        if rank == "species":
            return int(node)
    return int(taxid)


def resolve(metadata: Path, accession_taxid: Path, taxdump: Path) -> dict:
    if not metadata.is_file():
        raise FileNotFoundError(f"metadata not found: {metadata}")
    if not accession_taxid.is_file():
        raise FileNotFoundError(f"accession-taxid table not found: {accession_taxid}")
    found = find_taxdump(taxdump)
    if found is None:
        raise FileNotFoundError(f"nodes.dmp/names.dmp not found under: {taxdump}")

    print(f"[1/4] loading accession->taxid: {accession_taxid}", flush=True)
    acc2taxid, malformed = load_accession_taxids(accession_taxid)
    print(f"      mappings={len(acc2taxid):,}; skipped={malformed:,}", flush=True)
    print(f"[2/4] loading taxonomy: {found}", flush=True)
    taxonomy = Taxonomy.from_taxdump(found)

    temp_path = metadata.with_suffix(".resolved.tmp.tsv")
    backup_path = metadata.with_suffix(".before_taxonomy.tsv")
    report_path = metadata.with_suffix(".resolution_report.json")
    unresolved_path = metadata.with_suffix(".unresolved_accessions.tsv")
    counts = Counter()
    unresolved_accessions = Counter()

    print(f"[3/4] resolving: {metadata}", flush=True)
    with metadata.open(newline="", encoding="utf-8") as source, temp_path.open(
        "w", newline="", encoding="utf-8"
    ) as target:
        reader = csv.DictReader(source, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError("metadata TSV has no header")
        fieldnames = list(reader.fieldnames)
        for required in (
            "record_id", "split", "legacy_class", "species_taxid",
            "euk_group", "label_status",
        ):
            if required not in fieldnames:
                fieldnames.append(required)
        writer = csv.DictWriter(
            target, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore"
        )
        writer.writeheader()

        for row in reader:
            counts["rows"] += 1
            unresolved = (
                row.get("legacy_class") == "eukarya" and not row.get("euk_group")
            )
            if not unresolved:
                writer.writerow(row)
                continue
            counts["unresolved_input"] += 1
            accession = accession_of(row.get("record_id", ""))
            if not accession:
                counts["no_accession_in_record_id"] += 1
                row["label_status"] = "unresolved_no_accession"
                writer.writerow(row)
                continue
            taxid = acc2taxid.get(accession)
            if taxid is None:
                counts["accession_not_in_taxid_tsv"] += 1
                unresolved_accessions[accession] += 1
                row["label_status"] = "unresolved_no_taxid"
                writer.writerow(row)
                continue
            info = taxonomy.info(taxid)
            clade = CLADE_REMAP.get(info.clade, info.clade)
            if not info.ok:
                counts["taxid_not_in_taxdump"] += 1
                unresolved_accessions[accession] += 1
                row["label_status"] = "unresolved_taxdump"
                writer.writerow(row)
                continue
            if clade not in VALID_EUK_GROUPS:
                counts[f"invalid_clade:{clade}"] += 1
                unresolved_accessions[accession] += 1
                row["label_status"] = f"unresolved_clade:{clade}"
                writer.writerow(row)
                continue
            row["species_taxid"] = str(nearest_species_taxid(taxonomy, taxid))
            row["euk_group"] = clade
            row["label_status"] = "resolved_taxdump"
            counts["resolved"] += 1
            counts[f"resolved:{clade}"] += 1
            writer.writerow(row)
            if counts["rows"] % 250000 == 0:
                print(
                    f"      rows={counts['rows']:,}; resolved={counts['resolved']:,}",
                    flush=True,
                )

    with unresolved_path.open("w", encoding="utf-8") as handle:
        handle.write("accession\tfragment_records\n")
        for accession, number in sorted(unresolved_accessions.items()):
            handle.write(f"{accession}\t{number}\n")

    counts["unresolved_output"] = counts["unresolved_input"] - counts["resolved"]
    report = {
        "metadata": str(metadata),
        "backup": str(backup_path),
        "taxid_table": str(accession_taxid),
        "taxdump": str(found),
        "accession_taxid_mappings": len(acc2taxid),
        "counts": dict(sorted(counts.items())),
        "unresolved_accessions": len(unresolved_accessions),
        "unresolved_accessions_file": str(unresolved_path),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print("[4/4] publishing resolved metadata", flush=True)
    if backup_path.exists():
        backup_path.unlink()
    shutil.copy2(metadata, backup_path)
    os.replace(temp_path, metadata)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata",
        default="/data/shouhanyu/Tiara2_v2_3_0/metadata/hierarchical_labels.tsv",
    )
    parser.add_argument(
        "--accession-taxid",
        default="/data/shouhanyu/Tiara2/select/taxid.tsv",
    )
    parser.add_argument(
        "--taxdump", default="/data/shouhanyu/Tiara2/taxonomy"
    )
    args = parser.parse_args(argv)
    report = resolve(Path(args.metadata), Path(args.accession_taxid), Path(args.taxdump))
    return 0 if report["counts"].get("unresolved_output", 0) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

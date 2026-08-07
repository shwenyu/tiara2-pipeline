"""Build donor identities aligned to the frozen Euk feature-row order."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from tiara.hierarchical.data import fasta, rid
from tiara.hierarchical.freeze import sha256_file
from tiara.hierarchical.train import load_feature_set
from tiara2.taxonomy import Taxonomy, find_taxdump


def _codes(values):
    mapping = {}
    result = np.empty(len(values), dtype=np.int32)
    for index, value in enumerate(values):
        if value not in mapping:
            mapping[value] = len(mapping)
        result[index] = mapping[value]
    return result, mapping


def build(metadata, train_ready, features, taxdump_dir, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    db_path = out / ".metadata.sqlite"
    if db_path.exists():
        db_path.unlink()
    db = sqlite3.connect(db_path)
    db.execute("CREATE TABLE meta(record_id TEXT PRIMARY KEY, accession TEXT NOT NULL, species_taxid TEXT NOT NULL, euk_group TEXT NOT NULL) WITHOUT ROWID")
    indexed = 0
    with open(metadata, newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            group = row.get("euk_group", "").strip()
            if row.get("split") != "train" or not group:
                continue
            record = row.get("record_id", "").strip()
            accession = (row.get("accession") or record.split("|", 1)[0]).strip()
            taxid = row.get("species_taxid", "").strip()
            if not record or not accession or not taxid:
                raise ValueError(f"incomplete Euk donor metadata for {record!r}")
            db.execute("INSERT INTO meta VALUES(?,?,?,?)", (record, accession, taxid, group))
            indexed += 1
            if indexed % 100000 == 0:
                db.commit()
    db.commit()

    feature_manifest, schema, train_set, _ = load_feature_set(features)
    root = train_set.labels("root")
    leaf = train_set.labels("euk")
    euk_mask = root == schema.index("root")["euk_nuclear"]
    expected = int(euk_mask.sum())
    expected_leaf = leaf[euk_mask]
    taxdump = find_taxdump(taxdump_dir)
    if taxdump is None:
        raise FileNotFoundError(f"taxdump not found under {taxdump_dir}")
    taxonomy = Taxonomy.from_taxdump(taxdump)

    accessions, species, genera = [], [], []
    unresolved_genus_taxids = set()
    group_counts = {name: 0 for name in schema.classes("euk")}
    fasta_path = Path(train_ready) / "train" / "eukarya.fasta"
    for position, (header, _sequence) in enumerate(fasta(fasta_path)):
        record = rid(header)
        row = db.execute("SELECT accession,species_taxid,euk_group FROM meta WHERE record_id=?", (record,)).fetchone()
        if row is None:
            raise ValueError(f"frozen Euk FASTA record lacks metadata: {record}")
        accession, taxid, group = row
        if position >= expected:
            raise ValueError("frozen Euk FASTA has more rows than feature labels")
        resolved_leaf = schema.index("euk")[group]
        if int(expected_leaf[position]) != resolved_leaf:
            raise ValueError(f"feature/metadata leaf mismatch for {record}: {expected_leaf[position]} vs {group}")
        info = taxonomy.info(taxid)
        # New assemblies can carry taxids newer than the frozen taxdump.  A
        # species-specific synthetic key is conservative: it never merges two
        # unrelated donors and accession/species caps still remain active.
        genus_key = info.genus_key if info.ok and info.genus_key else "unresolved:t" + str(taxid)
        if genus_key.startswith("unresolved:"):
            unresolved_genus_taxids.add(str(taxid))
        accessions.append(accession)
        species.append("t" + str(taxid))
        genera.append(genus_key)
        group_counts[group] += 1
    if len(accessions) != expected:
        raise ValueError(f"frozen Euk row mismatch: FASTA={len(accessions)} features={expected}")

    arrays = {}
    unique = {}
    for name, values in (("accession", accessions), ("species", species), ("genus", genera)):
        array, mapping = _codes(values)
        path = out / f"{name}.i32.npy"
        np.save(path, array, allow_pickle=False)
        arrays[name] = str(path.resolve())
        unique[name] = len(mapping)
    manifest = {
        "schema_version": 1,
        "version": "2.3.2",
        "stage": "aligned_donor_index",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": expected,
        "unique": unique,
        "taxonomy_fallback": {
            "policy": "unresolved taxid gets a species-specific synthetic genus key",
            "unresolved_genus_taxids": len(unresolved_genus_taxids),
            "examples": sorted(unresolved_genus_taxids)[:20],
        },
        "euk_counts": group_counts,
        "arrays": arrays,
        "array_sha256": {name: sha256_file(path) for name, path in arrays.items()},
        "inputs": {
            "metadata": str(Path(metadata).resolve()),
            "metadata_sha256": sha256_file(metadata),
            "train_fasta": str(fasta_path.resolve()),
            "feature_manifest_version": feature_manifest.get("version"),
            "feature_manifest_format": feature_manifest.get("format"),
            "taxdump": str(taxdump.resolve()),
        },
        "ready": True,
    }
    manifest_path = out / "donor_index_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    db.close()
    for suffix in ("", "-wal", "-shm"):
        try:
            Path(str(db_path) + suffix).unlink()
        except FileNotFoundError:
            pass
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--train-ready", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--taxdump-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build(args.metadata, args.train_ready, args.features, args.taxdump_dir, args.out), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

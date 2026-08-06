import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tiara.hierarchical.completeness import audit_euk_completeness
from tiara.hierarchical.delta import build_euk_delta
from tiara.hierarchical.evaluate import evaluate
from tiara.hierarchical.freeze import freeze_base, verify_freeze
from tiara.hierarchical.gates import audit_combined_leakage
from tiara.hierarchical.schema import EUK


FIELDS = [
    "record_id",
    "accession",
    "split",
    "legacy_class",
    "species_taxid",
    "split_group_id",
    "duplicate_cluster_id",
    "euk_group",
    "source_group",
]


def write_tsv(path, rows, fields=FIELDS):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sequence(index, length=40):
    alphabet = "ACGT"
    token = ""
    value = index
    for _ in range(8):
        token += alphabet[value % 4]
        value //= 4
    return (token + "ACGTGCTA" * 8)[:length]


def fasta(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record_id, seq in records:
            handle.write(f">{record_id}\n{seq}\n")


class V231PipelineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.base = self.root / "base"
        self.base_rows = []
        counter = 1
        class_file = {
            "bacteria": "bacteria.fasta",
            "archaea": "archaea.fasta",
            "eukarya": "eukarya.fasta",
            "mitochondria": "mitochondria.fasta",
            "plastids": "plastids.fasta",
        }
        for split in ("train", "validation"):
            for legacy_class in class_file:
                accession = f"BASE_{split[:1].upper()}_{legacy_class}"
                record = accession + "|0"
                fasta(
                    self.base / split / class_file[legacy_class],
                    [(record, sequence(counter))],
                )
                row = {
                    "record_id": record,
                    "accession": accession,
                    "split": split,
                    "legacy_class": legacy_class,
                    "species_taxid": str(1000 + counter),
                    "split_group_id": f"base-sg-{counter}",
                    "duplicate_cluster_id": f"base-dc-{counter}",
                    # Exercise recovery of a missing Euk label through source metadata.
                    "euk_group": "" if legacy_class == "eukarya" else "",
                    "source_group": "",
                }
                self.base_rows.append(row)
                counter += 1
        self.base_metadata = self.root / "base_metadata.tsv"
        write_tsv(self.base_metadata, self.base_rows)

        source_rows = []
        for row in self.base_rows:
            if row["legacy_class"] == "eukarya":
                source_rows.append({**row, "euk_group": "fungi", "source_group": "fungi"})

        self.candidate_root = self.root / "candidate"
        self.candidate_rows = []
        for split_index, split in enumerate(("train", "validation")):
            records = []
            for group_index, group in enumerate(EUK[1:], start=1):
                accession = f"CAND_{split}_{group}"
                record = accession + "|0"
                seq_index = 100 + split_index * 20 + group_index
                records.append((record, sequence(seq_index)))
                row = {
                    "record_id": record,
                    "accession": accession,
                    "split": split,
                    "legacy_class": "eukarya",
                    "species_taxid": str(10_000 + seq_index),
                    "split_group_id": f"cand-sg-{seq_index}",
                    "duplicate_cluster_id": f"cand-dc-{seq_index}",
                    "euk_group": group,
                    "source_group": "",
                }
                self.candidate_rows.append(row)
                source_rows.append(row)
            fasta(self.candidate_root / split / "eukarya.fasta", records)

        # One sequence duplicate to the frozen base: it must be rejected.
        duplicate_record = "CAND_DUPSEQ|0"
        with (self.candidate_root / "train" / "eukarya.fasta").open("a") as handle:
            handle.write(f">{duplicate_record}\n{sequence(3)}\n")
        duplicate_row = {
            "record_id": duplicate_record,
            "accession": "CAND_DUPSEQ",
            "split": "train",
            "legacy_class": "eukarya",
            "species_taxid": "29001",
            "split_group_id": "cand-sg-dup",
            "duplicate_cluster_id": "cand-dc-dup",
            "euk_group": "algae",
            "source_group": "",
        }
        self.candidate_rows.append(duplicate_row)
        source_rows.append(duplicate_row)

        self.source_index = self.root / "source.tsv"
        write_tsv(self.source_index, source_rows)
        self.pool_metadata = self.root / "pool.tsv"
        write_tsv(self.pool_metadata, self.candidate_rows)
        self.candidate_metadata = self.root / "candidate.tsv"
        write_tsv(self.candidate_metadata, self.candidate_rows)

        self.tfidf = self.root / "tfidf"
        self.tfidf.mkdir()
        np.save(self.tfidf / "model.npy", np.ones(4 ** 7, dtype=np.float32))
        (self.tfidf / "params.txt").write_text(
            "k:7\nfragment_len:5000\nverbose:False\nsmooth:True\nN:1\ndata_names:test\n"
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_audit_freeze_delta_features_and_evaluation(self):
        audit_dir = self.root / "audit"
        audit = audit_euk_completeness(
            str(self.base_metadata),
            str(audit_dir),
            source_index=str(self.source_index),
            pool_metadata=str(self.pool_metadata),
        )
        self.assertEqual(audit["rows"]["unresolved_euk"], 0)
        self.assertEqual(audit["base_by_split"]["train"]["fungi"], 1)
        self.assertIn("alveolata", audit["missing_by_split"]["train"])
        self.assertEqual(audit["label_actions"]["label_error_missing"], 2)

        freeze_path = self.root / "base_freeze.json"
        freeze_base(
            str(self.base),
            str(self.tfidf),
            str(self.base_metadata),
            str(freeze_path),
        )
        self.assertTrue(verify_freeze(str(freeze_path), "full")["ok"])

        delta_dir = self.root / "delta"
        delta = build_euk_delta(
            str(self.base),
            audit["outputs"]["curated_metadata"],
            str(self.candidate_root),
            str(self.candidate_metadata),
            str(delta_dir),
            audit_report=audit["outputs"]["report"],
            base_freeze_manifest=str(freeze_path),
            verify_base="full",
        )
        self.assertTrue(delta["ready"])
        self.assertGreaterEqual(
            delta["rejected"].get("duplicate_sequence_to_base_or_delta", 0), 1
        )
        for split in ("train", "validation"):
            for group in EUK[1:]:
                self.assertEqual(delta["accepted"][split][group], 1)

        leakage = audit_combined_leakage(
            audit["outputs"]["curated_metadata"],
            str(delta_dir / "euk_delta_metadata.tsv"),
        )
        self.assertTrue(leakage["ok"])

        # Build the original single-shard cache, then verify v2.3.1 reuses it
        # and only featurizes delta rows.
        try:
            from tiara.hierarchical.data import prepare
            from tiara.hierarchical.features_v231 import prepare_v231_features

            base_features = self.root / "base_features"
            prepare(
                str(self.base),
                str(base_features),
                str(self.tfidf),
                "2.3.0",
                audit["outputs"]["curated_metadata"],
            )
            composite_dir = self.root / "composite"
            composite = prepare_v231_features(
                str(base_features),
                str(self.base),
                audit["outputs"]["curated_metadata"],
                str(delta_dir),
                str(delta_dir / "euk_delta_metadata.tsv"),
                str(self.tfidf),
                str(composite_dir),
                base_freeze_manifest=str(freeze_path),
                verify_base="full",
                chunk=2,
            )
            self.assertTrue(composite["ready"])
            for split in ("train", "validation"):
                self.assertTrue(
                    all(composite["combined_euk_counts"][split][g] > 0 for g in EUK)
                )
                self.assertEqual(len(composite["splits"][split]["shards"]), 2)
        except ModuleNotFoundError as exc:
            # The bootstrap deliberately does not install the heavy Tiara ML
            # environment. Audit/delta/evaluation still run in base Python;
            # feature integration is exercised when tqdm/Bio/numba are present.
            self.assertIn(exc.name, {"tqdm", "Bio", "numba"})

        truth_rows = []
        prediction_rows = []
        with Path(audit["outputs"]["curated_metadata"]).open(newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if row["legacy_class"] != "eukarya":
                    continue
                truth_rows.append(row)
                prediction_rows.append(
                    {
                        "record_id": row["record_id"],
                        "root": "euk_nuclear",
                        "leaf": row["euk_group"],
                    }
                )
        with (delta_dir / "euk_delta_metadata.tsv").open(newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                truth_rows.append(row)
                prediction_rows.append(
                    {"record_id": row["record_id"], "root": "euk_nuclear", "leaf": row["euk_group"]}
                )
        truth_path = self.root / "truth.tsv"
        write_tsv(truth_path, truth_rows)
        prediction_path = self.root / "pred.tsv"
        write_tsv(prediction_path, prediction_rows, ["record_id", "root", "leaf"])
        metrics = evaluate(str(truth_path), str(prediction_path))
        self.assertEqual(metrics["euk"]["macro_f1"], 1.0)
        self.assertEqual(metrics["cascade"]["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

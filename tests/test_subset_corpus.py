#!/usr/bin/env python3
"""Tests for scripts/subset_corpus_by_plan.py.

The behaviour that matters most here is class routing: an organelle record is
tagged ``label=euk sg=Organelle-Mito`` because its host is a eukaryote, so the
compartment in ``sg`` -- not ``label``, and not the file it happens to sit in
-- decides the class.  Getting that wrong silently empties the mitochondria and
plastids classes, which is exactly the failure this script exists to avoid
repeating.
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
import contextlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for _p in (str(REPO / "scripts"), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import subset_corpus_by_plan as sub  # noqa: E402


def fasta(records):
    """records = [(frag_id, sg, label, epoch, seq)] -> FASTA text."""
    out = []
    for frag_id, sg, label, epoch, seq in records:
        out.append(f">{frag_id} sg={sg} label={label} epoch={epoch}")
        out.append(seq)
    return "\n".join(out) + "\n"


SELECTION_HEADER = "assembly_accession\torganism_name\tclade\ttarget_bp\tanchor"


def selection(rows):
    lines = [SELECTION_HEADER]
    for acc, organism, clade, target, anchor in rows:
        lines.append(f"{acc}\t{organism}\t{clade}\t{target}\t{anchor}")
    return "\n".join(lines) + "\n"


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.src = self.tmp / "train_ready_old"
        self.dest = self.tmp / "train_ready_new"
        (self.src / "train").mkdir(parents=True)
        (self.src / "test").mkdir(parents=True)

    def write_corpus(self, split, klass, records):
        (self.src / split / f"{klass}.fasta").write_text(fasta(records))

    def write_selection(self, rows):
        path = self.tmp / "curation_selection.tsv"
        path.write_text(selection(rows))
        return path

    def write_anchors(self, accs):
        path = self.tmp / "anchors.txt"
        path.write_text("\n".join(accs) + "\n")
        return path

    def run_main(self, *extra, expect=0):
        argv = [
            "--source-root", str(self.src),
            "--dest-root", str(self.dest),
            "--selection", str(self.tmp / "curation_selection.tsv"),
            "--report", str(self.tmp / "report.json"),
            "--allow-same-device", "--quiet",
        ] + list(extra)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            code = sub.main(argv)
        self.assertEqual(code, expect, buf.getvalue())
        return buf.getvalue()

    def report(self):
        return json.loads((self.tmp / "report.json").read_text())

    def out(self, split, klass):
        path = self.dest / split / f"{klass}.fasta"
        return path.read_text() if path.exists() else ""


class TestClassRouting(Base):
    def test_organelle_labelled_euk_routes_by_sg(self):
        """label=euk sg=Organelle-Mito must land in mitochondria, not eukarya."""
        self.write_corpus("train", "eukarya", [
            ("GCA_000001.1|0", "Organelle-Mito", "euk", "train", "A" * 100),
            ("GCA_000001.1|1", "Organelle-Plastid", "euk", "train", "C" * 100),
            ("GCA_000001.1|2", "Opisthokonta-Fungi", "euk", "train", "G" * 100),
        ])
        self.write_selection([("GCA_000001.1", "Some fungus", "fungi", "", "0")])
        self.run_main()
        self.assertIn("GCA_000001.1|0", self.out("train", "mitochondria"))
        self.assertIn("GCA_000001.1|1", self.out("train", "plastids"))
        self.assertIn("GCA_000001.1|2", self.out("train", "eukarya"))
        self.assertNotIn("GCA_000001.1|0", self.out("train", "eukarya"))
        self.assertEqual(self.report()["class_mismatch"], 2)

    def test_archaeplastida_is_not_a_plastid(self):
        """The substring trap: 'plastid' in 'archaeplastida' is True."""
        self.write_corpus("train", "eukarya", [
            ("GCA_000002.1|0", "Archaeplastida", "euk", "train", "A" * 100),
            ("GCA_000002.1|1", "Kinetoplastida", "euk", "train", "T" * 100),
        ])
        self.write_selection([("GCA_000002.1", "A plant", "algae", "", "0")])
        self.run_main()
        self.assertEqual(self.out("train", "plastids"), "")
        text = self.out("train", "eukarya")
        self.assertIn("GCA_000002.1|0", text)
        self.assertIn("GCA_000002.1|1", text)
        self.assertEqual(self.report()["class_mismatch"], 0)

    def test_hint_policy_trusts_the_file_name(self):
        self.write_corpus("train", "eukarya", [
            ("GCA_000001.1|0", "Organelle-Mito", "euk", "train", "A" * 100),
        ])
        self.write_selection([("GCA_000001.1", "x", "fungi", "", "0")])
        self.run_main("--on-class-mismatch", "hint")
        self.assertIn("GCA_000001.1|0", self.out("train", "eukarya"))
        self.assertEqual(self.out("train", "mitochondria"), "")

    def test_fail_policy_stops_on_disagreement(self):
        self.write_corpus("train", "eukarya", [
            ("GCA_000001.1|0", "Organelle-Mito", "euk", "train", "A" * 100),
        ])
        self.write_selection([("GCA_000001.1", "x", "fungi", "", "0")])
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stdout(io.StringIO()):
                sub.main([
                    "--source-root", str(self.src),
                    "--dest-root", str(self.dest),
                    "--selection", str(self.tmp / "curation_selection.tsv"),
                    "--on-class-mismatch", "fail",
                    "--allow-same-device", "--quiet",
                ])

    def test_prokaryotes_keep_their_class(self):
        self.write_corpus("train", "archaea", [
            ("GCA_000003.1|0", "Archaea", "prok", "train", "A" * 100),
        ])
        self.write_corpus("train", "bacteria", [
            ("GCA_000004.1|0", "Bacteria", "prok", "train", "C" * 100),
        ])
        self.write_selection([
            ("GCA_000003.1", "an archaeon", "", "", "0"),
            ("GCA_000004.1", "a bacterium", "", "", "0"),
        ])
        self.run_main()
        self.assertIn("GCA_000003.1|0", self.out("train", "archaea"))
        self.assertIn("GCA_000004.1|0", self.out("train", "bacteria"))
        self.assertEqual(self.report()["class_mismatch"], 0)


class TestPlanFiltering(Base):
    def test_unselected_accessions_are_dropped(self):
        self.write_corpus("train", "bacteria", [
            ("GCA_000010.1|0", "Bacteria", "prok", "train", "A" * 100),
            ("GCA_000011.1|0", "Bacteria", "prok", "train", "C" * 100),
        ])
        self.write_selection([("GCA_000010.1", "keep me", "", "", "0")])
        self.run_main()
        text = self.out("train", "bacteria")
        self.assertIn("GCA_000010.1|0", text)
        self.assertNotIn("GCA_000011.1|0", text)
        self.assertEqual(self.report()["not_in_plan"], 1)

    def test_target_bp_caps_each_genome(self):
        records = [(f"GCA_000020.1|{i}", "Opisthokonta-Fungi", "euk", "train",
                    "A" * 100) for i in range(10)]
        self.write_corpus("train", "eukarya", records)
        self.write_selection([("GCA_000020.1", "deep genome", "fungi", "250", "0")])
        self.run_main()
        kept = self.out("train", "eukarya").count(">")
        self.assertEqual(kept, 3)  # 100+100+100 >= 250 stops after the third
        rep = self.report()
        self.assertEqual(rep["over_budget"], 7)
        self.assertEqual(rep["bp_kept"], 300)

    def test_uncapped_when_target_bp_blank(self):
        records = [(f"GCA_000021.1|{i}", "Bacteria", "prok", "train", "A" * 50)
                   for i in range(5)]
        self.write_corpus("train", "bacteria", records)
        self.write_selection([("GCA_000021.1", "x", "", "", "0")])
        self.run_main()
        self.assertEqual(self.out("train", "bacteria").count(">"), 5)

    def test_default_target_bp_applies_to_blank_rows(self):
        records = [(f"GCA_000022.1|{i}", "Bacteria", "prok", "train", "A" * 50)
                   for i in range(5)]
        self.write_corpus("train", "bacteria", records)
        self.write_selection([("GCA_000022.1", "x", "", "", "0")])
        self.run_main("--default-target-bp", "100")
        self.assertEqual(self.out("train", "bacteria").count(">"), 2)

    def test_version_drift_still_matches(self):
        self.write_corpus("train", "bacteria", [
            ("GCA_000030.1|0", "Bacteria", "prok", "train", "A" * 100),
        ])
        self.write_selection([("GCA_000030.2", "newer version", "", "", "0")])
        self.run_main()
        self.assertIn("GCA_000030.1|0", self.out("train", "bacteria"))
        self.assertEqual(self.report()["matched_other_version"], 1)

    def test_splits_are_preserved_independently(self):
        self.write_corpus("train", "bacteria", [
            ("GCA_000040.1|0", "Bacteria", "prok", "train", "A" * 100)])
        self.write_corpus("test", "bacteria", [
            ("GCA_000041.1|0", "Bacteria", "prok", "test", "C" * 100)])
        self.write_selection([
            ("GCA_000040.1", "a", "", "", "0"),
            ("GCA_000041.1", "b", "", "", "0"),
        ])
        self.run_main()
        self.assertIn("GCA_000040.1|0", self.out("train", "bacteria"))
        self.assertIn("GCA_000041.1|0", self.out("test", "bacteria"))

    def test_split_subset_option(self):
        self.write_corpus("train", "bacteria", [
            ("GCA_000040.1|0", "Bacteria", "prok", "train", "A" * 100)])
        self.write_corpus("test", "bacteria", [
            ("GCA_000041.1|0", "Bacteria", "prok", "test", "C" * 100)])
        self.write_selection([
            ("GCA_000040.1", "a", "", "", "0"),
            ("GCA_000041.1", "b", "", "", "0"),
        ])
        self.run_main("--splits", "train")
        self.assertEqual(self.out("test", "bacteria"), "")
        self.assertIn("GCA_000040.1|0", self.out("train", "bacteria"))


class TestReportingAndGuards(Base):
    def test_missing_accessions_are_listed(self):
        self.write_corpus("train", "bacteria", [
            ("GCA_000050.1|0", "Bacteria", "prok", "train", "A" * 100)])
        self.write_selection([
            ("GCA_000050.1", "present", "", "", "0"),
            ("GCA_000051.1", "absent", "", "1000", "0"),
        ])
        self.run_main()
        rep = self.report()
        self.assertEqual(rep["accessions_missing"], 1)
        listing = (self.dest / "missing_from_source_corpus.tsv").read_text()
        self.assertIn("GCA_000051.1", listing)
        self.assertIn("1000", listing)

    def test_missing_anchor_is_a_hard_failure(self):
        self.write_corpus("train", "eukarya", [
            ("GCA_000060.1|0", "Opisthokonta-Fungi", "euk", "train", "A" * 100)])
        self.write_selection([
            ("GCA_000060.1", "present", "fungi", "", "0"),
            ("GCA_000061.1", "missing anchor", "algae", "", "1"),
        ])
        anchors = self.write_anchors(["GCA_000061.1"])
        text = self.run_main("--anchors", str(anchors), expect=2)
        self.assertIn("GCA_000061.1", text)
        self.assertEqual(self.report()["anchors_missing"], 1)

    def test_missing_anchor_can_be_overridden(self):
        self.write_corpus("train", "eukarya", [
            ("GCA_000060.1|0", "Opisthokonta-Fungi", "euk", "train", "A" * 100)])
        self.write_selection([
            ("GCA_000060.1", "present", "fungi", "", "0"),
            ("GCA_000061.1", "missing anchor", "algae", "", "1"),
        ])
        anchors = self.write_anchors(["GCA_000061.1"])
        self.run_main("--anchors", str(anchors), "--allow-missing-anchors")

    def test_present_anchor_passes(self):
        self.write_corpus("train", "eukarya", [
            ("GCA_000060.1|0", "Opisthokonta-Fungi", "euk", "train", "A" * 100)])
        self.write_selection([("GCA_000060.1", "present", "fungi", "", "1")])
        anchors = self.write_anchors(["GCA_000060.1"])
        self.run_main("--anchors", str(anchors))
        self.assertEqual(self.report()["anchors_missing"], 0)

    def test_shortfall_is_reported(self):
        self.write_corpus("train", "eukarya", [
            ("GCA_000070.1|0", "Opisthokonta-Fungi", "euk", "train", "A" * 100)])
        self.write_selection([("GCA_000070.1", "thin", "fungi", "5000", "0")])
        self.run_main()
        rep = self.report()
        self.assertEqual(rep["accessions_short"], 1)
        listing = (self.dest / "shortfall_vs_target_bp.tsv").read_text()
        self.assertIn("GCA_000070.1", listing)
        self.assertIn("4900", listing)  # deficit

    def test_dry_run_writes_no_fasta(self):
        self.write_corpus("train", "bacteria", [
            ("GCA_000080.1|0", "Bacteria", "prok", "train", "A" * 100)])
        self.write_selection([("GCA_000080.1", "x", "", "", "0")])
        text = self.run_main("--dry-run")
        self.assertIn("DRY RUN", text)
        self.assertFalse(list(self.dest.glob("*/*.fasta")))
        self.assertEqual(self.report()["records_kept"], 1)

    def test_nothing_matched_exits_two(self):
        self.write_corpus("train", "bacteria", [
            ("GCA_000090.1|0", "Bacteria", "prok", "train", "A" * 100)])
        self.write_selection([("GCA_999999.1", "other corpus", "", "", "0")])
        text = self.run_main(expect=2)
        self.assertIn("nothing was kept", text)

    def test_refuses_destination_inside_source(self):
        self.write_corpus("train", "bacteria", [
            ("GCA_000100.1|0", "Bacteria", "prok", "train", "A" * 100)])
        path = self.write_selection([("GCA_000100.1", "x", "", "", "0")])
        with self.assertRaises(SystemExit):
            sub.main(["--source-root", str(self.src / "train"),
                      "--dest-root", str(self.src),
                      "--selection", str(path),
                      "--allow-same-device", "--quiet"])

    def test_same_device_requires_a_flag(self):
        self.write_corpus("train", "bacteria", [
            ("GCA_000110.1|0", "Bacteria", "prok", "train", "A" * 100)])
        path = self.write_selection([("GCA_000110.1", "x", "", "", "0")])
        with self.assertRaises(SystemExit) as ctx:
            sub.main(["--source-root", str(self.src),
                      "--dest-root", str(self.dest),
                      "--selection", str(path), "--quiet"])
        self.assertIn("same device", str(ctx.exception))

    def test_existing_destination_needs_force(self):
        self.write_corpus("train", "bacteria", [
            ("GCA_000120.1|0", "Bacteria", "prok", "train", "A" * 100)])
        self.write_selection([("GCA_000120.1", "x", "", "", "0")])
        self.run_main()
        with self.assertRaises(SystemExit) as ctx:
            sub.main(["--source-root", str(self.src),
                      "--dest-root", str(self.dest),
                      "--selection", str(self.tmp / "curation_selection.tsv"),
                      "--allow-same-device", "--quiet"])
        self.assertIn("--force", str(ctx.exception))
        self.run_main("--force")

    def test_empty_source_is_an_error(self):
        self.write_selection([("GCA_000130.1", "x", "", "", "0")])
        with self.assertRaises(SystemExit) as ctx:
            sub.main(["--source-root", str(self.src),
                      "--dest-root", str(self.dest),
                      "--selection", str(self.tmp / "curation_selection.tsv"),
                      "--allow-same-device", "--quiet"])
        self.assertIn("no <split>/<class>.fasta", str(ctx.exception))


class TestHelpers(unittest.TestCase):
    def test_base_accession(self):
        self.assertEqual(sub.base_accession("GCA_000411095.2"), "GCA_000411095")
        self.assertEqual(sub.base_accession("GCA_000411095"), "GCA_000411095")
        self.assertEqual(sub.base_accession(""), "")

    def test_as_int(self):
        self.assertEqual(sub.as_int("1200"), 1200)
        self.assertEqual(sub.as_int("1.2e3"), 1200)
        self.assertIsNone(sub.as_int(""))
        self.assertIsNone(sub.as_int("not a number"))


if __name__ == "__main__":
    unittest.main(verbosity=2)

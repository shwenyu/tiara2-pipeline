#!/usr/bin/env python3
"""Tests for scripts/merge_download_plans.py (the three-plan join).

The failure modes worth pinning down are all silent ones: dropping an anchor,
losing the per-genome target_bp cap, matching only exact accession versions, and
producing an empty manifest from two plans that belong to different rounds.

Run directly:  python3 tests/test_merge_plans.py
"""
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import merge_download_plans as M  # noqa: E402

SCRIPT = ROOT / "scripts" / "merge_download_plans.py"

DL_COLUMNS = ["assembly_accession", "organism_name", "clade", "ftp_path"]
DL_ROWS = [
    ["GCA_000001.1", "Genus_a sp", "fungi", "ftp://x/1"],
    ["GCA_000002.1", "Genus_a other", "fungi", "ftp://x/2"],
    ["GCA_000003.2", "Genus_b sp", "algae", "ftp://x/3"],
    ["GCA_000004.1", "Genus_c sp", "alveolata", "ftp://x/4"],
]
CUR_COLUMNS = ["assembly_accession", "organism_name", "target_bp",
               "target_fragments", "anchor", "ftp_path"]
CUR_ROWS = [
    # one genome per genus: a and b, not the second Genus_a
    ["GCA_000001.1", "Genus_a sp", "23700000", "4740", "0", "ftp://x/1"],
    # version drift on purpose: curate says .1, the manifest has .2
    ["GCA_000003.1", "Genus_b sp", "23700000", "4740", "1", "ftp://x/3"],
    # selected but absent from the manifest
    ["GCA_000009.1", "Genus_z sp", "1200000", "240", "0", "ftp://x/9"],
]
T1_COLUMNS = ["accession", "organism_name", "fine_class", "status", "is_anchor"]
T1_ROWS = [
    ["GCA_000003.1", "Genus_b sp", "eukarya", "already_held", "1"],
    ["GCA_000004.1", "Genus_c sp", "eukarya", "already_held", "1"],
    ["GCA_000002.1", "Genus_a other", "mitochondria", "new", "0"],
]


def write(path, columns, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(columns)
        w.writerows(rows)
    return path


class Fixture:
    def __init__(self, tmp, dl_rows=None, cur_rows=None, t1_rows=None,
                 anchors=("GCA_000003.1", "GCA_000004.1")):
        self.dir = Path(tmp)
        self.downloads = write(self.dir / "download_all.tsv", DL_COLUMNS,
                               DL_ROWS if dl_rows is None else dl_rows)
        self.curation = write(self.dir / "curation_selection.tsv", CUR_COLUMNS,
                              CUR_ROWS if cur_rows is None else cur_rows)
        self.tiara1 = write(self.dir / "tiara1_selection.tsv", T1_COLUMNS,
                            T1_ROWS if t1_rows is None else t1_rows)
        self.anchors = self.dir / "anchors.txt"
        self.anchors.write_text("# comment\n" + "\n".join(anchors) + "\n")
        self.out = self.dir / "download_curated.tsv"
        self.report = self.dir / "report.json"

    def run(self, *extra):
        return subprocess.run(
            [sys.executable, str(SCRIPT),
             "--downloads", str(self.downloads),
             "--curation", str(self.curation),
             "--tiara1", str(self.tiara1),
             "--anchors", str(self.anchors),
             "--out", str(self.out),
             "--report", str(self.report), *extra],
            capture_output=True, text=True)

    def rows(self):
        with open(self.out, newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh, delimiter="\t"))


class TestHelpers(unittest.TestCase):
    def test_base_accession(self):
        self.assertEqual(M.base_accession("GCA_000008085.1"), "GCA_000008085")
        self.assertEqual(M.base_accession(""), "")

    def test_pick_column_is_case_and_hash_insensitive(self):
        self.assertEqual(
            M.pick_column(["#Assembly_Accession", "x"], M.ACCESSION_KEYS),
            "#Assembly_Accession")

    def test_anchor_file_ignores_comments(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "a.txt"
            p.write_text("# header\n\nGCA_1.1\nGCA_2.1  trailing\n")
            self.assertEqual(M.read_anchors(p), ["GCA_1.1", "GCA_2.1"])

    def test_tiara1_required_keeps_anchors_regardless_of_status(self):
        rows = [{"status": "already_held", "is_anchor": "1"},
                {"status": "already_held", "is_anchor": "0"},
                {"status": "new", "is_anchor": "0"}]
        kept = M.tiara1_required(rows, ["new"])
        self.assertEqual(len(kept), 2)
        self.assertNotIn({"status": "already_held", "is_anchor": "0"}, kept)


class TestMerge(unittest.TestCase):
    def test_happy_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(tmp)
            res = fx.run()
            self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
            rows = fx.rows()
            accs = {r["assembly_accession"] for r in rows}
            # curate picked 1 and 3; tiara1 forces 4 (anchor) and 2 (new)
            self.assertEqual(accs, {"GCA_000001.1", "GCA_000003.2",
                                    "GCA_000004.1", "GCA_000002.1"})

    def test_target_bp_survives_the_join(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(tmp)
            fx.run()
            by_acc = {r["assembly_accession"]: r for r in fx.rows()}
            self.assertEqual(by_acc["GCA_000001.1"]["target_bp"], "23700000")
            self.assertEqual(by_acc["GCA_000001.1"]["plan_source"], "curate")

    def test_version_drift_matches_on_base_accession(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(tmp)
            fx.run()
            by_acc = {r["assembly_accession"]: r for r in fx.rows()}
            self.assertIn("GCA_000003.2", by_acc)
            self.assertEqual(by_acc["GCA_000003.2"]["plan_source"],
                             "curate_other_version")

    def test_original_columns_are_preserved_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(tmp)
            fx.run()
            with open(fx.out, encoding="utf-8") as fh:
                header = fh.readline().rstrip("\n").split("\t")
            self.assertEqual(header[:len(DL_COLUMNS)], DL_COLUMNS)
            for col in M.APPENDED:
                self.assertIn(col, header)

    def test_selected_but_undownloadable_is_reported_not_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(tmp)
            fx.run()
            miss = fx.dir / "missing_from_downloads.tsv"
            self.assertTrue(miss.is_file())
            text = miss.read_text(encoding="utf-8")
            self.assertIn("GCA_000009.1", text)
            self.assertIn("NOT_IN_DOWNLOAD_MANIFEST", text)

    def test_report_json_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(tmp)
            fx.run()
            rep = json.loads(fx.report.read_text(encoding="utf-8"))
            self.assertEqual(rep["output_rows"], 4)
            self.assertEqual(rep["counts"]["curate_exact"], 1)
            self.assertEqual(rep["counts"]["curate_base"], 1)
            self.assertEqual(rep["counts"]["curate_missing"], 1)
            self.assertEqual(rep["anchors_missing"], [])

    def test_no_duplicate_rows_when_both_plans_want_the_same_genome(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(tmp)
            fx.run()
            accs = [r["assembly_accession"] for r in fx.rows()]
            self.assertEqual(len(accs), len(set(accs)))


class TestGates(unittest.TestCase):
    def test_missing_anchor_exits_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(tmp, anchors=("GCA_000003.1", "GCA_999999.1"))
            res = fx.run()
            self.assertEqual(res.returncode, 2)
            self.assertIn("anchor", res.stderr.lower())

    def test_missing_anchor_can_be_overridden(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(tmp, anchors=("GCA_999999.1",))
            res = fx.run("--allow-missing-anchors")
            self.assertEqual(res.returncode, 0, res.stdout + res.stderr)

    def test_disjoint_plans_exit_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(
                tmp,
                cur_rows=[["GCA_777777.1", "Other sp", "1", "1", "0", "ftp://x"]],
                t1_rows=[], anchors=())
            res = fx.run()
            self.assertEqual(res.returncode, 2)
            self.assertIn("nothing matched", res.stderr.lower())

    def test_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(tmp)
            res = fx.run("--dry-run")
            self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
            self.assertFalse(fx.out.exists())
            self.assertIn("dry run", res.stdout)


class TestProgressPlumbing(unittest.TestCase):
    def test_progress_goes_to_stderr_and_report_to_stdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(tmp)
            res = fx.run("--progress-seconds", "0")
            self.assertIn("[merge]", res.stderr)
            self.assertNotIn("[merge]", res.stdout)
            self.assertIn("OUTPUT rows", res.stdout)

    def test_quiet_silences_stderr(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(tmp)
            res = fx.run("--quiet")
            self.assertEqual(res.stderr.strip(), "")


if __name__ == "__main__":
    unittest.main(verbosity=2)

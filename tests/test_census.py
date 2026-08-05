"""census: per-class data VOLUME, and the live-download guard.

Why this is separate from inventory's byte report
-------------------------------------------------
``inventory.corpus_inventory`` answers "how many bytes" in 15 stat() calls.
The curation and subsampling decisions are denominated in FRAGMENTS, and
bytes -> fragments runs through record count and sequence length. These tests
pin the conversion, the extrapolation from a head sample, and -- most
importantly -- that a download in flight is detected, because a census taken
mid-download is a moving target.

Invariants protected here:
  * exact mode counts records and bp correctly, including multi-line FASTA;
  * fast mode extrapolates within a few percent and SAYS it sampled;
  * a missing file is reported, never raised;
  * the fragment projection reproduces the configured class shares and the
    delta against the Tiara S3 priors;
  * partial files, fresh mtimes and live acquire logs each independently mark
    the corpus as active;
  * a settled corpus is NOT reported as active (no false alarms).
"""
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tiara2 import census as CEN  # noqa: E402


def write_fasta(path, n_records, seq_len, line_width=60):
    """Realistic multi-line FASTA: real files wrap, and the wrapping newlines
    must not be counted as base pairs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    seq = "ACGT" * (seq_len // 4) + "A" * (seq_len % 4)
    with open(path, "w") as fh:
        for i in range(n_records):
            fh.write(f">rec_{i} some description here\n")
            for j in range(0, len(seq), line_width):
                fh.write(seq[j:j + line_width] + "\n")
    return path


def make_cfg(base, **kw):
    cfg = {
        "base": str(base), "data_home": str(base),
        "source_ready": str(base / "src"),
        "work_root": str(base / "work"),
        "splits": ["train"],
        "classes": ["bacteria", "archaea", "eukarya", "mitochondria",
                    "plastids"],
        "curate": {"budget": {"mean_fragment_bp": 4200,
                              "census_split": "train"},
                   "target_priors": dict(CEN.TIARA_S3_PRIORS)},
        "census": {"sample_mb": 1, "active_window_s": 900, "watch_dirs": []},
    }
    cfg.update(kw)
    return cfg


class TestSampleFasta(unittest.TestCase):
    def test_exact_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = write_fasta(Path(tmp) / "a.fasta", 100, 1000)
            out = CEN.sample_fasta(p)          # no limit == exact
            self.assertEqual(out["mode"], "exact")
            self.assertEqual(out["records"], 100)
            self.assertEqual(out["bp"], 100_000)
            self.assertAlmostEqual(out["mean_len"], 1000.0)

    def test_newlines_are_not_base_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = write_fasta(Path(tmp) / "a.fasta", 10, 300, line_width=10)
            self.assertEqual(CEN.sample_fasta(p)["bp"], 3000)

    def test_fast_mode_extrapolates(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = write_fasta(Path(tmp) / "a.fasta", 4000, 1000)
            exact = CEN.sample_fasta(p)
            fast = CEN.sample_fasta(p, 256 * 1024)
            self.assertEqual(fast["mode"], "fast")
            self.assertLess(fast["sampled_bytes"], fast["total_bytes"])
            err = abs(fast["bp"] - exact["bp"]) / exact["bp"]
            self.assertLess(err, 0.05, f"extrapolation off by {err:.1%}")

    def test_small_file_is_exact_even_in_fast_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = write_fasta(Path(tmp) / "a.fasta", 3, 100)
            self.assertEqual(CEN.sample_fasta(p, 10 * 1024 * 1024)["mode"],
                             "exact")

    def test_missing_file_never_raises(self):
        out = CEN.sample_fasta("/nonexistent/nope.fasta")
        self.assertFalse(out["exists"])
        self.assertEqual(out["records"], 0)

    def test_empty_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "e.fasta"
            p.write_text("")
            out = CEN.sample_fasta(p)
            self.assertTrue(out["exists"])
            self.assertEqual(out["bp"], 0)


class TestClassCensus(unittest.TestCase):
    def test_covers_every_split_and_class(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_cfg(Path(tmp))
            out = CEN.class_census(cfg, exact=True)
            self.assertEqual(len(out["rows"]), 5)
            self.assertTrue(all(not r["exists"] for r in out["rows"]))

    def test_half_downloaded_tree_still_reports(self):
        """Exactly the situation right now: some classes present, some not."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            write_fasta(base / "src" / "train" / "bacteria.fasta", 50, 1000)
            out = CEN.class_census(make_cfg(base), exact=True)
            present = [r for r in out["rows"] if r["exists"]]
            self.assertEqual(len(present), 1)
            self.assertEqual(present[0]["class"], "bacteria")
            self.assertEqual(present[0]["records"], 50)


class TestProjection(unittest.TestCase):
    def _projection(self, base):
        cfg = make_cfg(base)
        return CEN.fragment_projection(CEN.class_census(cfg, exact=True), cfg)

    def test_fragments_from_bp(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            write_fasta(base / "src" / "train" / "bacteria.fasta", 100, 42_000)
            proj = self._projection(base)
            # 100 x 42 kb / 4.2 kb per fragment == 1000
            self.assertEqual(proj["total_fragments"], 1000)

    def test_delta_against_target_prior(self):
        """A single-class corpus is 100% of itself and must show the gap."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            write_fasta(base / "src" / "train" / "eukarya.fasta", 100, 42_000)
            proj = self._projection(base)
            euk = [c for c in proj["classes"] if c["class"] == "eukarya"][0]
            self.assertAlmostEqual(euk["share"], 1.0)
            self.assertAlmostEqual(euk["target_share"], 0.31667691, places=6)
            self.assertGreater(euk["delta_pp"], 68.0)
            self.assertLess(euk["fragments_to_target"], 0)  # must SHED

    def test_underweight_class_asks_for_more(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "src" / "train"
            write_fasta(root / "eukarya.fasta", 100, 42_000)
            write_fasta(root / "archaea.fasta", 1, 42_000)
            proj = self._projection(base)
            arc = [c for c in proj["classes"] if c["class"] == "archaea"][0]
            self.assertGreater(arc["fragments_to_target"], 0)


class TestDownloadGuard(unittest.TestCase):
    def test_partial_file_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            d = base / "src" / "train"
            d.mkdir(parents=True)
            (d / "eukarya.fasta.part").write_text("x")
            act = CEN.download_activity(make_cfg(base))
            self.assertTrue(act["active"])
            self.assertEqual(act["partial_count"], 1)

    def test_every_partial_suffix_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            d = base / "src" / "train"
            d.mkdir(parents=True)
            for suffix in CEN.PARTIAL_SUFFIXES:
                (d / ("f" + suffix)).write_text("x")
            act = CEN.download_activity(make_cfg(base))
            self.assertEqual(act["partial_count"], len(CEN.PARTIAL_SUFFIXES))

    def test_fresh_mtime_is_detected_without_a_partial_suffix(self):
        """A writer that finishes each chunk cleanly leaves no .part behind."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            write_fasta(base / "src" / "train" / "bacteria.fasta", 5, 100)
            act = CEN.download_activity(make_cfg(base))
            self.assertTrue(act["active"])
            self.assertEqual(act["partial_count"], 0)
            self.assertGreaterEqual(act["recent_count"], 1)

    def test_settled_corpus_is_not_flagged(self):
        """No false alarms, or the guard gets disabled and stops protecting."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            p = write_fasta(base / "src" / "train" / "bacteria.fasta", 5, 100)
            old = time.time() - 86400
            os.utime(p, (old, old))
            for d in (p.parent, base / "src"):
                os.utime(d, (old, old))
            act = CEN.download_activity(make_cfg(base), window_s=600)
            self.assertFalse(act["active"], act)

    def test_live_acquire_log_is_detected(self):
        """Catches a download stalled on the network but not finished."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            logs = base / "work" / "acquire" / "logs"
            logs.mkdir(parents=True)
            (logs / "acquire_download.log").write_text("downloading...\n")
            act = CEN.download_activity(make_cfg(base))
            self.assertTrue(act["active"])
            self.assertTrue(act["active_logs"])

    def test_scan_is_capped(self):
        """The guard must never turn into a full walk of a 36 TB tree."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            d = base / "src" / "train"
            d.mkdir(parents=True)
            for i in range(50):
                (d / f"f{i}.txt").write_text("x")
            act = CEN.download_activity(make_cfg(base), max_entries=10)
            self.assertTrue(act["truncated"])


class TestRender(unittest.TestCase):
    def test_render_is_readable_and_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            write_fasta(base / "src" / "train" / "eukarya.fasta", 20, 5000)
            cfg = make_cfg(base)
            snap = CEN.run_census(cfg, exact=True)
            text = CEN.render_census(snap["census"], snap["projection"],
                                     snap["activity"])
            self.assertIn("eukarya", text)
            self.assertIn("MISSING", text)          # undownloaded classes
            self.assertIn("DOWNLOAD IN PROGRESS", text)
            self.assertIn("target", text)

    def test_snapshot_is_json_serialisable(self):
        """`tiara2 census --json` must not blow up on a real snapshot."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            write_fasta(base / "src" / "train" / "eukarya.fasta", 5, 1000)
            snap = CEN.run_census(make_cfg(base), exact=True)
            json.dumps(snap)


class TestCli(unittest.TestCase):
    def test_census_subcommand_runs(self):
        from tiara2 import cli
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            write_fasta(base / "src" / "train" / "eukarya.fasta", 5, 1000)
            out = base / "snap.json"
            rc = cli.main([
                "census", "--config", str(ROOT / "config" / "config.yaml"),
                "--set", f"source_ready={base / 'src'}",
                "--set", f"work_root={base / 'work'}",
                "--exact", "--json", str(out)])
            self.assertEqual(rc, 0)
            self.assertTrue(out.exists())


def _run():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite(
        loader.loadTestsFromTestCase(c) for c in (
            TestSampleFasta, TestClassCensus, TestProjection,
            TestDownloadGuard, TestRender, TestCli))
    res = unittest.TextTestRunner(verbosity=2).run(suite)
    total = res.testsRun
    bad = len(res.failures) + len(res.errors)
    print(f"\n{total - bad}/{total} passed")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    _run()

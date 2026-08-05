"""dedup is an OPTIONAL stage; regroup must stay correct when it is skipped.

The expensive invariant being protected here: with dedup off, the pipeline must
still produce a usable ``corpus_ready`` for training WITHOUT copying the
multi-TB corpus. It mirrors with symlinks instead.
"""
import json
import logging
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tiara2.stages import dedup as dedup_mod  # noqa: E402
from tiara2.stages import regroup as regroup_mod  # noqa: E402

SPLITS = ["train", "validation", "test"]
CLASSES = ["bacteria", "archaea"]


class Ctx:
    """Minimal stand-in for StageContext."""

    def __init__(self, cfg, work_dir):
        self.cfg = cfg
        self.work_dir = Path(work_dir)
        self.log = logging.getLogger("test")
        self.dry_run = False
        self.force = False
        self.debug = False
        self.extra = {}


def make_cfg(base: Path, *, dedup_enabled: bool) -> dict:
    return {
        "base": str(base),
        "source_ready": str(base / "train_ready_c1"),
        "corpus_ready": str(base / "corpus_ready_c1"),
        "results_root": str(base / "results_v1"),
        "work_root": str(base / ".work_c1"),
        "splits": SPLITS,
        "classes": CLASSES,
        "dedup": {"enabled": dedup_enabled, "min_seq_id": 0.95, "min_cov": 0.95},
        "resources": {"threads": 4},
        "split_priority": ["test", "validation", "train"],
    }


def seed_source(cfg) -> int:
    n = 0
    for split in SPLITS:
        d = Path(cfg["source_ready"]) / split
        d.mkdir(parents=True, exist_ok=True)
        for cls in CLASSES:
            (d / f"{cls}.fasta").write_text(f">{split}_{cls}_1\nACGT\n")
            n += 1
    return n


class TestDedupOptional(unittest.TestCase):
    def setUp(self):
        self.base = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.base, ignore_errors=True)

    def test_disabled_dedup_writes_marker_and_runs_no_mmseqs(self):
        cfg = make_cfg(self.base, dedup_enabled=False)
        ctx = Ctx(cfg, self.base / ".work_c1" / "dedup")
        # If it tried to shell out to mmseqs this would raise (no binary here).
        out = dedup_mod.DedupStage().run(ctx)
        self.assertEqual(out["counts"], {"skipped": 1})
        marker = Path(cfg["work_root"]) / "dedup" / ".dedup.done"
        self.assertTrue(marker.exists(), "regroup depends on this marker")
        self.assertEqual(marker.read_text().strip(), "skipped")

    def test_marker_distinguishes_skipped_from_real_run(self):
        cfg = make_cfg(self.base, dedup_enabled=False)
        ctx = Ctx(cfg, self.base / ".work_c1" / "dedup")
        dedup_mod.DedupStage().run(ctx)
        marker = Path(cfg["work_root"]) / "dedup" / ".dedup.done"
        # A skipped corpus must never be mistaken for a deduped one.
        self.assertNotEqual(marker.read_text().strip(), "ok")


class TestRegroupMirror(unittest.TestCase):
    def setUp(self):
        self.base = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.base, ignore_errors=True)
        self.cfg = make_cfg(self.base, dedup_enabled=False)
        self.n = seed_source(self.cfg)
        self.ctx = Ctx(self.cfg, self.base / ".work_c1" / "regroup")

    def test_mirrors_with_symlinks_and_copies_nothing(self):
        out = regroup_mod.RegroupStage().run(self.ctx)
        self.assertEqual(out["counts"]["linked"], self.n)
        self.assertEqual(out["counts"]["removed_files"], 0)
        for split in SPLITS:
            for cls in CLASSES:
                dst = Path(self.cfg["corpus_ready"]) / split / f"{cls}.fasta"
                self.assertTrue(dst.is_symlink(), f"{dst} must be a symlink, not a copy")
                self.assertTrue(dst.exists(), "symlink must resolve")
                self.assertEqual(dst.read_text(), f">{split}_{cls}_1\nACGT\n")

    def test_mirror_is_idempotent(self):
        regroup_mod.RegroupStage().run(self.ctx)
        out = regroup_mod.RegroupStage().run(self.ctx)  # must not raise on existing links
        self.assertEqual(out["counts"]["linked"], self.n)

    def test_summary_records_mirror_mode(self):
        regroup_mod.RegroupStage().run(self.ctx)
        summary = json.loads(
            (Path(self.cfg["results_root"]) / "regroup_summary.json").read_text())
        self.assertEqual(summary["mode"], "mirror")
        self.assertEqual(summary["removed_files"], 0)

    def test_training_path_resolves_to_real_data(self):
        # train_ready == corpus_ready, so training must find readable FASTA.
        regroup_mod.RegroupStage().run(self.ctx)
        p = Path(self.cfg["corpus_ready"]) / "train" / "bacteria.fasta"
        self.assertTrue(os.path.exists(p))
        self.assertIn("ACGT", p.read_text())

    def test_missing_source_file_is_tolerated(self):
        (Path(self.cfg["source_ready"]) / "test" / "archaea.fasta").unlink()
        out = regroup_mod.RegroupStage().run(self.ctx)
        self.assertEqual(out["counts"]["linked"], self.n - 1)


class TestConfigDefault(unittest.TestCase):
    def test_shipped_config_has_dedup_off_and_corpus_ready_wired(self):
        import tiara2.config as C
        cfg = C.load(ROOT / "config" / "config.yaml")
        self.assertFalse(cfg["dedup"]["enabled"], "dedup ships OFF (user opt-in)")
        self.assertEqual(cfg["train"]["train_ready"], cfg["corpus_ready"])


def _run():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite(
        loader.loadTestsFromTestCase(c)
        for c in (TestDedupOptional, TestRegroupMirror, TestConfigDefault))
    res = unittest.TextTestRunner(verbosity=2).run(suite)
    total = res.testsRun
    bad = len(res.failures) + len(res.errors)
    print(f"\n{total - bad}/{total} passed")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    _run()

#!/usr/bin/env python3
"""Tests for the two-tier storage layout and its preflight.

The expensive failure mode here is silent: one hot key stays on /data, the run
starts anyway, and 30 hours later you discover the feature cache was on
spindles the whole time. So the preflight is tested for the cases that are easy
to get wrong -- a key left on the cold tier, tiers that collapse onto one
device, and symlinks on the hot tier that point back into the cold one.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tiara2 import config as config_mod  # noqa: E402
from scripts import check_storage as cs  # noqa: E402


class TestTieredConfig(unittest.TestCase):
    """The shipped config must actually separate the tiers."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = config_mod.load(ROOT / "config" / "config.yaml")

    def test_fast_base_is_declared_and_distinct(self):
        self.assertTrue(self.cfg.get("fast_base"))
        self.assertNotEqual(self.cfg["fast_base"], self.cfg["base"])

    def test_every_hot_key_resolves_under_fast_base(self):
        fast = self.cfg["fast_base"]
        for key in self.cfg["storage"]["hot_keys"]:
            value = cs.dotted_get(self.cfg, key)
            self.assertIsInstance(value, str, key)
            self.assertTrue(value.startswith(fast), f"{key} -> {value}")

    def test_every_cold_key_resolves_under_base(self):
        base = self.cfg["base"]
        for key in self.cfg["storage"]["cold_keys"]:
            value = cs.dotted_get(self.cfg, key)
            self.assertIsInstance(value, str, key)
            self.assertTrue(value.startswith(base), f"{key} -> {value}")

    def test_source_ready_is_hot(self):
        # With dedup off, regroup symlinks source_ready into corpus_ready. A hot
        # corpus_ready whose links target a cold source_ready is pointless.
        self.assertIn("source_ready", self.cfg["storage"]["hot_keys"])
        self.assertTrue(self.cfg["source_ready"].startswith(self.cfg["fast_base"]))

    def test_feature_cache_and_seq_pack_are_hot(self):
        for key in ("feature_cache", "seq_pack"):
            self.assertTrue(self.cfg["train"][key].startswith(self.cfg["fast_base"]), key)

    def test_published_models_stay_cold(self):
        # Durability beats speed for anything we would have to retrain to recover.
        self.assertTrue(self.cfg["train"]["out_models"].startswith(self.cfg["base"]))

    def test_tags_were_bumped_for_the_rebuild(self):
        # A rebuilt corpus must not reuse the tag whose manifests describe the
        # old one, or every stage would report itself already done.
        self.assertNotEqual(self.cfg["corpus_tag"], "v2_0_hybrid")
        self.assertIn(self.cfg["corpus_tag"], self.cfg["work_root"])


class TestDottedGet(unittest.TestCase):
    def test_nested(self):
        self.assertEqual(cs.dotted_get({"a": {"b": 1}}, "a.b"), 1)

    def test_missing_is_none(self):
        self.assertIsNone(cs.dotted_get({"a": {}}, "a.b"))
        self.assertIsNone(cs.dotted_get({"a": 1}, "a.b"))


class TestInspect(unittest.TestCase):
    """Everything below runs on one real device, so `tiers_collapsed` is True and
    the per-key device check is suppressed. That is the honest behaviour: we
    cannot verify tiering when there is only one disk, so we say so instead of
    reporting a false pass."""

    def _cfg(self, tmp: Path, **extra):
        cfg = {
            "base": str(tmp / "cold"),
            "fast_base": str(tmp / "hot"),
            "source_ready": str(tmp / "hot" / "train_ready"),
            "corpus_ready": str(tmp / "hot" / "corpus_ready"),
            "work_root": str(tmp / "hot" / "work"),
            "results_root": str(tmp / "cold" / "results"),
            "storage": {
                "hot_keys": ["source_ready", "corpus_ready", "work_root"],
                "cold_keys": ["results_root"],
                "min_fast_free_gb": 0,
                "min_cold_free_gb": 0,
            },
        }
        cfg.update(extra)
        return cfg

    def test_missing_hot_tier_is_a_problem(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            (tmp / "cold").mkdir()
            report = cs.inspect(self._cfg(tmp), measure=False)
            self.assertTrue(any("hot tier does not exist" in p for p in report["problems"]))

    def test_collapsed_tiers_are_reported(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            (tmp / "cold").mkdir()
            (tmp / "hot").mkdir()
            report = cs.inspect(self._cfg(tmp), measure=False)
            self.assertTrue(report["tiers_collapsed"])
            self.assertTrue(any("same device" in p for p in report["problems"]))

    def test_free_space_floor_is_enforced(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            (tmp / "cold").mkdir()
            (tmp / "hot").mkdir()
            cfg = self._cfg(tmp)
            cfg["storage"]["min_fast_free_gb"] = 10 ** 9   # nothing has an exabyte
            report = cs.inspect(cfg, measure=False)
            self.assertTrue(any("hot tier has" in p for p in report["problems"]))

    def test_sizes_skip_symlinks(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            real = tmp / "cold" / "train_ready"
            real.mkdir(parents=True)
            (real / "eukarya.fasta").write_text(">a\n" + "ACGT" * 100 + "\n")
            mirror = tmp / "hot" / "corpus_ready"
            mirror.mkdir(parents=True)
            os.symlink(real / "eukarya.fasta", mirror / "eukarya.fasta")
            # The mirror is symlinks only: it must not be counted as real bytes.
            self.assertEqual(cs.dir_size_gb(mirror), 0.0)
            self.assertGreater(cs.dir_size_gb(real), 0.0)

    def test_cross_tier_symlinks_are_counted(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            target = tmp / "cold" / "x.fasta"
            target.parent.mkdir(parents=True)
            target.write_text(">a\nACGT\n")
            mirror = tmp / "hot" / "corpus_ready"
            mirror.mkdir(parents=True)
            os.symlink(target, mirror / "x.fasta")
            # Same device in the sandbox, so nothing crosses; the point of the
            # test is that the walk finds and stats the link at all.
            crossing, checked = cs.count_cross_tier_symlinks(mirror, cs.device_of(mirror))
            self.assertEqual(checked, 1)
            self.assertEqual(crossing, 0)
            # A device that nothing lives on: now the link must count as crossing.
            crossing, checked = cs.count_cross_tier_symlinks(mirror, -12345)
            self.assertEqual((crossing, checked), (1, 1))

    def test_clean_layout_has_no_problems(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = self._cfg(tmp, fast_base=str(tmp / "cold"))  # collapse on purpose
            (tmp / "cold").mkdir()
            cfg["source_ready"] = str(tmp / "cold" / "train_ready")
            cfg["corpus_ready"] = str(tmp / "cold" / "corpus_ready")
            cfg["work_root"] = str(tmp / "cold" / "work")
            report = cs.inspect(cfg, measure=True)
            # fast_base == base is the documented single-disk mode: no complaint.
            self.assertEqual(report["problems"], [])


if __name__ == "__main__":
    unittest.main(verbosity=1)

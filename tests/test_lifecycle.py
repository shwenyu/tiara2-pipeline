"""Tests for the run-lifecycle surface: inventory, retention, versioning.

Run directly (the sandbox has no pytest):

    python3 tests/test_lifecycle.py

NOTE: `python3 -m unittest tests.test_lifecycle` reports "Ran 0 tests" AND
exits 0 in this environment, which looks green while testing nothing. Always
invoke the file directly.

What is actually being protected here
-------------------------------------
1. Retention deletes multi-TB directories. The safety guard (inside base, not a
   protected root, not a PARENT of a protected root) is the only thing between
   a config typo and the corpus, so it is tested from both directions.
2. The two-tag layout exists to stop a training-version bump from re-running
   the mmseqs dedup. That is a property of the resolved config, so it is
   asserted against the REAL config/config.yaml rather than a fixture.
3. Prompts must never block a nohup run: confirm() has to return its default
   when stdin is not a TTY.
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tiara2 import inventory, retention, versioning  # noqa: E402
from tiara2 import config as cfg_mod  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def write(path: Path, size: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return path


def make_cfg(base: Path) -> dict:
    """Minimal but realistic config rooted at a temp dir."""
    return {
        "base": str(base),
        "data_home": str(base / "home"),
        "source_ready": str(base / "train_ready_c1"),
        "work_root": str(base / ".work_c1"),
        "results_root": str(base / "results_v1"),
        "log_dir": str(base / "log" / "pipeline_v1"),
        "splits": ["train", "validation"],
        "classes": ["bacteria", "archaea"],
        "corpus_tag": "c1",
        "version_tag": "v1",
        "model_tag": "m1",
        "train": {
            "train_ready": str(base / "train_ready_c1"),
            "flat_data": str(base / "flat_c1"),
            "tfidf_dir": str(base / "tfidf_c1"),
            "log_dir": str(base / "log" / "train_v1"),
            "out_models": str(base / "models_m1"),
            "feature_cache": str(base / "feature_cache_v1"),
            "seq_pack": str(base / "seqpack_c1"),
        },
        "retention": {"policy": "report",
                       "delete": ["flat_data", "scratch", "binned_shards"],
                       "keep": ["seq_pack", "feature_cache"]},
        "versioning": {"registry": str(base / "versions.json"),
                        "snapshot": str(base / "results_v1" / "snap.json")},
    }


class TestInventory(unittest.TestCase):
    def test_human_units(self):
        self.assertEqual(inventory.human(0), "0 B")
        self.assertEqual(inventory.human(512), "512 B")
        self.assertEqual(inventory.human(1024), "1.0 KiB")
        self.assertIn("GiB", inventory.human(3 * 1024 ** 3))
        self.assertIn("TiB", inventory.human(5 * 1024 ** 4))

    def test_dir_stat_counts_bytes_recursively(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "tree"
            write(root / "a.bin", 100)
            write(root / "sub" / "b.bin", 250)
            write(root / "sub" / "deep" / "c.bin", 50)
            st = inventory.dir_stat(root)
            self.assertTrue(st.exists)
            self.assertEqual(st.files, 3)
            self.assertEqual(st.bytes, 400)
            self.assertFalse(st.truncated)

    def test_dir_stat_truncates_and_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "many"
            for i in range(30):
                write(root / f"f{i}.bin", 10)
            st = inventory.dir_stat(root, max_entries=5)
            self.assertTrue(st.truncated)
            self.assertTrue(st.size_h.startswith(">="))

    def test_missing_paths_never_raise(self):
        st = inventory.dir_stat("/definitely/not/here")
        self.assertFalse(st.exists)
        self.assertEqual(st.bytes, 0)
        self.assertEqual(inventory.file_stat("/nope").exists, False)
        self.assertEqual(inventory.shallow_children("/nope"), [])

    def test_corpus_inventory_and_render(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            cfg = make_cfg(base)
            write(base / "train_ready_c1" / "train" / "bacteria.fasta", 300)
            write(base / "train_ready_c1" / "validation" / "archaea.fasta", 100)
            inv = inventory.corpus_inventory(cfg)
            self.assertEqual(inv["expected"], 4)   # 2 splits x 2 classes
            self.assertEqual(inv["present"], 2)
            self.assertEqual(inv["total_bytes"], 400)
            text = inventory.render_corpus(inv)
            self.assertIn("bacteria", text)
            self.assertIn("MISSING", text)


class TestRetentionSafety(unittest.TestCase):
    def test_refuses_paths_outside_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_cfg(Path(tmp))
            ok, why = retention.is_safe_target(cfg, "/etc")
            self.assertFalse(ok)
            self.assertIn("outside base", why)

    def test_refuses_base_itself(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_cfg(Path(tmp))
            ok, _ = retention.is_safe_target(cfg, tmp)
            self.assertFalse(ok)

    def test_refuses_protected_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            cfg = make_cfg(base)
            for guarded in (cfg["source_ready"], cfg["train"]["tfidf_dir"],
                            cfg["train"]["out_models"], cfg["train"]["log_dir"],
                            cfg["results_root"]):
                ok, why = retention.is_safe_target(cfg, guarded)
                self.assertFalse(ok, f"should refuse {guarded}: {why}")

    def test_refuses_parent_of_protected_root(self):
        """Deleting log/ would take log/train_v1 (hp results) with it."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            cfg = make_cfg(base)
            ok, why = retention.is_safe_target(cfg, base / "log")
            self.assertFalse(ok)
            self.assertIn("protected root", why)

    def test_allows_ordinary_derived_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            cfg = make_cfg(base)
            ok, why = retention.is_safe_target(cfg, cfg["train"]["flat_data"])
            self.assertTrue(ok, why)


class TestRetentionPlan(unittest.TestCase):
    def _populated(self, tmp: str) -> dict:
        base = Path(tmp)
        cfg = make_cfg(base)
        write(base / "flat_c1" / "bacteria_fr.fasta", 1000)
        write(base / "feature_cache_v1" / "first" / "k4" / "train_X.f32", 500)
        write(base / "seqpack_c1" / "shard0.pack", 700)
        write(base / ".work_c1" / "chop_bin" / "binned" / "s.fasta", 300)
        write(base / "flat_c1.tmp.123.456" / "junk", 50)
        return cfg

    def test_flat_data_is_dead_when_cache_configured(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._populated(tmp)
            items = {i.key: i for i in retention.plan(cfg)}
            self.assertEqual(items["flat_data"].category, retention.CAT_DEAD)
            self.assertEqual(items["flat_data"].stat.bytes, 1000)

    def test_flat_data_is_needed_without_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._populated(tmp)
            cfg["train"]["feature_cache"] = ""
            items = {i.key: i for i in retention.plan(cfg)}
            self.assertEqual(items["flat_data"].category,
                             retention.CAT_EXPENSIVE)

    def test_seq_pack_is_expensive_and_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._populated(tmp)
            items = retention.select(cfg, retention.plan(cfg))
            pack = [i for i in items if i.key == "seq_pack"][0]
            self.assertEqual(pack.category, retention.CAT_EXPENSIVE)
            self.assertFalse(pack.selected, "seq_pack is in retention.keep")

    def test_keep_overrides_delete(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._populated(tmp)
            cfg["retention"]["delete"] = ["flat_data", "feature_cache"]
            cfg["retention"]["keep"] = ["feature_cache"]
            items = {i.key: i for i in
                     retention.select(cfg, retention.plan(cfg))}
            self.assertTrue(items["flat_data"].selected)
            self.assertFalse(items["feature_cache"].selected)

    def test_binned_shards_dead_only_after_regroup(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._populated(tmp)
            items = {i.key: i for i in retention.plan(cfg)}
            self.assertEqual(items["binned_shards"].category,
                             retention.CAT_EXPENSIVE)
            manifest = Path(cfg["work_root"]) / "regroup" / "manifest.json"
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text("{}")
            items = {i.key: i for i in retention.plan(cfg)}
            self.assertEqual(items["binned_shards"].category,
                             retention.CAT_DEAD)

    def test_scratch_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._populated(tmp)
            keys = [i.key for i in retention.plan(cfg)]
            self.assertTrue(any(k.startswith("scratch:") for k in keys),
                            f"no scratch entry in {keys}")

    def test_dry_run_apply_removes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._populated(tmp)
            items = retention.select(cfg, retention.plan(cfg))
            result = retention.apply(cfg, items, dry_run=True)
            self.assertGreater(result["freed_bytes"], 0)
            self.assertTrue(Path(cfg["train"]["flat_data"]).exists())

    def test_apply_deletes_only_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._populated(tmp)
            items = retention.select(cfg, retention.plan(cfg))
            retention.apply(cfg, items, dry_run=False)
            self.assertFalse(Path(cfg["train"]["flat_data"]).exists())
            self.assertTrue(Path(cfg["train"]["seq_pack"]).exists())
            self.assertTrue(Path(cfg["train"]["feature_cache"]).exists())

    def test_apply_reenforces_guard_on_forced_selection(self):
        """Even a hand-forced selection cannot delete a protected root."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._populated(tmp)
            evil = retention.Item(key="evil", path=cfg["source_ready"],
                                  category=retention.CAT_DEAD, reason="-",
                                  rebuild="-",
                                  stat=inventory.dir_stat(cfg["source_ready"]))
            evil.selected = True
            result = retention.apply(cfg, [evil], dry_run=False)
            self.assertEqual(result["removed"], [])
            self.assertEqual(len(result["skipped"]), 1)

    def test_render_is_printable(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._populated(tmp)
            text = retention.render(retention.select(cfg, retention.plan(cfg)))
            self.assertIn("DEAD", text)
            self.assertIn("marked for deletion", text)


class TestVersioning(unittest.TestCase):
    def test_validate_tag(self):
        self.assertEqual(versioning.validate_tag(" v2_1 "), "v2_1")
        for bad in ("", "has space", "../escape", "a/b", "-leading"):
            with self.assertRaises(versioning.VersionError):
                versioning.validate_tag(bad)

    def test_suggest_next_bumps_last_number(self):
        self.assertEqual(versioning.suggest_next("v2_0_hybrid"), "v2_1_hybrid")
        self.assertEqual(versioning.suggest_next("v9"), "v10")
        self.assertEqual(versioning.suggest_next("plain"), "plain_2")

    def test_tag_accessors_fall_back_to_legacy_keys(self):
        self.assertEqual(versioning.corpus_tag({"input_tag": "old"}), "old")
        self.assertEqual(versioning.version_tag({"output_tag": "out"}), "out")

    def test_register_run_appends_atomically(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_cfg(Path(tmp))
            versioning.register_run(cfg, stages=["train"], note="first")
            versioning.register_run(cfg, stages=["publish"], note="second")
            data = versioning.read_registry(cfg)
            self.assertEqual(len(data["runs"]), 2)
            self.assertEqual(data["runs"][0]["note"], "first")
            self.assertEqual(versioning.last_run(cfg)["note"], "second")
            self.assertEqual(versioning.known_versions(cfg), ["v1"])

    def test_corrupt_registry_is_not_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_cfg(Path(tmp))
            Path(cfg["versioning"]["registry"]).write_text("{not json")
            self.assertEqual(versioning.read_registry(cfg), {"runs": []})
            versioning.register_run(cfg, stages=["train"])
            self.assertEqual(len(versioning.read_registry(cfg)["runs"]), 1)

    def test_snapshot_config_writes_resolved_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_cfg(Path(tmp))
            path = versioning.snapshot_config(cfg)
            self.assertIsNotNone(path)
            payload = json.loads(Path(path).read_text())
            self.assertEqual(payload["version_tag"], "v1")
            self.assertEqual(payload["corpus_tag"], "c1")
            self.assertEqual(payload["config"]["base"], str(Path(tmp)))
            self.assertEqual(len(payload["config_sha"]), 64)

    def test_config_sha_is_stable_and_sensitive(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_cfg(Path(tmp))
            first = versioning.config_sha(cfg)
            self.assertEqual(first, versioning.config_sha(dict(cfg)))
            cfg["model_tag"] = "changed"
            self.assertNotEqual(first, versioning.config_sha(cfg))

    def test_confirm_defaults_when_not_a_tty(self):
        """The nohup case: must not block, must not raise, must use default."""
        saved = sys.stdin
        sys.stdin = io.StringIO("")   # a StringIO is never a TTY
        try:
            self.assertFalse(versioning.confirm("delete?", default=False))
            self.assertTrue(versioning.confirm("keep?", default=True))
            self.assertTrue(versioning.confirm("x", default=False,
                                               assume_yes=True))
            self.assertFalse(versioning.interactive())
        finally:
            sys.stdin = saved

    def test_prompt_for_version_returns_none_when_not_a_tty(self):
        saved_in, saved_out = sys.stdin, sys.stdout
        sys.stdin = io.StringIO("")
        sys.stdout = io.StringIO()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                cfg = make_cfg(Path(tmp))
                self.assertIsNone(versioning.prompt_for_version(cfg))
        finally:
            sys.stdin, sys.stdout = saved_in, saved_out


class TestTwoTagConfig(unittest.TestCase):
    """Asserted against the REAL config: this is the whole point of the split."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = cfg_mod.load(REPO / "config" / "config.yaml")

    def test_legacy_aliases_follow_the_new_tags(self):
        self.assertEqual(self.cfg["input_tag"], self.cfg["corpus_tag"])
        self.assertEqual(self.cfg["output_tag"], self.cfg["version_tag"])

    def test_corpus_layer_is_keyed_on_corpus_tag(self):
        corpus = self.cfg["corpus_tag"]
        self.assertTrue(self.cfg["work_root"].endswith(corpus),
                        self.cfg["work_root"])
        self.assertTrue(self.cfg["source_ready"].endswith(corpus))
        self.assertTrue(self.cfg["train"]["tfidf_dir"].endswith(corpus))
        self.assertTrue(self.cfg["train"]["seq_pack"].endswith(corpus))

    def test_training_layer_is_keyed_on_version_tag(self):
        version = self.cfg["version_tag"]
        self.assertTrue(self.cfg["train"]["feature_cache"].endswith(version))
        self.assertTrue(self.cfg["train"]["log_dir"].endswith(version))
        self.assertTrue(self.cfg["results_root"].endswith(version))

    def test_bumping_version_tag_does_not_move_work_root(self):
        """The regression this whole change exists to prevent."""
        bumped = cfg_mod.load(REPO / "config" / "config.yaml",
                              overrides=["version_tag=v9_9_test"])
        self.assertEqual(bumped["work_root"], self.cfg["work_root"])
        self.assertEqual(bumped["source_ready"], self.cfg["source_ready"])
        self.assertEqual(bumped["train"]["tfidf_dir"],
                         self.cfg["train"]["tfidf_dir"])
        # ...but the training layer DOES move.
        self.assertNotEqual(bumped["train"]["feature_cache"],
                            self.cfg["train"]["feature_cache"])
        self.assertIn("v9_9_test", bumped["train"]["log_dir"])
        self.assertIn("v9_9_test", bumped["results_root"])

    def test_bumping_corpus_tag_moves_the_corpus_layer(self):
        bumped = cfg_mod.load(REPO / "config" / "config.yaml",
                              overrides=["corpus_tag=c9_test"])
        self.assertIn("c9_test", bumped["work_root"])
        self.assertIn("c9_test", bumped["source_ready"])
        self.assertIn("c9_test", bumped["train"]["tfidf_dir"])

    def test_retention_and_versioning_blocks_resolve(self):
        self.assertIn(self.cfg["retention"]["policy"],
                      ("off", "report", "prompt", "auto"))
        self.assertIn("flat_data", self.cfg["retention"]["delete"])
        self.assertIn("seq_pack", self.cfg["retention"]["keep"])
        registry = self.cfg["versioning"]["registry"]
        self.assertTrue(registry.startswith(self.cfg["base"]), registry)
        self.assertNotIn("{", self.cfg["versioning"]["snapshot"])


class TestFlatDataIsNoLongerBuilt(unittest.TestCase):
    def test_stage_skips_flat_when_cache_configured(self):
        from tiara2.stages.train import TrainStage
        self.assertFalse(TrainStage._needs_flat({"train": {"feature_cache": "/c"}}))
        self.assertTrue(TrainStage._needs_flat({"train": {}}))
        self.assertTrue(TrainStage._needs_flat({"train": {"feature_cache": ""}}))

    def test_backend_points_at_train_ready_when_cache_configured(self):
        from tiara2.model_backend import TiaraBackend
        with_cache = {"train_ready": "/tr", "flat_data": "/flat",
                      "feature_cache": "/fc"}
        self.assertEqual(TiaraBackend._model_data_dir(with_cache), "/tr")
        without = {"train_ready": "/tr", "flat_data": "/flat"}
        self.assertEqual(TiaraBackend._model_data_dir(without), "/flat")


def _run() -> int:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for case in (TestInventory, TestRetentionSafety, TestRetentionPlan,
                 TestVersioning, TestTwoTagConfig, TestFlatDataIsNoLongerBuilt):
        suite.addTests(loader.loadTestsFromTestCase(case))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    total = result.testsRun
    bad = len(result.failures) + len(result.errors)
    print(f"\n{total - bad}/{total} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(_run())

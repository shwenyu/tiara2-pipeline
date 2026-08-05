"""acquire is now a per-step, optional, full-stack stage.

Invariants protected here:
  * step order is owned by STEP_ORDER, never by the config;
  * the stage never shells out when disabled, dry-run, or declined;
  * the download step is the one that carries --class;
  * a missing external script fails with a message naming where we looked.
"""
import logging
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tiara2.stages import acquire as A  # noqa: E402


class Ctx:
    def __init__(self, cfg, work_dir, *, dry_run=False, extra=None):
        self.cfg = cfg
        self.work_dir = Path(work_dir)
        self.log = logging.getLogger("test")
        self.dry_run = dry_run
        self.force = False
        self.debug = False
        self.extra = extra or {}


def make_cfg(base: Path, **acq) -> dict:
    steps = acq.pop("steps", None)
    cfg = {
        "base": str(base),
        "data_home": str(base),
        "source_ready": str(base / "train_ready_c1"),
        "splits": ["train"],
        "classes": ["archaea"],
        "acquire": {
            "enabled": True,
            "script": "ncbi_pipeline.py",
            "legacy_config": "config.json",
            "confirm": False,
            "steps": steps if steps is not None else {},
        },
    }
    cfg["acquire"].update(acq)
    return cfg


def seed_script(base: Path, name: str = "ncbi_pipeline.py") -> Path:
    script = base / name
    script.write_text("print('stub')\n")
    (base / "config.json").write_text("{}\n")
    return script


class TestStepSelection(unittest.TestCase):
    def test_default_is_every_step_in_canonical_order(self):
        cfg = make_cfg(Path("/tmp"))
        self.assertEqual(A.planned_steps(cfg), list(A.STEP_ORDER))

    def test_config_cannot_reorder_steps(self):
        # Even declared backwards, execution order stays canonical.
        cfg = make_cfg(Path("/tmp"), steps={s: True for s in reversed(A.STEP_ORDER)})
        self.assertEqual(A.planned_steps(cfg), list(A.STEP_ORDER))

    def test_disabling_a_step_removes_only_that_step(self):
        cfg = make_cfg(Path("/tmp"), steps={"download": False})
        planned = A.planned_steps(cfg)
        self.assertNotIn("download", planned)
        self.assertIn("verify", planned)
        self.assertEqual(A.skipped_steps(cfg), ["download"])

    def test_string_falsey_values_from_set_are_honoured(self):
        for text in ("false", "0", "no", "off", "False"):
            cfg = make_cfg(Path("/tmp"), steps={"download": text})
            self.assertNotIn("download", A.planned_steps(cfg), text)

    def test_unknown_step_names_are_ignored(self):
        cfg = make_cfg(Path("/tmp"), steps={"not-a-step": True})
        self.assertEqual(A.planned_steps(cfg), list(A.STEP_ORDER))

    def test_download_is_classified_as_a_network_step(self):
        self.assertIn("download", A.NETWORK_STEPS)
        self.assertIn("fetch-metadata", A.NETWORK_STEPS)
        self.assertNotIn("build-index", A.NETWORK_STEPS)


class TestScriptResolution(unittest.TestCase):
    def setUp(self):
        self.base = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.base, ignore_errors=True)

    def test_bundled_copy_wins_by_default(self):
        """Since v2.2h the script ships at scripts/ and needs no configuration.

        A stray copy under ``base`` must NOT shadow it: the bundled one is the
        version the package was tested against.
        """
        seed_script(self.base)
        cfg = make_cfg(self.base)
        repo_root = Path(A.__file__).resolve().parents[2]
        self.assertEqual(A.resolve_script(cfg),
                         repo_root / "scripts" / "ncbi_pipeline.py")

    def test_finds_script_via_base(self):
        """``base`` is still searched, for a script name we do not bundle."""
        script = seed_script(self.base, name="ncbi_pipeline_fork.py")
        cfg = make_cfg(self.base, script="ncbi_pipeline_fork.py")
        self.assertEqual(A.resolve_script(cfg), script)

    def test_absolute_path_is_used_directly(self):
        script = seed_script(self.base)
        cfg = make_cfg(self.base, script=str(script))
        self.assertEqual(A.resolve_script(cfg), script)

    def test_missing_script_names_the_candidates(self):
        cfg = make_cfg(self.base, script="nope.py")
        with self.assertRaises(FileNotFoundError) as cm:
            A.resolve_script(cfg)
        self.assertIn("Tried:", str(cm.exception))

    def test_legacy_config_defaults_beside_script(self):
        script = seed_script(self.base)
        cfg = make_cfg(self.base)
        self.assertEqual(A.resolve_legacy_config(cfg, script),
                         self.base / "config.json")


class TestRunBehaviour(unittest.TestCase):
    def setUp(self):
        self.base = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.base, ignore_errors=True)
        seed_script(self.base)

    def _run(self, cfg, **kw):
        ctx = Ctx(cfg, self.base / "work" / "acquire", **kw)
        with mock.patch.object(A.subprocess, "run") as m:
            m.return_value = mock.Mock(returncode=0)
            out = A.AcquireStage().run(ctx)
        return out, m

    def test_disabled_stage_never_shells_out(self):
        cfg = make_cfg(self.base)
        cfg["acquire"]["enabled"] = False
        out, m = self._run(cfg)
        m.assert_not_called()
        self.assertEqual(out["counts"], {"skipped": 1})

    def test_dry_run_never_shells_out(self):
        out, m = self._run(make_cfg(self.base), dry_run=True)
        m.assert_not_called()
        self.assertEqual(out["counts"]["dry_run"], 1)

    def test_each_step_is_its_own_invocation(self):
        cfg = make_cfg(self.base, steps={s: False for s in A.STEP_ORDER
                                         if s not in ("select", "verify")})
        out, m = self._run(cfg)
        self.assertEqual(out["counts"]["steps"], 2)
        self.assertEqual(m.call_count, 2, "one subprocess per step")
        called = [c.args[0][2] for c in m.call_args_list]
        self.assertEqual(called, ["select", "verify"])

    def test_only_class_is_passed_to_download_alone(self):
        cfg = make_cfg(self.base, only_class="archaea",
                       steps={s: s in ("download", "verify")
                              for s in A.STEP_ORDER})
        _, m = self._run(cfg)
        by_step = {c.args[0][2]: c.args[0] for c in m.call_args_list}
        self.assertIn("--class", by_step["download"])
        self.assertNotIn("--class", by_step["verify"])

    def test_failed_step_names_the_step(self):
        cfg = make_cfg(self.base, steps={s: s == "select" for s in A.STEP_ORDER})
        ctx = Ctx(cfg, self.base / "work" / "acquire")
        with mock.patch.object(A.subprocess, "run") as m:
            m.return_value = mock.Mock(returncode=3)
            with self.assertRaises(RuntimeError) as cm:
                A.AcquireStage().run(ctx)
        self.assertIn("select", str(cm.exception))

    def test_declining_confirmation_fetches_nothing(self):
        cfg = make_cfg(self.base)
        cfg["acquire"]["confirm"] = True
        ctx = Ctx(cfg, self.base / "work" / "acquire")
        with mock.patch.object(A.versioning, "confirm", return_value=False), \
                mock.patch.object(A.subprocess, "run") as m:
            out = A.AcquireStage().run(ctx)
        m.assert_not_called()
        self.assertEqual(out["counts"], {"declined": 1})

    def test_no_confirmation_when_no_network_steps(self):
        cfg = make_cfg(self.base, steps={s: s == "build-index"
                                         for s in A.STEP_ORDER})
        cfg["acquire"]["confirm"] = True
        ctx = Ctx(cfg, self.base / "work" / "acquire")
        with mock.patch.object(A.versioning, "confirm") as conf, \
                mock.patch.object(A.subprocess, "run") as m:
            m.return_value = mock.Mock(returncode=0)
            A.AcquireStage().run(ctx)
        conf.assert_not_called()

    def test_per_step_log_files_are_written(self):
        cfg = make_cfg(self.base, steps={s: s == "select" for s in A.STEP_ORDER})
        self._run(cfg)
        log = self.base / "work" / "acquire" / "logs" / "acquire_select.log"
        self.assertTrue(log.exists(), "each step gets its own log")


class TestShippedConfig(unittest.TestCase):
    def test_config_exposes_every_step(self):
        import tiara2.config as C
        cfg = C.load(ROOT / "config" / "config.yaml")
        steps = cfg["acquire"]["steps"]
        for name in A.STEP_ORDER:
            self.assertIn(name, steps, f"{name} must be toggleable")
        self.assertFalse(cfg["acquire"]["enabled"], "acquire ships OFF")


def _run():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite(
        loader.loadTestsFromTestCase(c) for c in (
            TestStepSelection, TestScriptResolution, TestRunBehaviour,
            TestShippedConfig))
    res = unittest.TextTestRunner(verbosity=2).run(suite)
    total = res.testsRun
    bad = len(res.failures) + len(res.errors)
    print(f"\n{total - bad}/{total} passed")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    _run()

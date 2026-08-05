#!/usr/bin/env python3
"""Stage selection (`--only a,b`) and artifact tiering defaults.

Both of these were silent failure modes rather than crashes, which is why they
get tests:

- `--only chop_bin,regroup` used to be looked up as ONE stage named
  "chop_bin,regroup", print "stage not implemented yet ... (skipping)", and
  exit 0. A no-op run that looks like a successful one.
- Plan artifacts (write-once, read-once TSVs) written to the NVMe tier cost
  nothing visible today and quietly eat the space the feature cache needs.

Run directly:  python3 tests/test_stage_selection.py
"""
import io
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from tiara2 import cli  # noqa: E402
import import_tiara_corpus as I  # noqa: E402

CONFIG = str(ROOT / "config" / "config.yaml")


class Args:
    def __init__(self, only=None, from_stage=None, to_stage=None):
        self.only = only
        self.from_stage = from_stage
        self.to_stage = to_stage


class TestOnlySelection(unittest.TestCase):
    def test_single_stage(self):
        self.assertEqual(cli._resolve_order(Args(only="curate")), ["curate"])

    def test_comma_separated(self):
        self.assertEqual(cli._resolve_order(Args(only="chop_bin,regroup")),
                         ["chop_bin", "regroup"])

    def test_whitespace_and_duplicates_tolerated(self):
        self.assertEqual(cli._resolve_order(Args(only=" chop_bin , regroup ,chop_bin")),
                         ["chop_bin", "regroup"])

    def test_order_is_always_the_dag_order(self):
        # Asking backwards must not run regroup before chop_bin.
        self.assertEqual(cli._resolve_order(Args(only="regroup,chop_bin")),
                         ["chop_bin", "regroup"])

    def test_unknown_stage_fails_loudly(self):
        with self.assertRaises(SystemExit) as ctx:
            cli._resolve_order(Args(only="chop_bin,regoup"))
        self.assertIn("regoup", str(ctx.exception))
        self.assertIn("known stages", str(ctx.exception))

    def test_from_to_still_work(self):
        cli._load_stages()
        registered = cli.registry()
        if "chop_bin" not in registered or "regroup" not in registered:
            self.skipTest("chop_bin/regroup not registered in this environment")
        order = cli._resolve_order(Args(from_stage="chop_bin", to_stage="regroup"))
        self.assertEqual(order[0], "chop_bin")
        self.assertEqual(order[-1], "regroup")


class TestImportArtifactTier(unittest.TestCase):
    def test_default_out_dir_is_on_the_cold_tier(self):
        out = I.default_out_dir(CONFIG)
        cfg = I._load_cfg(CONFIG)
        self.assertTrue(out.startswith(str(cfg["base"])), out)
        self.assertTrue(out.endswith(str(Path("import") / "tiara1")), out)
        # And explicitly NOT the hot tier.
        self.assertFalse(out.startswith(str(cfg["fast_base"])), out)

    def test_hot_tier_out_dir_warns(self):
        cfg = I._load_cfg(CONFIG)
        buf = io.StringIO()
        with redirect_stderr(buf):
            I.warn_hot_tier(Path(cfg["fast_base"]) / ".import_tiara", CONFIG)
        self.assertIn("HOT tier", buf.getvalue())

    def test_cold_tier_out_dir_is_silent(self):
        cfg = I._load_cfg(CONFIG)
        buf = io.StringIO()
        with redirect_stderr(buf):
            I.warn_hot_tier(Path(cfg["base"]) / "import" / "tiara1", CONFIG)
        self.assertEqual(buf.getvalue(), "")


if __name__ == "__main__":
    unittest.main(verbosity=2)

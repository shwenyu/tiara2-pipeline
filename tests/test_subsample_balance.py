"""Per-class subsampling / class rebalancing.

The invariants worth protecting here are the ones whose violation is silent:
  * the default config must produce a byte-identical cache signature, or the
    existing multi-TB feature cache is invalidated for nothing;
  * the three call sites must agree on a class key, or they select different
    record sets and the final models train on data HP never scored;
  * val/test must never be rebalanced.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Import the training module top-level, mirroring test_seqpack.py and the
# pipeline's `python -m tiara.training.*` entry points. Going through the
# tiara package __init__ would drag in tqdm/Bio, which this environment lacks.
for _p in (str(ROOT / "tiara" / "training"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import seqpack  # noqa: E402


class TestClassKey(unittest.TestCase):
    def test_corpus_and_flat_layouts_agree(self):
        # These two spellings are the same class on different code paths.
        self.assertEqual(seqpack.class_of_path("/x/bacteria.fasta"), "bacteria")
        self.assertEqual(seqpack.class_of_path("/x/bacteria_fr.fasta"), "bacteria")

    def test_abbreviated_names_are_aliased(self):
        # plast_fr.fasta must NOT key as 'plast' or it silently misses its rate.
        self.assertEqual(seqpack.class_of_path("/x/plast_fr.fasta"), "plastids")
        self.assertEqual(seqpack.class_of_path("/x/mitochondria_fr.fasta"),
                         "mitochondria")

    def test_case_and_extensions(self):
        for name in ("Archaea.FASTA", "archaea.fa", "archaea.fna"):
            self.assertEqual(seqpack.class_of_path(f"/x/{name}"), "archaea", name)


class TestRateResolution(unittest.TestCase):
    def test_falls_back_to_uniform_rate(self):
        spec = {"rate": 0.05}
        self.assertAlmostEqual(seqpack.rate_for(spec, "archaea"), 0.05)

    def test_per_class_rate_wins(self):
        spec = {"rate": 0.05, "class_rates": {"archaea": 1.0}}
        self.assertAlmostEqual(seqpack.rate_for(spec, "archaea"), 1.0)
        self.assertAlmostEqual(seqpack.rate_for(spec, "eukarya"), 0.05)

    def test_rate_for_path_matches_rate_for_class(self):
        spec = {"rate": 0.05, "class_rates": {"plastids": 0.7}}
        self.assertAlmostEqual(
            seqpack.rate_for_path(spec, "/d/plast_fr.fasta"), 0.7)


class TestBalanceRates(unittest.TestCase):
    counts = {"eukarya": 26_000_000, "bacteria": 10_000_000,
              "archaea": 46_000, "mitochondria": 200_000,
              "plastids": 300_000}

    def test_none_is_uniform(self):
        r = seqpack.balance_rates(self.counts, mode="none", base_rate=0.05)
        self.assertEqual(set(r.values()), {0.05})

    def test_sqrt_lifts_rare_classes_above_abundant_ones(self):
        r = seqpack.balance_rates(self.counts, mode="sqrt",
                                  target_total=4_000_000)
        self.assertGreater(r["archaea"], r["bacteria"])
        self.assertGreater(r["bacteria"], r["eukarya"])

    def test_sqrt_is_gentler_than_linear(self):
        sq = seqpack.balance_rates(self.counts, mode="sqrt", target_total=4_000_000)
        ln = seqpack.balance_rates(self.counts, mode="linear", target_total=4_000_000)
        # linear drives the abundant class down harder than sqrt does.
        self.assertLess(ln["eukarya"], sq["eukarya"])

    def test_never_oversamples(self):
        # A rare class cannot exceed rate 1.0 -- we can drop rows, not invent them.
        r = seqpack.balance_rates(self.counts, mode="linear",
                                  target_total=100_000_000)
        for cls, rate in r.items():
            self.assertLessEqual(rate, 1.0, cls)

    def test_sqrt_moves_the_mix_toward_balance(self):
        r = seqpack.balance_rates(self.counts, mode="sqrt", target_total=4_000_000)
        rows = seqpack.expected_rows(self.counts, r)
        total = sum(rows.values())
        before = self.counts["archaea"] / sum(self.counts.values())
        after = rows["archaea"] / total
        self.assertGreater(after, before * 5, "archaea share must rise sharply")

    def test_empty_and_zero_counts_are_safe(self):
        self.assertEqual(seqpack.balance_rates({}, mode="sqrt"), {})
        self.assertEqual(seqpack.balance_rates({"a": 0}, mode="sqrt"), {})

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            seqpack.balance_rates(self.counts, mode="bogus")


class TestSignatureStability(unittest.TestCase):
    """The expensive one: don't invalidate ~6 TB of cache by accident."""

    base_cfg = {
        "enabled": True, "seed": 42,
        "train": {"rate": 0.05, "min_acgt_purity": 0.9, "min_len": 1000},
        "validation": {"rate": 1.0, "min_acgt_purity": 0.0, "min_len": 0},
        "test": {"rate": 1.0, "min_acgt_purity": 0.0, "min_len": 0},
    }

    def test_empty_class_rates_do_not_change_the_spec(self):
        with_key = dict(self.base_cfg)
        with_key["train"] = dict(self.base_cfg["train"], class_rates={})
        self.assertEqual(seqpack.normalize_subsample(self.base_cfg),
                         seqpack.normalize_subsample(with_key))

    def test_no_class_rates_key_when_unused(self):
        spec = seqpack.split_spec(seqpack.normalize_subsample(self.base_cfg),
                                  "train")
        self.assertNotIn("class_rates", spec)

    def test_setting_class_rates_does_change_the_spec(self):
        # It must invalidate: the selected rows genuinely differ.
        cfg = dict(self.base_cfg)
        cfg["train"] = dict(self.base_cfg["train"], class_rates={"archaea": 1.0})
        self.assertNotEqual(seqpack.normalize_subsample(self.base_cfg),
                            seqpack.normalize_subsample(cfg))

    def test_class_rates_are_order_independent(self):
        a = dict(self.base_cfg, train=dict(
            self.base_cfg["train"], class_rates={"archaea": 1.0, "bacteria": 0.3}))
        b = dict(self.base_cfg, train=dict(
            self.base_cfg["train"], class_rates={"bacteria": 0.3, "archaea": 1.0}))
        self.assertEqual(seqpack.normalize_subsample(a),
                         seqpack.normalize_subsample(b))

    def test_validation_is_not_rebalanced(self):
        cfg = dict(self.base_cfg, train=dict(
            self.base_cfg["train"], class_rates={"archaea": 1.0}))
        val = seqpack.split_spec(seqpack.normalize_subsample(cfg), "validation")
        self.assertNotIn("class_rates", val)
        self.assertAlmostEqual(seqpack.rate_for(val, "archaea"), 1.0)
        self.assertAlmostEqual(seqpack.rate_for(val, "eukarya"), 1.0)


class TestSelectionConsistency(unittest.TestCase):
    """All three paths must select the SAME records for a class."""

    def test_pack_and_flat_paths_pick_identical_records(self):
        spec = {"rate": 0.05, "min_purity": 0.0, "min_len": 0, "seed": 42,
                "class_rates": {"plastids": 0.4}}
        seqs = [f"ACGT{i}ACGTACGTACGTACGT" for i in range(4000)]

        corpus_rate = seqpack.rate_for_path(spec, "/corpus/plastids.fasta")
        flat_rate = seqpack.rate_for_path(spec, "/flat/plast_fr.fasta")
        explicit = seqpack.rate_for(spec, "plastids")
        self.assertEqual(corpus_rate, flat_rate)
        self.assertEqual(corpus_rate, explicit)

        keep = seqpack.keep_sequence
        a = [s for s in seqs if keep(s, rate=corpus_rate, seed=42)]
        b = [s for s in seqs if keep(s, rate=flat_rate, seed=42)]
        self.assertEqual(a, b)
        self.assertTrue(0 < len(a) < len(seqs), "sanity: actually subsampled")

    def test_rebalance_actually_changes_the_kept_count(self):
        seqs = [f"ACGT{i}ACGTACGTACGTACGT" for i in range(4000)]
        keep = seqpack.keep_sequence
        low = [s for s in seqs if keep(s, rate=0.05, seed=42)]
        high = [s for s in seqs if keep(s, rate=1.0, seed=42)]
        self.assertLess(len(low), len(high))
        # Nested: a higher rate keeps a superset, so raising a rare class's rate
        # only ADDS records rather than reshuffling the selection.
        self.assertTrue(set(low).issubset(set(high)))


class TestShippedConfig(unittest.TestCase):
    def test_default_ships_with_empty_class_rates(self):
        import tiara2.config as C
        cfg = C.load(ROOT / "config" / "config.yaml")
        sub = cfg["train"]["subsample"]
        self.assertEqual(sub["train"].get("class_rates"), {},
                         "must ship empty so existing caches stay valid")
        self.assertNotIn("class_rates", sub["validation"])
        self.assertNotIn("class_rates", sub["test"])

    def test_default_spec_omits_the_key(self):
        import tiara2.config as C
        cfg = C.load(ROOT / "config" / "config.yaml")
        norm = seqpack.normalize_subsample(cfg["train"]["subsample"])
        self.assertNotIn("class_rates", seqpack.split_spec(norm, "train"))


def _run():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite(
        loader.loadTestsFromTestCase(c) for c in (
            TestClassKey, TestRateResolution, TestBalanceRates,
            TestSignatureStability, TestSelectionConsistency, TestShippedConfig))
    res = unittest.TextTestRunner(verbosity=2).run(suite)
    total = res.testsRun
    bad = len(res.failures) + len(res.errors)
    print(f"\n{total - bad}/{total} passed")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    _run()

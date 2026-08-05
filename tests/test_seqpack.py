"""Tests for the 2-bit sequence pack (D2) and deterministic subsampling.

Run in a numpy-only environment (numba/Bio/torch are NOT required): the feature
builder degrades to a pure-Python k-mer counter and the pack build path uses the
byte-range FASTA reader, so this exercises the real code that runs on the box.
"""
import os
import sys
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

# The vendored training package is imported top-level (mirrors the pipeline's
# `python -m tiara.training.*` entry points).
_HERE = Path(__file__).resolve().parent
_TRAINING = _HERE.parent / "tiara" / "training"
for _p in (str(_TRAINING), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import seqpack  # noqa: E402
import featurize_cache as fc  # noqa: E402


def _write_fasta(path: Path, records):
    with path.open("w") as fh:
        for name, seq in records:
            fh.write(f">{name}\n{seq}\n")


class TestEncodeDecode(unittest.TestCase):
    def test_roundtrip_kmer_parity(self):
        # Sequences with clean ACGT, embedded N, and a lowercase run (which the
        # k-mer counter treats as a break, exactly like N).
        seqs = [
            "ACGTACGTACGTAAAACCCGGGTTT",
            "ACGTNNNACGTACGTNACGT",
            "ACGTacgtACGTACGTACGT",
            "AAAAAAAAAA",
            "ACGT",
            "N" * 12,
            "",
        ]
        for seq in seqs:
            packed, ncodes, resets = seqpack.encode_seq(seq)
            decoded = seqpack.decode_to_ascii(packed, ncodes, resets)
            for k in (2, 3, 4, 5):
                dim = 4 ** k
                raw = np.frombuffer(seq.encode("ascii", errors="ignore"),
                                    dtype=np.uint8)
                a = np.zeros(dim, dtype=np.float32)
                b = np.zeros(dim, dtype=np.float32)
                fc.count_kmers_into(raw, k, a)
                fc.count_kmers_into(np.ascontiguousarray(decoded), k, b)
                self.assertTrue(
                    np.array_equal(a, b),
                    f"kmer mismatch seq={seq!r} k={k}")


class TestKeepSequence(unittest.TestCase):
    def test_determinism(self):
        seq = "ACGT" * 300
        r1 = seqpack.keep_sequence(seq, rate=0.5, seed=7)
        r2 = seqpack.keep_sequence(seq, rate=0.5, seed=7)
        self.assertEqual(r1, r2)

    def test_rate_bounds(self):
        seq = "ACGTAAA"
        self.assertTrue(seqpack.keep_sequence(seq, rate=1.0))
        self.assertFalse(seqpack.keep_sequence(seq, rate=0.0))

    def test_min_len_and_purity_gates(self):
        short = "ACGT"
        self.assertFalse(seqpack.keep_sequence(short, rate=1.0, min_len=1000))
        dirty = "ACGT" + "N" * 96  # 4% ACGT
        self.assertFalse(
            seqpack.keep_sequence(dirty, rate=1.0, min_purity=0.9))
        clean = "ACGT" * 25
        self.assertTrue(
            seqpack.keep_sequence(clean, rate=1.0, min_purity=0.9))

    def test_case_insensitive_purity(self):
        # lowercase acgt must count toward purity the same as uppercase, so the
        # FASTA (raw) path and the train_models (upper) path agree on decisions.
        self.assertEqual(
            seqpack.keep_sequence("acgtacgt" * 20, rate=1.0, min_purity=0.9,
                                  seed=3),
            seqpack.keep_sequence("ACGTACGT" * 20, rate=1.0, min_purity=0.9,
                                  seed=3))

    def test_keeps_roughly_proportional(self):
        kept = sum(
            seqpack.keep_sequence(f"ACGT{i}ACGTACGTACGT", rate=0.3, seed=42)
            for i in range(2000))
        self.assertTrue(400 <= kept <= 800, f"kept={kept} not ~30% of 2000")


class TestNormalizeSubsample(unittest.TestCase):
    def test_disabled_returns_none(self):
        self.assertIsNone(seqpack.normalize_subsample(None))
        self.assertIsNone(seqpack.normalize_subsample({"enabled": False}))

    def test_alias_and_split_spec(self):
        cfg = {
            "enabled": True, "seed": 9,
            "train": {"rate": 0.05, "min_acgt_purity": 0.9, "min_len": 1000},
            "validation": {"rate": 1.0},
        }
        sub = seqpack.normalize_subsample(cfg)
        self.assertIsNotNone(sub)
        tr = seqpack.split_spec(sub, "train")
        self.assertEqual(tr["rate"], 0.05)
        self.assertEqual(tr["min_purity"], 0.9)
        self.assertEqual(tr["min_len"], 1000)
        self.assertEqual(tr["seed"], 9)
        # JSON-serializable so it can live in the cache signature.
        json.dumps(sub)


class TestPackFeatureParity(unittest.TestCase):
    """build_pack + _build_split_from_pack must produce byte-identical feature
    files to the direct FASTA _build_split (rate=1.0 keeps every record)."""

    def _make_inputs(self, root: Path, split: str):
        d = root / split
        d.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(0)
        bases = np.array(list("ACGT"))

        def rand_seq(n):
            return "".join(rng.choice(bases, size=n))

        _write_fasta(d / "plastids.fasta",
                     [(f"p{i}", rand_seq(rng.integers(50, 200)))
                      for i in range(15)])
        _write_fasta(d / "mitochondria.fasta",
                     [(f"m{i}", rand_seq(rng.integers(50, 200)))
                      for i in range(11)])
        _write_fasta(d / "bacteria.fasta",
                     [(f"b{i}", rand_seq(rng.integers(50, 200)))
                      for i in range(23)])
        _write_fasta(d / "archaea.fasta",
                     [(f"a{i}", rand_seq(rng.integers(50, 200)))
                      for i in range(9)])
        _write_fasta(d / "eukarya.fasta",
                     [(f"e{i}", rand_seq(rng.integers(50, 200)))
                      for i in range(17)])

    def test_parity(self):
        stage, split, tag = "first", "train", "train"
        ks = [3, 4]
        idf_by_k = {k: np.ones(4 ** k, dtype=np.float32) for k in ks}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "train_ready"
            self._make_inputs(input_dir, split)
            inputs = fc.stage_split_inputs(input_dir, stage, split)
            self.assertTrue(inputs)
            base_sig = fc._inputs_sig(inputs)

            # Reference: direct FASTA build.
            ref_root = root / "feat_ref"
            ref_shapes = fc._build_split(
                ref_root, input_dir, stage, split, ks, idf_by_k, inputs,
                base_sig, workers=1, chunk=7, log=lambda _m: None)

            # Pack build + featurize-from-pack.
            spec = seqpack.split_spec(None, split)  # rate 1.0 -> keep all
            pack_root = root / "pack"
            n_rec = seqpack.build_pack(pack_root, stage, split, inputs,
                                       base_sig, spec, workers=1,
                                       log=lambda _m: None)
            pdir = seqpack.pack_dir(pack_root, stage, split)
            pack_root_feat = root / "feat_pack"
            pack_shapes = fc._build_split_from_pack(
                pack_root_feat, pdir, stage, split, ks, idf_by_k, base_sig,
                workers=1, chunk=7, log=lambda _m: None, n_records=n_rec)

            self.assertEqual(ref_shapes, pack_shapes)
            self.assertEqual(n_rec, sum(1 for _ in fc.stage_split_inputs(
                input_dir, stage, split)) and ref_shapes[ks[0]][0])

            for k in ks:
                rk = fc._kdir(ref_root, stage, k)
                pk = fc._kdir(pack_root_feat, stage, k)
                for fname in (f"{tag}_X.f32", f"{tag}_y.i64"):
                    rb = (rk / fname).read_bytes()
                    pb = (pk / fname).read_bytes()
                    self.assertEqual(rb, pb, f"byte mismatch k={k} {fname}")

    def test_pack_resume_is_valid(self):
        stage, split = "first", "train"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "train_ready"
            self._make_inputs(input_dir, split)
            inputs = fc.stage_split_inputs(input_dir, stage, split)
            base_sig = fc._inputs_sig(inputs)
            spec = seqpack.split_spec(None, split)
            pack_root = root / "pack"
            n1 = seqpack.build_pack(pack_root, stage, split, inputs, base_sig,
                                    spec, workers=1, log=lambda _m: None)
            pdir = seqpack.pack_dir(pack_root, stage, split)
            self.assertIsNotNone(seqpack.pack_valid(pdir, base_sig, spec))
            # Second call is a cache hit and returns the same record count.
            n2 = seqpack.build_pack(pack_root, stage, split, inputs, base_sig,
                                    spec, workers=1, log=lambda _m: None)
            self.assertEqual(n1, n2)


if __name__ == "__main__":
    unittest.main(verbosity=2)

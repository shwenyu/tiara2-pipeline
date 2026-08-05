#!/usr/bin/env python3
"""Tests for the HP-search stratified row subsetting.

hyperparameter_search_gpu imports torch/skorch, which are absent from the CI
sandbox, so we extract the two NUMPY-ONLY functions with `ast` and exec them in
isolation. This keeps the test runnable anywhere while still testing the REAL
shipped source (not a copy).
"""
import ast
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "tiara" / "training" / "hyperparameter_search_gpu.py"
WANT = {"_stratified_indices", "build_hp_subset"}


def load_funcs():
    tree = ast.parse(SRC.read_text())
    ns = {"np": np, "os": os, "Path": Path}
    found = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in WANT:
            mod = ast.Module(body=[node], type_ignores=[])
            exec(compile(mod, str(SRC), "exec"), ns)
            found.add(node.name)
    missing = WANT - found
    if missing:
        raise AssertionError(f"could not extract {missing} from {SRC}")
    return ns


NS = load_funcs()
_stratified_indices = NS["_stratified_indices"]
build_hp_subset = NS["build_hp_subset"]

# Class-imbalanced, CONTIGUOUS blocks -- mirrors the real cache layout, where
# each class is written as one run of rows.
COUNTS = [50000, 30000, 12000, 5000, 3000]
DIM = 8


def make_cache(root, tag, counts, dim):
    """Write {tag}_X.f32 / {tag}_y.i64 where row i is filled with the value i,
    so a subset row proves WHICH source row it came from."""
    y = np.concatenate([np.full(c, lbl, dtype=np.int64)
                        for lbl, c in enumerate(counts)])
    n = int(y.size)
    x = np.repeat(np.arange(n, dtype=np.float32)[:, None], dim, axis=1)
    (root / f"{tag}_X.f32").write_bytes(x.tobytes())
    (root / f"{tag}_y.i64").write_bytes(y.tobytes())
    return n, y


def test_stratified_indices_preserve_proportions():
    y = np.concatenate([np.full(c, lbl, dtype=np.int64)
                        for lbl, c in enumerate(COUNTS)])
    n = int(y.size)
    cap = 10000
    idx = _stratified_indices(y, cap)

    assert idx is not None
    # ascending + unique => sequential-ish reads, no duplicated rows
    assert np.all(np.diff(idx) > 0), "indices must be strictly ascending"
    # total lands on the cap (within per-class rounding)
    assert abs(int(idx.size) - cap) <= len(COUNTS), f"size={idx.size}"

    _, c_all = np.unique(y, return_counts=True)
    _, c_sub = np.unique(y[idx], return_counts=True)
    share_all = c_all / n
    share_sub = c_sub / idx.size
    assert np.allclose(share_all, share_sub, atol=0.005), (
        f"class proportions drifted: {share_all} -> {share_sub}")
    print("OK proportions preserved:",
          np.round(share_all, 4).tolist(), "->", np.round(share_sub, 4).tolist())


def test_stratified_indices_deterministic_and_passthrough():
    y = np.concatenate([np.full(c, lbl, dtype=np.int64)
                        for lbl, c in enumerate(COUNTS)])
    a = _stratified_indices(y, 7777)
    b = _stratified_indices(y, 7777)
    assert np.array_equal(a, b), "selection must be deterministic"
    # cap disabled or already fitting => no subsetting at all
    assert _stratified_indices(y, 0) is None
    assert _stratified_indices(y, int(y.size)) is None
    assert _stratified_indices(y, int(y.size) + 1) is None
    print("OK deterministic + passthrough when cap>=n or cap==0")


def test_build_hp_subset_content_and_symlink():
    tmp = Path(tempfile.mkdtemp())
    try:
        kdir = tmp / "cache"
        kdir.mkdir()
        n_tr, y_tr = make_cache(kdir, "train", COUNTS, DIM)
        n_va, _y_va = make_cache(kdir, "val", [2000, 1500, 700, 400, 400], DIM)
        shapes = {"train": (n_tr, DIM), "val": (n_va, DIM)}

        scratch = tmp / "scratch"
        scratch.mkdir()
        cap_tr = 10000
        # val cap 0 => must fall back to SYMLINK (never copy, never delete cache)
        out = build_hp_subset(kdir, str(scratch), shapes,
                             {"train": cap_tr, "val": 0},
                             log=lambda m: print("   ", m))

        m_tr = out["train"][0]
        assert out["train"][1] == DIM
        assert abs(m_tr - cap_tr) <= len(COUNTS)
        assert out["val"] == (n_va, DIM), out["val"]

        # val must be a symlink pointing at the cache (cache is never consumed)
        assert os.path.islink(scratch / "val_X.f32")
        assert os.path.islink(scratch / "val_y.i64")
        assert not os.path.islink(scratch / "train_X.f32")
        print("OK val symlinked (cap=0), train materialized")

        # Written rows must be EXACTLY the selected source rows, in order.
        idx = _stratified_indices(y_tr, cap_tr)
        gx = np.memmap(scratch / "train_X.f32", dtype=np.float32, mode="r",
                       shape=(m_tr, DIM))
        gy = np.fromfile(scratch / "train_y.i64", dtype=np.int64)
        assert gy.size == m_tr
        assert np.array_equal(gy, y_tr[idx]), "labels misaligned with rows"
        # row i of the subset was filled with its ORIGINAL row number
        assert np.array_equal(gx[:, 0], idx.astype(np.float32)), \
            "subset rows do not match the selected source rows"
        assert np.all(gx == gx[:, :1]), "row contents corrupted"
        del gx
        print(f"OK content parity: {n_tr:,} -> {m_tr:,} rows, labels+rows aligned")

        # Rebuilding into a fresh scratch must be byte-identical (resume safe).
        scratch2 = tmp / "scratch2"
        scratch2.mkdir()
        out2 = build_hp_subset(kdir, str(scratch2), shapes,
                              {"train": cap_tr, "val": 0})
        assert out2 == out
        assert (scratch / "train_X.f32").read_bytes() == \
               (scratch2 / "train_X.f32").read_bytes()
        assert (scratch / "train_y.i64").read_bytes() == \
               (scratch2 / "train_y.i64").read_bytes()
        print("OK rebuild is byte-identical (--resume stable)")

        # Deleting scratch must NOT touch the persistent cache.
        shutil.rmtree(scratch)
        assert (kdir / "val_X.f32").is_file()
        assert (kdir / "train_X.f32").is_file()
        print("OK rmtree(scratch) leaves the persistent cache intact")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_high_dim_block_stepping():
    """dim larger than the 4M-float block must still write every row once."""
    tmp = Path(tempfile.mkdtemp())
    try:
        kdir = tmp / "cache"
        kdir.mkdir()
        dim = 4096  # k=6 width; step = (1<<22)//4096 = 1024 rows per block
        counts = [3000, 2000]
        n, y = make_cache(kdir, "train", counts, dim)
        make_cache(kdir, "val", [10, 10], dim)
        shapes = {"train": (n, dim), "val": (20, dim)}
        scratch = tmp / "s"
        scratch.mkdir()
        cap = 2500
        out = build_hp_subset(kdir, str(scratch), shapes,
                             {"train": cap, "val": 0})
        m = out["train"][0]
        idx = _stratified_indices(y, cap)
        gx = np.memmap(scratch / "train_X.f32", dtype=np.float32, mode="r",
                       shape=(m, dim))
        assert np.array_equal(gx[:, 0], idx.astype(np.float32))
        assert (scratch / "train_X.f32").stat().st_size == m * dim * 4
        del gx
        print(f"OK dim={dim} multi-block write correct ({m:,} rows)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_stratified_indices_preserve_proportions()
    test_stratified_indices_deterministic_and_passthrough()
    test_build_hp_subset_content_and_symlink()
    test_high_dim_block_stepping()
    print("\nALL HP-SUBSET TESTS PASSED")
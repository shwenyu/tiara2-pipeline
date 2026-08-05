#!/usr/bin/env python3
"""Tests for the feature-cache training path in train_models_gpu.

train_models_gpu imports torch/skorch/numba/Bio, none of which exist in every
environment, so -- exactly like tests/test_hp_subset.py -- we lift the pure
numpy helpers out of the source with ast and exec only those.

The property that matters most is LABEL ALIGNMENT: the streaming path shuffles
each window after reading it, and if that permutation were applied to X but not
to y (or drawn twice from the RNG) every model would silently train on
mismatched labels. Each synthetic row encodes its own source index, so we can
assert y_out[j] == y_all[row_id(X_out[j])] for every single row.
"""
from __future__ import annotations

import ast
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tiara" / "training" / "train_models_gpu.py"
WANT = {"_cache_kdir", "_cache_shape", "_stratified_picks", "_share",
        "build_training_matrix"}
HELPER_CLASSES = {"_ConcatMemmap"}


def load_funcs():
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    # The extracted functions run outside their module, so every global they
    # reference must be injected here by hand -- a missing one shows up as a
    # NameError at call time, not at extraction time.
    ns: dict = {"np": np, "os": os, "json": json, "Path": Path, "time": time}
    found = set()
    for node in tree.body:
        # build_training_matrix's streaming path instantiates the module-level
        # _ConcatMemmap helper, so classes have to be lifted too -- extracting
        # functions alone left it undefined and every streaming test blew up
        # with a NameError long before it could check label alignment.
        if isinstance(node, ast.ClassDef) and node.name in HELPER_CLASSES:
            exec(compile(ast.Module(body=[node], type_ignores=[]),
                         str(SRC), "exec"), ns)
            continue
        if isinstance(node, ast.FunctionDef) and node.name in WANT:
            exec(compile(ast.Module(body=[node], type_ignores=[]),
                         str(SRC), "exec"), ns)
            found.add(node.name)
    missing = WANT - found
    assert not missing, f"could not extract {sorted(missing)} from {SRC}"
    return ns


F = load_funcs()
DIM = 8
COUNTS = {0: 500, 1: 300, 3: 120, 4: 80}  # class-block ordered, like the cache


def make_cache(tmp: Path, counts=None, dim: int = DIM):
    """Write a synthetic <stage>/k<k> cache; row i is filled with the value i."""
    counts = counts or COUNTS
    kdir = tmp / "first" / "k4"
    kdir.mkdir(parents=True, exist_ok=True)
    y = np.concatenate([np.full(c, lbl, dtype=np.int64)
                        for lbl, c in counts.items()])
    n = int(y.size)
    X = np.repeat(np.arange(n, dtype=np.float32)[:, None], dim, axis=1)
    X.tofile(kdir / "train_X.f32")
    y.tofile(kdir / "train_y.i64")
    (kdir / "meta.json").write_text(json.dumps(
        {"train": {"n": n, "dim": dim, "inputs_sig": "synthetic"}}))
    return kdir, y


def row_ids(X) -> np.ndarray:
    return np.asarray(X[:, 0], dtype=np.int64)


def check_alignment(X, y_out, y_all, what: str) -> None:
    ids = row_ids(X)
    assert ids.min() >= 0 and ids.max() < y_all.size, f"{what}: row id out of range"
    bad = int(np.count_nonzero(y_all[ids] != y_out))
    assert bad == 0, f"{what}: {bad} rows carry a label that is not their own"
    # every column of a row holds the same id -> no torn/misaligned reads
    assert np.all(np.asarray(X) == ids[:, None].astype(np.float32)), \
        f"{what}: row contents are torn"


def test_ram_path_uncapped():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        kdir, y_all = make_cache(tmp)
        X, y, shuffle = F["build_training_matrix"](
            kdir, tmp / "scratch", 0, 1 << 30)
        assert shuffle is True, "in-RAM path must let the loader shuffle"
        assert X.shape == (y_all.size, DIM), X.shape
        assert np.array_equal(y, y_all)
        check_alignment(X, y, y_all, "ram/uncapped")
        assert not (tmp / "scratch").exists(), "RAM path must not write scratch"
    print("OK ram path: all 1,000 rows, no scratch copy written")


def test_ram_path_capped_preserves_class_share():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        kdir, y_all = make_cache(tmp)
        X, y, shuffle = F["build_training_matrix"](
            kdir, tmp / "scratch", 200, 1 << 30)
        assert shuffle is True
        assert 190 <= y.size <= 210, y.size
        before = dict(zip(*[a.tolist() for a in np.unique(y_all, return_counts=True)]))
        after = dict(zip(*[a.tolist() for a in np.unique(y, return_counts=True)]))
        assert set(before) == set(after), "a class disappeared from the subset"
        for lbl in before:
            got = after[lbl] / y.size
            want = before[lbl] / y_all.size
            assert abs(got - want) < 0.02, f"class {lbl}: {got:.4f} vs {want:.4f}"
        check_alignment(X, y, y_all, "ram/capped")
    print("OK ram path: cap 1,000 -> ~200 rows, class shares preserved")


def test_streaming_path_alignment_and_shuffle():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        kdir, y_all = make_cache(tmp)
        scratch = tmp / "scratch"
        # ram_budget=0 forces the memmap path regardless of size
        X, y, shuffle = F["build_training_matrix"](kdir, scratch, 400, 0)
        assert shuffle is False, "streamed data is pre-shuffled on disk"
        assert isinstance(X, np.memmap), type(X)
        assert (scratch / "train_X.f32").is_file()
        assert X.shape[0] == y.size
        assert 380 <= y.size <= 420, y.size
        # THE critical property: X and y were permuted together
        check_alignment(X, y, y_all, "stream")
        # and the on-disk order really is shuffled, not class-blocked
        assert not np.array_equal(np.sort(row_ids(X)), row_ids(X)), \
            "streamed rows are still in ascending (class-blocked) order"
    print("OK stream path: labels travel with their rows, order is shuffled")


def test_streaming_matches_ram_selection():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        kdir, _ = make_cache(tmp)
        Xr, yr, _ = F["build_training_matrix"](kdir, tmp / "s1", 300, 1 << 30)
        Xs, ys, _ = F["build_training_matrix"](kdir, tmp / "s2", 300, 0)
        assert np.array_equal(np.sort(row_ids(Xr)), np.sort(row_ids(Xs))), \
            "RAM and streamed paths selected different rows"
        assert np.array_equal(np.sort(yr), np.sort(ys))
    print("OK both paths select the identical row set (only the order differs)")


def test_selection_is_deterministic_for_resume():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        kdir, _ = make_cache(tmp)
        a, ya, _ = F["build_training_matrix"](kdir, tmp / "a", 250, 1 << 30)
        b, yb, _ = F["build_training_matrix"](kdir, tmp / "b", 250, 1 << 30)
        assert np.array_equal(row_ids(a), row_ids(b))
        assert np.array_equal(ya, yb)
        # the streamed path is seeded too
        c, yc, _ = F["build_training_matrix"](kdir, tmp / "c", 250, 0)
        d, yd, _ = F["build_training_matrix"](kdir, tmp / "d", 250, 0)
        assert np.array_equal(row_ids(c), row_ids(d))
        assert np.array_equal(yc, yd)
    print("OK selection + shuffle are deterministic (--resume repeats them)")


def test_cap_larger_than_data_keeps_everything():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        kdir, y_all = make_cache(tmp)
        X, y, _ = F["build_training_matrix"](kdir, tmp / "s", 10_000_000, 1 << 30)
        assert y.size == y_all.size
        assert np.array_equal(row_ids(X), np.arange(y_all.size))
    print("OK cap above the row count is a no-op")


def test_streaming_uncapped_still_aligned():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        kdir, y_all = make_cache(tmp)
        X, y, shuffle = F["build_training_matrix"](kdir, tmp / "s", 0, 0)
        assert shuffle is False
        assert y.size == y_all.size, "uncapped stream must keep every row"
        check_alignment(X, y, y_all, "stream/uncapped")
    print("OK uncapped stream path keeps every row, still aligned")


def test_corrupt_and_missing_inputs_raise():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        kdir, _ = make_cache(tmp)

        empty = tmp / "second" / "k9"
        empty.mkdir(parents=True)
        try:
            F["build_training_matrix"](empty, tmp / "s", 0, 1 << 30)
        except FileNotFoundError as exc:
            assert "meta" in str(exc).lower()
        else:
            raise AssertionError("missing meta.json must raise")

        # a truncated matrix must be caught, never silently reshaped
        bad = tmp / "bad" / "k4"
        bad.mkdir(parents=True)
        shutil.copy(kdir / "meta.json", bad / "meta.json")
        shutil.copy(kdir / "train_y.i64", bad / "train_y.i64")
        (bad / "train_X.f32").write_bytes(
            (kdir / "train_X.f32").read_bytes()[:-64])
        try:
            F["build_training_matrix"](bad, tmp / "s", 0, 1 << 30)
        except ValueError as exc:
            assert "bytes" in str(exc)
        else:
            raise AssertionError("truncated train_X.f32 must raise")
    print("OK missing meta.json and truncated matrices are rejected")


def test_progress_is_emitted_and_can_be_silenced():
    """Both paths must report progress; silence here means a hung-looking run."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        kdir, y_all = make_cache(tmp)
        for name, budget in (("ram", 1 << 30), ("stream", 0)):
            msgs: list[str] = []
            X, y, _ = F["build_training_matrix"](
                kdir, tmp / f"p_{name}", 0, budget,
                log=msgs.append, progress_secs=1e-9)
            bars = [m for m in msgs if m.startswith(("load ", "write "))]
            assert bars, f"{name}: emitted no progress lines"
            assert any("ETA" in m and "MiB/s" in m for m in bars), \
                f"{name}: progress lines lack rate/ETA"
            # progress must never disturb the data itself
            check_alignment(X, y, y_all, f"progress/{name}")

            quiet: list[str] = []
            F["build_training_matrix"](kdir, tmp / f"q_{name}", 0, budget,
                                       log=quiet.append, progress_secs=0)
            assert not [m for m in quiet
                        if m.startswith(("load ", "write "))], \
                f"{name}: progress_secs=0 did not silence the bar"
    print("OK progress lines on both paths, silenced at progress_secs=0")


def test_helpers():
    assert str(F["_cache_kdir"]("/c", "second", 7)).endswith("second/k7")
    y = np.array([0, 0, 0, 1], dtype=np.int64)
    assert F["_share"](y) == [0.75, 0.25]
    assert F["_share"](np.array([], dtype=np.int64)) == []
    assert F["_stratified_picks"](y, 0) is None, "cap=0 means no cap"
    assert F["_stratified_picks"](y, 99) is None, "cap above n means no cap"
    picks = F["_stratified_picks"](np.concatenate(
        [np.zeros(100, np.int64), np.ones(100, np.int64)]), 50)
    assert picks is not None and len(picks) == 2
    assert all(np.all(np.diff(p) > 0) for p in picks), "picks must be ascending"
    print("OK helpers: kdir, share, cap edge cases")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nALL MODEL-MATRIX TESTS PASSED")

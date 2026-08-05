#!/usr/bin/env python3
"""End-to-end tests for the fragment bp-balancing stage (v2.1.3).

These lock down the four defects that produced the v2.1.2 model collapse:

  1. only `train` was balanced, so the fit saw a 98.6%-eukarya validation set;
  2. `eval_mode: symlink` mirrored splits that were supposed to be balanced,
     so a "balanced" directory could actually be a symlink to the raw corpus;
  3. a re-run over an existing symlinked split silently wrote through the link
     into the source corpus;
  4. the report had no per-split structure, so nobody could see any of it.

The sampler is stdlib-only, so it runs here for real on a tiny synthetic
corpus -- no mocks, no fixtures.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "scripts" / "sample_fragments_by_bp.py"
CLASSES = ["archaea", "bacteria", "eukarya", "mitochondria", "plastids"]


def load_sampler():
    # The sampler fans out over a process pool, and the children re-import the
    # module by name to unpickle the worker function. Loading it purely from a
    # file path leaves it un-importable in the children, so put both the repo
    # root and scripts/ on sys.path and register it in sys.modules.
    for entry in (str(ROOT), str(ROOT / "scripts")):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    name = "sample_fragments_by_bp"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SRC)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


M = load_sampler()


def write_fasta(path: Path, count: int, length: int, tag: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for i in range(count):
            handle.write(f">GCA_{tag}{i:06d}.1|{i} sg=opisthokonta-fungi\n")
            handle.write(("ACGT" * (length // 4)) + "\n")


def make_corpus(root: Path, *, euk_heavy: bool = True) -> None:
    """train and validation are both eukarya-dominated, like the real corpus."""
    for split in ("train", "validation", "test"):
        for cls in CLASSES:
            n = 400 if (euk_heavy and cls == "eukarya") else 40
            write_fasta(root / split / f"{cls}.fasta", n, 2000, f"{split[:2]}{cls[:2]}")


def run_sampler(src: Path, out: Path, *, splits: str = "train,validation",
                extra: list[str] | None = None) -> int:
    # No --target-total-bp on purpose: the sampler then uses the largest budget
    # it can hit WITHOUT oversampling any class. Asking for more than the
    # rarest class can supply is a hard error, which is itself correct
    # behaviour (v2.1.2 quietly let archaea saturate at rate 1.0).
    argv = [
        "sample_fragments_by_bp.py",
        "--source-root", str(src),
        "--out-root", str(out),
        "--balance-splits", splits,
        "--seed", "42",
        "--min-len", "1000",
        "--min-acgt-purity", "0.9",
        "--eval-mode", "symlink",
        "--workers", "2",
        "--euk-workers", "2",
        "--io-buffer-mb", "1",
        "--force",
    ] + (extra or [])
    old = sys.argv
    sys.argv = argv
    try:
        return int(M.main() or 0)
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        # SystemExit("message") means failure with a diagnostic string
        return code if isinstance(code, int) else 1
    finally:
        sys.argv = old


def fasta_count(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.startswith(">"))


def test_parse_splits_refuses_test():
    assert M.parse_splits("train,validation") == ["train", "validation"]
    assert M.parse_splits("train") == ["train"]
    for bad in ("test", "train,test", ""):
        try:
            M.parse_splits(bad)
        except (SystemExit, ValueError, Exception):
            continue
        raise AssertionError(f"parse_splits accepted {bad!r}")
    print("ok: test split can never be balanced away")


def test_both_splits_are_balanced_and_test_is_mirrored():
    tmp = Path(tempfile.mkdtemp())
    try:
        src, out = tmp / "src", tmp / "out"
        make_corpus(src)
        assert run_sampler(src, out) == 0

        # 1) both fitted splits exist as REAL directories with real files
        for split in ("train", "validation"):
            d = out / split
            assert d.is_dir() and not d.is_symlink(), f"{split} is not a real dir"
            for cls in CLASSES:
                f = d / f"{cls}.fasta"
                assert f.exists() and not f.is_symlink(), f"{split}/{cls}"

        # 2) the eukarya landslide is gone from BOTH splits, which is the whole
        #    point: v2.1.2 balanced train and left validation at 98.6% eukarya.
        for split in ("train", "validation"):
            counts = {c: fasta_count(out / split / f"{c}.fasta") for c in CLASSES}
            total = sum(counts.values())
            assert total > 0, f"{split} is empty"
            share = counts["eukarya"] / total
            assert share < 0.75, f"{split} eukarya share {share:.3f} still dominant"

        # 3) test is mirrored, never resampled -- it is the only unbiased set
        assert (out / "test").is_symlink(), "test split must stay a mirror"
        assert (out / "test").resolve() == (src / "test").resolve()

        print("ok: train+validation balanced, test mirrored untouched")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_report_has_per_split_schema_v2():
    tmp = Path(tempfile.mkdtemp())
    try:
        src, out = tmp / "src", tmp / "out"
        make_corpus(src)
        assert run_sampler(src, out) == 0

        plan = json.loads((out / "sampling_plan.json").read_text())
        assert plan["schema"] == 2, plan.get("schema")
        assert set(plan["splits"]) == {"train", "validation"}, plan["splits"].keys()
        assert plan["balanced_splits"] == ["train", "validation"]
        assert plan["eval_mode"] == "symlink"
        assert plan["mirrored_splits"] == ["test"], plan["mirrored_splits"]

        # train stays mirrored at the top level so older readers (run-report,
        # dashboards) keep working against schema 2.
        for key in ("target_total_bp", "strata"):
            assert key in plan, key
            assert plan[key] == plan["splits"]["train"][key], key

        rows = (out / "sampling_report.tsv").read_text().splitlines()
        header = rows[0].split("\t")
        assert header[0] == "split", header
        seen = {r.split("\t")[0] for r in rows[1:] if r.strip()}
        assert seen == {"train", "validation"}, seen
        print("ok: schema 2 report carries both splits")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_rerun_replaces_stale_symlink_instead_of_writing_through_it():
    """A v2.1.2 output dir has validation/ as a symlink into the raw corpus.

    Writing through that link would corrupt the source corpus in place. The
    sampler must unlink it and materialise a real directory.
    """
    tmp = Path(tempfile.mkdtemp())
    try:
        src, out = tmp / "src", tmp / "out"
        make_corpus(src)
        out.mkdir(parents=True)
        (out / "validation").symlink_to(src / "validation", target_is_directory=True)
        before = fasta_count(src / "validation" / "eukarya.fasta")

        assert run_sampler(src, out) == 0

        assert not (out / "validation").is_symlink(), "stale link was written through"
        assert (out / "validation").is_dir()
        after = fasta_count(src / "validation" / "eukarya.fasta")
        assert after == before, "source corpus was modified through the symlink"
        print("ok: stale symlink replaced, source corpus untouched")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_out_root_may_not_alias_source_root():
    tmp = Path(tempfile.mkdtemp())
    try:
        src = tmp / "src"
        make_corpus(src)
        alias = tmp / "alias"
        alias.symlink_to(src, target_is_directory=True)
        assert run_sampler(src, alias) != 0, "aliased out-root must be refused"
        print("ok: out-root aliasing source-root is refused")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_require_clades_fails_but_still_writes_the_report():
    """With no taxdump the euk clade split is unresolved; --require-clades must
    fail the run, but only AFTER the report is on disk so the failure is
    diagnosable."""
    tmp = Path(tempfile.mkdtemp())
    try:
        src, out = tmp / "src", tmp / "out"
        make_corpus(src)
        code = run_sampler(src, out, extra=["--require-clades"])
        assert code != 0, "unresolved clades must fail the run"
        assert (out / "sampling_plan.json").exists(), "report must survive the failure"
        plan = json.loads((out / "sampling_plan.json").read_text())
        assert "euk_resolution" in plan or "splits" in plan
        print("ok: require-clades fails loudly and leaves evidence")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("all bp-balance tests passed")

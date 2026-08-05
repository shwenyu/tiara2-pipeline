"""Sandbox tests for featurize_cache (numpy-only, numba-absent path)."""
import random
import sys
from pathlib import Path

import numpy as np

# Import as a TOP-LEVEL module (put its dir on sys.path) so the test does not
# trigger tiara/__init__ (which imports tqdm/torch, absent in the CI sandbox)
# AND so forked Pool workers can re-import it by name when unpickling tasks.
# featurize_cache itself has no tiara import at module load time.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tiara" / "training"))
import featurize_cache as fc

random.seed(7)


def _rand_seq(n):
    return "".join(random.choice("ACGTN") for _ in range(n))


def _write_fasta(path, seqs, linewrap=0):
    with open(path, "w") as fh:
        for i, s in enumerate(seqs):
            fh.write(f">rec{i}\n")
            if linewrap:
                for j in range(0, len(s), linewrap):
                    fh.write(s[j:j + linewrap] + "\n")
            else:
                fh.write(s + "\n")


def _read_all(path):
    seqs, cur, hdr = [], [], False
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if hdr:
                    seqs.append("".join(cur))
                hdr = True
                cur = []
            elif hdr:
                cur.append(line.strip())
    if hdr:
        seqs.append("".join(cur))
    return seqs


def test_shard_parser_partition(tmp):
    tmp.mkdir(parents=True, exist_ok=True)
    f = tmp / "x.fasta"
    seqs = [_rand_seq(random.randint(1, 400)) for _ in range(137)]
    _write_fasta(f, seqs, linewrap=60)
    assert _read_all(f) == seqs, "reference parser mismatch"
    size = f.stat().st_size
    for nshards in (1, 2, 3, 5, 8, 13, 100, 500):
        step = max(1, size // nshards)
        bounds = sorted(set(list(range(0, size, step)) + [0, size]))
        got = []
        for a, b in zip(bounds, bounds[1:]):
            got.extend(list(fc.iter_fasta_shard(str(f), a, b)))
        assert got == seqs, f"partition broke at nshards={nshards}: {len(got)} vs {len(seqs)}"
    print("OK shard parser partition (no dup / no gap) across many boundaries")

    # Per-record raw byte accounting: every byte belongs to exactly one record,
    # so summing raw_bytes over the whole file equals the file size. This is
    # what drives the byte-based (smooth) progress bar.
    raw = list(fc._iter_fasta_shard_raw(str(f), 0, size))
    assert [s for s, _ in raw] == seqs, "raw reader sequence mismatch"
    assert sum(nb for _, nb in raw) == size, (sum(nb for _, nb in raw), size)
    print("OK per-record raw byte accounting sums to file size")


def _reference(groups, k, idf, dim):
    flat = [s for g in groups for s in g]
    return fc.featurize_block(flat, k, idf, dim)


def test_build_and_reuse(tmp):
    root = tmp / "tr"
    (root / "train").mkdir(parents=True)
    (root / "validation").mkdir(parents=True)
    plast = [_rand_seq(random.randint(1, 300)) for _ in range(90)]
    mito = [_rand_seq(random.randint(1, 300)) for _ in range(40)]
    _write_fasta(root / "train" / "plastids.fasta", plast, linewrap=70)
    _write_fasta(root / "train" / "mitochondria.fasta", mito, linewrap=70)
    vp = [_rand_seq(random.randint(1, 200)) for _ in range(20)]
    vm = [_rand_seq(random.randint(1, 200)) for _ in range(12)]
    _write_fasta(root / "validation" / "plastids.fasta", vp, linewrap=70)
    _write_fasta(root / "validation" / "mitochondria.fasta", vm, linewrap=70)

    ks = [1, 2, 3]
    idf_map = {k: np.ones(4 ** k, dtype=np.float32) for k in ks}
    cache = tmp / "cache"

    blog = []
    shapes = fc.ensure_features(cache, root, "second", ks, idf_map=idf_map,
                                workers=3, chunk=7, progress_secs=0.0,
                                log=blog.append)
    for k in ks:
        assert shapes["train"][k] == (130, 4 ** k), shapes["train"][k]
        assert shapes["val"][k] == (32, 4 ** k), shapes["val"][k]

    prog = [m for m in blog if "read+featurize" in m]
    assert prog, f"no progress lines emitted: {blog}"
    assert all("%" in m and "shards" in m and "seqs" in m for m in prog), prog
    assert any("100.0%" in m for m in prog), prog
    # bytes must actually move (not stuck at 0B) -- the final line reports the
    # full byte total, proving the byte-based bar advanced rather than only the
    # seq counter.
    assert not all("| 0B/" in m for m in prog), prog
    print(f"OK progress output emitted during parallel build ({len(prog)} lines)")

    for k in ks:
        dim = 4 ** k
        kdir = cache / "second" / f"k{k}"
        X = np.frombuffer((kdir / "train_X.f32").read_bytes(), dtype=np.float32).reshape(130, dim)
        y = np.frombuffer((kdir / "train_y.i64").read_bytes(), dtype=np.int64)
        ref = _reference([plast, mito], k, idf_map[k], dim)
        assert np.allclose(X, ref, atol=1e-6), f"feature mismatch k={k}"
        assert list(y) == [0] * 90 + [2] * 40, f"label mismatch k={k}"
    print("OK build parity (features + labels) for k=1,2,3, workers=3")

    logs2 = []
    fc.ensure_features(cache, root, "second", ks, idf_map=idf_map,
                       workers=3, chunk=7, log=logs2.append)
    assert any("all k cached" in m for m in logs2), logs2
    assert not any("building k=" in m for m in logs2), logs2
    print("OK reuse: fully cached, nothing rebuilt")

    logs3 = []
    idf_map[4] = np.ones(4 ** 4, dtype=np.float32)
    fc.ensure_features(cache, root, "second", [1, 2, 3, 4], idf_map=idf_map,
                       workers=2, chunk=5, log=logs3.append)
    assert any("building k=[4]" in m for m in logs3), logs3
    print("OK partial reuse: only the new k is built")

    plast2 = plast + [_rand_seq(50)]
    _write_fasta(root / "train" / "plastids.fasta", plast2, linewrap=70)
    logs4 = []
    shapes4 = fc.ensure_features(cache, root, "second", [2], idf_map=idf_map,
                                 workers=2, chunk=5, log=logs4.append)
    assert shapes4["train"][2] == (131, 16), shapes4["train"][2]
    assert any("building k=[2]" in m and "second/train" in m for m in logs4), logs4
    print("OK invalidation: input size change forces rebuild (90->91 plastids)")

    cache2 = tmp / "cache1"
    fc.ensure_features(cache2, root, "second", [2], idf_map=idf_map,
                       workers=1, chunk=1000, log=lambda m: None)
    kdir = cache2 / "second" / "k2"
    X1 = np.frombuffer((kdir / "train_X.f32").read_bytes(), dtype=np.float32).reshape(131, 16)
    ref1 = _reference([plast2, mito], 2, idf_map[2], 16)
    assert np.allclose(X1, ref1, atol=1e-6)
    print("OK workers=1 path parity")


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        test_shard_parser_partition(tmp / "a")
        b = tmp / "b"
        b.mkdir()
        test_build_and_reuse(b)
    print("\nALL FEATURIZE-CACHE TESTS PASSED")

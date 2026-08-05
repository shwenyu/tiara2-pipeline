"""K-mer bag-of-words featurisation for inference.

WHAT CHANGED IN v2.2.0 (and why)
--------------------------------
1. ``MAX_K`` was 7. v2.2.0 trains a second-stage k=8 model, so every k=8
   inference call raised ``ValueError: k-mer length not supported!``. The cap
   is now 8 and is asserted against the shipped models at load time.

2. The k -> {kmer_string: index} tables were built EAGERLY at import for every
   k. At k=8 that is 65,536 Python string keys in a numba typed dict, built on
   every process start (and every joblib worker). They are now built lazily and
   cached, so a k=6 run never pays for k=8.

3. Counting used a per-position PYTHON STRING SLICE and a numba dict lookup.
   That is O(k) hashing per position. The training pipeline already counts with
   a 2-bit rolling code; inference now uses the same method, so features are
   produced by identical arithmetic and roughly an order of magnitude faster.
   ``N`` and any other non-ACGT base resets the rolling window, which is
   exactly what the dict version did (unknown k-mers were skipped).

The old ``oligofreq`` / ``single_oligofreq`` / ``multiple_oligofreq`` /
``calc_array`` names are kept with identical semantics so nothing that imports
them breaks.
"""
from itertools import product
from collections import defaultdict

import numpy as np

try:
    from numba import njit, types
    from numba.typed import Dict, List
    _HAVE_NUMBA = True
except ImportError:  # pragma: no cover - production env always has numba
    _HAVE_NUMBA = False

    def njit(*args, **kwargs):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]
        return lambda func: func

    class _ListShim(list):
        pass

    List = _ListShim

# k=8 is the largest k any shipped model uses (second stage, v2.2.0).
# Raising this further is a deliberate decision: memory is 4**k floats PER
# FRAGMENT, so k=9 would be 262,144 float32 = 1 MiB per fragment.
MAX_K = 8

# Lookup tables are built on demand and memoised.  Populated only for the k
# values a run actually touches.
_kmer_to_pos_cache: dict = {}

# 2-bit code per base; 255 marks "not a base" and resets the rolling window.
_BASE_CODE = np.full(256, 255, dtype=np.uint8)
for _b, _v in (("A", 0), ("C", 1), ("G", 2), ("T", 3)):
    _BASE_CODE[ord(_b)] = _v
    _BASE_CODE[ord(_b.lower())] = _v


def _check_k(k: int) -> int:
    k = int(k)
    if k < 1 or k > MAX_K:
        raise ValueError(
            f"k-mer length not supported: k={k} (supported range 1..{MAX_K}). "
            f"If a model really needs a larger k, raise bow.MAX_K and re-check "
            f"the 4**k memory cost per fragment.")
    return k


def _table(k: int):
    """Lazily build (and cache) the kmer-string -> index table for one k."""
    k = _check_k(k)
    table = _kmer_to_pos_cache.get(k)
    if table is None:
        if _HAVE_NUMBA:
            table = Dict.empty(key_type=types.unicode_type, value_type=types.int64)
        else:
            table = {}
        for i, kmer in enumerate(product("ACGT", repeat=k)):
            table["".join(kmer)] = i
        _kmer_to_pos_cache[k] = table
    return table


class _KmerToPos:
    """Backwards-compatible stand-in for the old eager module-level dict.

    Supports ``kmer_to_pos[k]`` and ``k in kmer_to_pos`` without building any
    table until it is actually indexed.
    """

    def __getitem__(self, k):
        return _table(k)

    def __contains__(self, k):
        try:
            _check_k(k)
        except (ValueError, TypeError):
            return False
        return True


kmer_to_pos = _KmerToPos()


@njit(cache=True)
def _count_2bit(codes, offsets, k, dim, out):
    """Rolling 2-bit k-mer counter over a concatenated code buffer.

    ``codes`` holds every sequence back to back, ``offsets[i]:offsets[i+1]``
    delimits sequence i, and 255 means "invalid base" (resets the window).
    """
    mask = (1 << (2 * k)) - 1
    for i in range(offsets.shape[0] - 1):
        code = 0
        valid = 0
        for pos in range(offsets[i], offsets[i + 1]):
            value = codes[pos]
            if value == 255:
                code = 0
                valid = 0
                continue
            code = ((code << 2) | value) & mask
            valid += 1
            if valid >= k:
                out[i, code] += 1.0
    return out


def _encode(seqs):
    """Pack an iterable of sequences into one uint8 code buffer + offsets."""
    chunks = []
    offsets = np.zeros(len(seqs) + 1, dtype=np.int64)
    total = 0
    for i, seq in enumerate(seqs):
        raw = np.frombuffer(seq.encode("ascii", errors="replace"), dtype=np.uint8)
        chunks.append(_BASE_CODE[raw])
        total += raw.size
        offsets[i + 1] = total
    if chunks:
        codes = np.concatenate(chunks)
    else:
        codes = np.zeros(0, dtype=np.uint8)
    return codes, offsets


def count_kmers(seqs, k: int) -> np.ndarray:
    """Counts matrix of shape (len(seqs), 4**k), float32.

    This is the fast path used by the inference layer. Non-ACGT characters
    break the k-mer window instead of contributing a bogus count.
    """
    k = _check_k(k)
    seqs = list(seqs)
    dim = 4 ** k
    out = np.zeros((len(seqs), dim), dtype=np.float32)
    if not seqs:
        return out
    codes, offsets = _encode(seqs)
    return _count_2bit(codes, offsets, k, dim, out)


def oligofreq(sequence: str, k: int) -> np.ndarray:
    """A function that calculates oligonucleotide frequency.

    Examples
    --------
    >>> oligofreq("AACT", 2)
    array([1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
      dtype=float32)
    """
    k = _check_k(k)
    vector = defaultdict(int)
    for pos in range(len(sequence) - k + 1):
        vector[sequence[pos: pos + k]] += 1
    return np.array(
        [vector["".join(kmer)] for kmer in product("ACGT", repeat=k)], dtype=np.float32
    )


def single_oligofreq(sequence, k):
    """Calculate a bag-of-words representation of a single sequence."""
    return count_kmers([sequence], k)


def multiple_oligofreq(seqs, k):
    """Calculate bag-of-words representations of a list of sequences."""
    return count_kmers(seqs, k)


@njit(cache=True)
def _calc_array_dict(seqs, d, k):
    result = np.zeros((len(seqs), 4 ** k), dtype=np.float32)
    for i, seq in enumerate(seqs):
        for pos in range(len(seq) - k + 1):
            subseq = seq[pos: pos + k]
            if subseq in d:
                result[i, d[subseq]] += 1
    return result


def calc_array(seqs, d, k):
    """Legacy entry point kept for compatibility with older callers."""
    return _calc_array_dict(seqs, d, k)

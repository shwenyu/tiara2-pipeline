"""Shared helpers for the Tiara2 corpus pipeline.

Central design ideas
--------------------
1. Every fragment has a stable ``fragment_id`` (first FASTA header token).
2. All stages operate on IDs + a metadata table; sequences are materialized
   only when a tool (MMseqs) needs them. This avoids repeated multi-TiB FASTA
   rewrites.
3. Length binning is exact for double-95: two fragments can only be a
   double-95 duplicate if min(len)/max(len) >= MIN_COV, so fragments in
   different (non-overlapping, guard-banded) length bins can never collide.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

SPLIT_PRIORITY = {"test": 0, "validation": 1, "train": 2}


def higher_priority(a: str, b: str) -> bool:
    """True if split ``a`` outranks split ``b`` (test > validation > train)."""
    return SPLIT_PRIORITY[a] < SPLIT_PRIORITY[b]


def iter_fasta(path):
    """Yield (fragment_id, header, sequence_lines) preserving raw sequence lines."""
    header = None
    seq = []
    with open(path) as fh:
        for raw in fh:
            line = raw.rstrip("\r\n")
            if line.startswith(">"):
                if header is not None:
                    yield header.split(None, 1)[0], header, seq
                header = line[1:]
                seq = []
            elif line.strip():
                seq.append(line.strip())
    if header is not None:
        yield header.split(None, 1)[0], header, seq


@dataclass(frozen=True)
class LengthBinner:
    """Assigns a fragment length to a bin key.

    Two modes:

    * ``discrete``: exact T6 cut lengths (e.g. 1000/2000/3000/5000/10000).
      Each length is its own bin. Lossless because the discrete lengths are all
      more than 5% apart, so cross-bin double-95 duplicates are impossible.
    * ``geometric``: variable-length fragments. Bins are geometric with factor
      ``r`` and a guard band, so any two lengths within ``min_cov`` ratio share
      at least one comparison group. Use ``bins_to_search`` to include the
      neighbouring bin on each side.
    """

    mode: str
    min_cov: float = 0.95
    discrete_lengths: tuple = (1000, 2000, 3000, 5000, 10000)
    tolerance: int = 0  # allowed +/- deviation from a discrete length
    varlen_single: bool = True  # non-fixed fragments -> ONE shared "VAR" bin
    varlen_bin: str = "VAR"

    def bin_key(self, length: int) -> str:
        if self.mode == "discrete":
            for target in self.discrete_lengths:
                if abs(length - target) <= self.tolerance:
                    return f"L{target}"
            # Non-fixed fragment. Per design decision, collect ALL such
            # fragments into a single varlen bin (deduped with short-cov +
            # double-95), rather than many geometric sub-bins. Overlap odds
            # here are low, so one bin keeps the search cheap and complete.
            if self.varlen_single:
                return self.varlen_bin
            return f"G{self._geo_index(length)}"
        if self.mode == "geometric":
            return f"G{self._geo_index(length)}"
        raise ValueError(f"unknown binner mode: {self.mode}")

    def _geo_index(self, length: int) -> int:
        # Bin factor chosen so one bin spans a <=min_cov length ratio.
        r = 1.0 / self.min_cov
        return int(math.floor(math.log(max(length, 1)) / math.log(r)))

    def bins_to_search(self, bin_key: str):
        """Target bins to compare a query bin against.

        Discrete bins are self-contained (exact). Geometric bins must also look
        at the two neighbours to cover the guard band at bin boundaries.
        """
        # Fixed-length bins (L*) and the single varlen bin (VAR) are
        # self-contained. VAR is complete on its own because it is deduped
        # with cov-mode 5 (short-seq coverage), which catches containment
        # regardless of length disparity within the bin.
        if bin_key.startswith("L") or bin_key == self.varlen_bin:
            return [bin_key]
        idx = int(bin_key[1:])
        return [f"G{idx - 1}", f"G{idx}", f"G{idx + 1}"]


def dedup_params_for_bin(bin_key: str, cfg: dict) -> dict:
    """Return MMseqs coverage parameters for a bin.

    * Fixed-length bins (``L*``): symmetric double-95 (cov-mode 0), because all
      members share one length so both query and target coverage are
      meaningful.
    * Varlen bin (``VAR``): cov-mode 5 (short-seq coverage) so a short fragment
      fully contained in a longer one is still flagged as a duplicate. This is
      the "double cov + short cov" strategy for the high-variation bin.
    """
    d = cfg["dedup"]
    if bin_key == d.get("varlen_bin", "VAR"):
        v = d.get("varlen", {})
        return {
            "min_seq_id": v.get("min_seq_id", d["min_seq_id"]),
            "cov": v.get("min_cov", d["min_cov"]),
            "cov_mode": v.get("cov_mode", 5),
        }
    return {
        "min_seq_id": d["min_seq_id"],
        "cov": d["min_cov"],
        "cov_mode": d.get("cov_mode", 0),
    }


def can_be_duplicate(len_a: int, len_b: int, min_cov: float = 0.95) -> bool:
    """Necessary length condition for a double-95 duplicate (cov-mode 0)."""
    lo, hi = sorted((len_a, len_b))
    return hi > 0 and lo / hi >= min_cov

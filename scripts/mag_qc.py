#!/usr/bin/env python3
"""MAG decontamination QC (Model B / future rounds; NOT used this round).

MAG assemblers only apply within-kingdom QC, so a prokaryotic MAG can carry
eukaryotic contigs yet still be labelled prok. We assign each MAG contig a
kingdom by aligning it against a QC'd isolate reference DB, then discard MAGs
whose foreign fraction exceeds a threshold.

Key design choices
------------------
* Weight by base pairs, not contig count: one large foreign contig matters more
  than several tiny ones.
* Separate three outcomes per contig: matches own kingdom / matches a foreign
  kingdom / unassigned (no confident hit). Unassigned is NOT counted as
  contamination, only tracked, to avoid discarding novel-but-valid clades.
* A MAG is discarded if foreign_bp / assigned_bp >= foreign_threshold.

Input: MMseqs best-hit TSV of MAG contigs (query) vs isolate DB (target) with
format 'query,target,fident,qcov,tcov,tset' where tset carries the target
kingdom label; plus a contig-length TSV (contig_id, mag_id, mag_kingdom, len).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--contigs", required=True,
                   help="TSV: contig_id, mag_id, mag_kingdom, length")
    p.add_argument("--hits", required=True,
                   help="TSV: contig_id(query), target, fident, qcov, tcov, target_kingdom")
    p.add_argument("--out", required=True)
    p.add_argument("--min-id", type=float, default=0.90)
    p.add_argument("--min-cov", type=float, default=0.50)
    p.add_argument("--foreign-threshold", type=float, default=0.05)
    return p.parse_args()


def main():
    a = parse_args()
    contig_len = {}
    contig_mag = {}
    mag_kingdom = {}
    for line in open(a.contigs):
        if not line.strip() or line.startswith("contig_id"):
            continue
        cid, mid, king, length = line.rstrip("\n").split("\t")[:4]
        contig_len[cid] = int(length)
        contig_mag[cid] = mid
        mag_kingdom[mid] = king

    # best confident kingdom assignment per contig
    best = {}
    for line in open(a.hits):
        if not line.strip() or line.startswith("contig_id") or line.startswith("query"):
            continue
        f = line.rstrip("\n").split("\t")
        cid, _tgt, fident, qcov, tcov, tking = f[:6]
        if float(fident) < a.min_id or float(qcov) < a.min_cov or float(tcov) < a.min_cov:
            continue
        score = float(fident) * float(qcov)
        if cid not in best or score > best[cid][1]:
            best[cid] = (tking, score)

    mag = defaultdict(lambda: {"own_bp": 0, "foreign_bp": 0, "unassigned_bp": 0,
                               "total_bp": 0, "kingdom": None})
    for cid, length in contig_len.items():
        mid = contig_mag[cid]
        rec = mag[mid]
        rec["kingdom"] = mag_kingdom[mid]
        rec["total_bp"] += length
        assigned = best.get(cid)
        if assigned is None:
            rec["unassigned_bp"] += length
        elif assigned[0] == mag_kingdom[mid]:
            rec["own_bp"] += length
        else:
            rec["foreign_bp"] += length

    keep, drop = [], []
    for mid, rec in mag.items():
        assigned_bp = rec["own_bp"] + rec["foreign_bp"]
        frac = (rec["foreign_bp"] / assigned_bp) if assigned_bp else 0.0
        rec["foreign_fraction"] = frac
        (drop if frac >= a.foreign_threshold else keep).append(mid)

    out = {
        "params": {"min_id": a.min_id, "min_cov": a.min_cov,
                   "foreign_threshold": a.foreign_threshold},
        "n_mags": len(mag), "n_keep": len(keep), "n_drop": len(drop),
        "keep": sorted(keep), "drop": sorted(drop),
        "per_mag": mag,
    }
    Path(a.out).write_text(json.dumps(out, indent=2, default=dict))


if __name__ == "__main__":
    main()

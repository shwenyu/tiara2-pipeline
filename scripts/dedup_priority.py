#!/usr/bin/env python3
"""Within-bin CROSS-SPLIT-ONLY double-95 filter.

Given MMseqs hits where queries come from a lower-priority split and targets
from higher-priority splits, decide which query fragments to remove. Only
cross-split collisions remove a fragment; same-split duplicates are preserved.

Inputs are expressed as fragment IDs so this stage never rewrites sequences.
It emits a ``removed_ids`` list per split; materialization happens later.
"""
import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hits", required=True, help="MMseqs TSV: query,target,fident,qcov,tcov,...")
    p.add_argument("--query-split", required=True)
    p.add_argument("--out", required=True, help="JSON with removed_ids + stats")
    p.add_argument("--min-id", type=float, default=0.95)
    p.add_argument("--min-cov", type=float, default=0.95)
    p.add_argument("--query-count", type=int, default=None,
                   help="Total query fragments in this bin (for retention stats).")
    return p.parse_args()


def main():
    a = parse_args()
    remove = set()
    rows = 0
    with open(a.hits) as fh:
        for line in fh:
            if not line.strip():
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 5:
                raise SystemExit(f"malformed hit row: {line[:200]!r}")
            query, target, fident, qcov, tcov = f[:5]
            if query == target:
                continue  # self hit; never remove a fragment because of itself
            if (float(fident) + 1e-12 >= a.min_id
                    and float(qcov) + 1e-12 >= a.min_cov
                    and float(tcov) + 1e-12 >= a.min_cov):
                remove.add(query)
                rows += 1
    out = {
        "query_split": a.query_split,
        "accepted_hit_rows": rows,
        "removed_ids": sorted(remove),
        "removed_count": len(remove),
        "query_count": a.query_count,
        "retention": (None if not a.query_count
                      else (a.query_count - len(remove)) / a.query_count),
    }
    tmp = Path(a.out + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(out, indent=2))
    tmp.replace(a.out)


if __name__ == "__main__":
    main()

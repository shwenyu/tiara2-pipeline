#!/usr/bin/env python3
"""Build row-aligned k4/k5/k6 residual and frozen-base k7 crop features."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


FILES = ("bacteria", "archaea", "eukarya", "mitochondria", "plastids")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crops", required=True)
    parser.add_argument("--multiscale-tfidf", required=True)
    parser.add_argument("--base-tfidf", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=56)
    parser.add_argument("--chunk", type=int, default=2000)
    parser.add_argument("--splits", default="train,validation")
    args = parser.parse_args(argv)

    from tiara.src.transformations import TfidfWeighter
    from tiara.training import featurize_cache as fc

    # The crop generator preserves this exact frozen-corpus order. Labels in
    # this generic cache are placeholders; authoritative hierarchical labels
    # remain the immutable v2.3.2 arrays and are checked by row count below.
    fc.STAGE_SPEC["first"] = {
        "files": list(FILES),
        "labels": list(range(len(FILES))),
        "idf": "first-stage",
    }
    idfs = {}
    for k in (4, 5, 6):
        model = Path(args.multiscale_tfidf) / f"k{k}-first-stage"
        idfs[k] = np.asarray(TfidfWeighter.load_params(str(model)).idfs, dtype=np.float32)
    idfs[7] = np.asarray(
        TfidfWeighter.load_params(str(Path(args.base_tfidf))).idfs,
        dtype=np.float32,
    )
    for k, values in idfs.items():
        if values.shape != (4**k,):
            raise ValueError(f"bad IDF shape for k{k}: {values.shape}")

    splits = tuple(x.strip() for x in args.splits.split(",") if x.strip())
    shapes = fc.ensure_features(
        Path(args.out), args.crops, "first", (4, 5, 6, 7),
        idf_map=idfs, workers=args.workers, chunk=args.chunk,
        splits=splits, progress_secs=20.0,
        log=lambda message: print(message, flush=True),
    )
    manifest = {
        "version": "2.4.1-B",
        "contract": "continuous-crop-multiscale-plus-frozen-base-k7",
        "files_order": list(FILES),
        "residual_k": [4, 5, 6],
        "residual_dim": sum(4**k for k in (4, 5, 6)),
        "base_k": 7,
        "base_dim": 4**7,
        "shapes": {s: {str(k): list(v) for k, v in by_k.items()} for s, by_k in shapes.items()},
        "crops": str(Path(args.crops).resolve()),
        "multiscale_tfidf": str(Path(args.multiscale_tfidf).resolve()),
        "base_tfidf": str(Path(args.base_tfidf).resolve()),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "v241b_features.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

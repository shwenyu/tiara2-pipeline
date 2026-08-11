#!/usr/bin/env python3
"""Compare the M6 k7 capacity candidate with the frozen v2.3.2 control."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

HEADS = ("root", "euk", "prok", "organelle")
CONTROL_HIDDEN = [2048, 1024]
CANDIDATE_HIDDEN = [4096, 2048]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def metric(checkpoint: dict, head: str) -> float:
    return float(checkpoint["best_validation_metrics"][head]["macro_f1"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    control_path, candidate_path = Path(args.control), Path(args.candidate)
    control, candidate = load(control_path), load(candidate_path)

    if list(control["model"]["hidden"]) != CONTROL_HIDDEN:
        raise ValueError("control architecture mismatch")
    if list(candidate["model"]["hidden"]) != CANDIDATE_HIDDEN:
        raise ValueError("candidate architecture mismatch")
    if control["model"]["dim_in"] != 16384 or candidate["model"]["dim_in"] != 16384:
        raise ValueError("feature dimension mismatch")
    if control["schema"]["root"] != candidate["schema"]["root"]:
        raise ValueError("root class order mismatch")
    experiment = candidate.get("experiment") or {}
    if experiment.get("benchmark_access") is not False:
        raise ValueError("candidate does not declare benchmark isolation")

    heads = {}
    for head in HEADS:
        before, after = metric(control, head), metric(candidate, head)
        heads[head] = {
            "control_macro_f1": before,
            "candidate_macro_f1": after,
            "delta_macro_f1_pp": 100.0 * (after - before),
        }
    root_delta = heads["root"]["delta_macro_f1_pp"]
    leaf_deltas = [heads[name]["delta_macro_f1_pp"] for name in HEADS[1:]]
    capacity_signal = root_delta >= 0.05
    promotion_gate = capacity_signal and min(leaf_deltas) >= -0.05
    if promotion_gate:
        next_step = "repeat across seeds before any promotion"
    elif capacity_signal:
        next_step = "run M7 multi-objective checkpointing and shrinkage to resolve head trade-offs"
    else:
        next_step = "treat k7 as representation-limited and design RC-CNN expert"
    decision = {
        "capacity_signal": capacity_signal,
        "promotion_gate": promotion_gate,
        "root_gain_gate_pp": 0.05,
        "max_leaf_regression_pp": 0.05,
        "next": next_step,
    }
    report = {
        "experiment": experiment.get("name", "M6 k7 capacity ceiling"),
        "control": {"path": str(control_path), "sha256": sha256(control_path), "hidden": CONTROL_HIDDEN},
        "candidate": {"path": str(candidate_path), "sha256": sha256(candidate_path), "hidden": CANDIDATE_HIDDEN},
        "selection": "independent validation only; external benchmark not accessed",
        "heads": heads,
        "decision": decision,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

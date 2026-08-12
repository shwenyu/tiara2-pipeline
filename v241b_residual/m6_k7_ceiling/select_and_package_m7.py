#!/usr/bin/env python3
"""Select the preregistered M7 seed and build a benchmark-ready release bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch

HEADS = ("root", "euk", "prok", "organelle")
EXPECTED_HIDDEN = [4096, 2048]
EXPECTED_DIM = 16384
ROOT_MIN_GAIN_PP = 0.05
LEAF_MAX_REGRESSION_PP = 0.05


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_seed(path: Path) -> dict:
    comparison_path = path / "m7_comparison.json"
    checkpoint_path = path / "multiobjective_model.pt"
    history_path = path / "training_history.json"
    sampling_path = path / "sampling_plan.json"
    for required in (comparison_path, checkpoint_path, history_path, sampling_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    comparison = json.loads(comparison_path.read_text())
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    history = json.loads(history_path.read_text())
    sampling = json.loads(sampling_path.read_text())
    seed = int(sampling["seed"])
    deltas = {
        head: float(comparison["heads"][head]["delta_macro_f1_pp"])
        for head in HEADS
    }
    feasible = (
        deltas["root"] >= ROOT_MIN_GAIN_PP
        and min(deltas[head] for head in HEADS[1:]) >= -LEAF_MAX_REGRESSION_PP
    )
    score = min(deltas.values())

    if list(checkpoint["model"]["hidden"]) != EXPECTED_HIDDEN:
        raise ValueError(f"seed {seed}: hidden architecture mismatch")
    if int(checkpoint["model"]["dim_in"]) != EXPECTED_DIM:
        raise ValueError(f"seed {seed}: feature dimension mismatch")
    if checkpoint.get("checkpoint_selector") != "m7_maximin_head_delta":
        raise ValueError(f"seed {seed}: checkpoint selector mismatch")
    if checkpoint.get("experiment", {}).get("benchmark_access") is not False:
        raise ValueError(f"seed {seed}: benchmark isolation not declared")
    if not feasible or not comparison.get("decision", {}).get("promotion_gate"):
        raise ValueError(f"seed {seed}: preregistered promotion gate failed")
    if len(history) != 50:
        raise ValueError(f"seed {seed}: expected 50 epochs, found {len(history)}")

    checkpoint_hash = sha256(checkpoint_path)
    recorded_hash = comparison["candidate"]["sha256"]
    if checkpoint_hash != recorded_hash:
        raise ValueError(f"seed {seed}: checkpoint hash mismatch")
    return {
        "seed": seed,
        "directory": str(path.resolve()),
        "checkpoint": checkpoint_path,
        "checkpoint_sha256": checkpoint_hash,
        "selected_epoch": int(checkpoint["epoch"]),
        "deltas_pp": deltas,
        "score": score,
        "feasible": feasible,
        "epochs_completed": len(history),
        "comparison": comparison,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-dir", action="append", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--model-tag", default="v2.4.1-M7")
    args = parser.parse_args()

    results = [load_seed(Path(value)) for value in args.seed_dir]
    seeds = [item["seed"] for item in results]
    if len(results) != 3 or len(set(seeds)) != 3:
        raise ValueError("exactly three distinct completed seeds are required")
    selected = max(results, key=lambda item: (item["score"], item["deltas_pp"]["root"], -item["seed"]))

    destination = Path(args.destination)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"destination is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    model_path = destination / "multiobjective_model.pt"
    shutil.copy2(selected["checkpoint"], model_path)
    model_hash = sha256(model_path)

    seed_summary = []
    for item in sorted(results, key=lambda value: value["seed"]):
        seed_summary.append({
            "seed": item["seed"],
            "epochs_completed": item["epochs_completed"],
            "selected_epoch": item["selected_epoch"],
            "checkpoint_sha256": item["checkpoint_sha256"],
            "validation_deltas_pp": item["deltas_pp"],
            "maximin_score_pp": item["score"],
            "promotion_gate": item["feasible"],
        })

    manifest = {
        "manifest_version": 1,
        "model_family": "tiara2_hierarchical",
        "model_tag": args.model_tag,
        "status": "benchmark_candidate",
        "checkpoint": "multiobjective_model.pt",
        "checkpoint_metadata": {
            "dim_in": EXPECTED_DIM,
            "hidden": EXPECTED_HIDDEN,
            "dropout": 0.2,
            "head_sizes": {"root": 3, "euk": 8, "prok": 2, "organelle": 2},
            "epoch": selected["selected_epoch"],
            "seed": selected["seed"],
            "checkpoint_selector": "m7_maximin_head_delta",
        },
        "feature": {
            "k": 7,
            "tfidf": "../tfidf-models-v2.2.0/k7-first-stage",
        },
        "selection": {
            "data": "independent validation only",
            "external_benchmark_accessed": False,
            "required_seeds": sorted(seeds),
            "root_min_gain_pp": ROOT_MIN_GAIN_PP,
            "leaf_max_regression_pp": LEAF_MAX_REGRESSION_PP,
            "objective": "maximize minimum macro-F1 delta across root/euk/prok/organelle",
            "tie_break": "root delta, then lower seed",
            "selected_seed": selected["seed"],
            "selected_maximin_score_pp": selected["score"],
            "seeds": seed_summary,
        },
        "files": [{
            "name": "multiobjective_model.pt",
            "bytes": model_path.stat().st_size,
            "sha256": model_hash,
        }],
        "output_columns": [
            "record_id", "root", "leaf", "root_probability", "leaf_probability"
        ],
        "published_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "gates": {
            "all_seeds_completed": True,
            "all_seeds_validation_promotion_gate": all(item["feasible"] for item in results),
            "external_benchmark": "pending",
        },
    }
    manifest_path = destination / "model_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (destination / "cross_seed_selection.json").write_text(
        json.dumps(manifest["selection"], indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

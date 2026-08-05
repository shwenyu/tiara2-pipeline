#!/usr/bin/env python3
"""Atomically publish a trained Tiara2 hierarchical model into tiara/models.

Example:
  python publish_hierarchical_model.py \
    --source /data/shouhanyu/Tiara2_v2_3_0/models/v2.3.0 \
    --repo-root /home/shouhanyu/tiara2_pipeline \
    --tag v2.3.0 \
    --tfidf tiara/models/tfidf-models-v2.2.0/k7-first-stage \
    --k 7 --set-current

The inference package excludes last_checkpoint.pt by default because optimizer
state is not needed for inference. Pass --include-resume only for an intentional
training-resume archive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

REQUIRED_CHECKPOINT_KEYS = {
    "state_dict", "schema", "model", "version", "best_root_macro_f1"
}
OPTIONAL_FILES = (
    "training_history.json",
    "calibration.json",
    "temperatures.json",
    "thresholds.json",
    "evaluation.json",
)


class PublishError(RuntimeError):
    pass


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def validate_tag(tag: str) -> str:
    tag = str(tag).strip()
    if not tag or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for c in tag):
        raise PublishError(f"unsafe model tag: {tag!r}")
    return tag


def inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def resolve_repo_path(repo: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else repo / path


def validate_checkpoint(path: Path, expected_version: str) -> dict:
    try:
        import torch
    except ImportError as exc:
        raise PublishError(
            "PyTorch is required for checkpoint validation. Run this script in "
            "the tiara2 training environment."
        ) from exc
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise PublishError(f"cannot load checkpoint {path}: {exc}") from exc
    if not isinstance(checkpoint, dict):
        raise PublishError("checkpoint root must be a mapping")
    missing = sorted(REQUIRED_CHECKPOINT_KEYS - set(checkpoint))
    if missing:
        raise PublishError("checkpoint missing keys: " + ", ".join(missing))
    version = str(checkpoint.get("version", ""))
    normalized_tag = expected_version.removeprefix("v")
    if version not in (expected_version, normalized_tag):
        raise PublishError(
            f"checkpoint version={version!r} does not match tag={expected_version!r}"
        )
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise PublishError("checkpoint state_dict is empty")
    nonfinite = []
    tensor_count = 0
    parameter_count = 0
    for name, value in state.items():
        if not hasattr(value, "numel"):
            continue
        tensor_count += 1
        parameter_count += int(value.numel())
        try:
            finite = bool(torch.isfinite(value).all())
        except (TypeError, RuntimeError):
            finite = True
        if not finite:
            nonfinite.append(name)
    if nonfinite:
        raise PublishError(
            f"checkpoint has non-finite values in {len(nonfinite)} tensors: "
            + ", ".join(nonfinite[:10])
        )
    score = float(checkpoint["best_root_macro_f1"])
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise PublishError(f"invalid best_root_macro_f1: {score}")
    schema = checkpoint["schema"]
    if not isinstance(schema, dict) or not schema.get("root"):
        raise PublishError("checkpoint schema/root is missing")
    model = checkpoint["model"]
    if not isinstance(model, dict) or not model.get("dim_in"):
        raise PublishError("checkpoint model configuration is missing")
    return {
        "version": version,
        "best_root_macro_f1": score,
        "root_classes": list(schema["root"]),
        "head_sizes": dict(model.get("head_sizes") or {}),
        "dim_in": int(model["dim_in"]),
        "hidden": list(model.get("hidden") or []),
        "dropout": float(model.get("dropout", 0.0)),
        "tensor_count": tensor_count,
        "parameter_count": parameter_count,
        "world_size": checkpoint.get("world_size"),
        "epoch": checkpoint.get("epoch"),
    }


def read_json_if_present(path: Path) -> object | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise PublishError(f"invalid JSON file {path}: {exc}") from exc


def atomic_symlink(link: Path, target_name: str) -> None:
    temp = link.with_name(link.name + f".tmp.{os.getpid()}")
    if temp.exists() or temp.is_symlink():
        temp.unlink()
    os.symlink(target_name, temp)
    os.replace(temp, link)


def publish(args) -> dict:
    repo = Path(args.repo_root).expanduser().resolve()
    source = Path(args.source).expanduser().resolve()
    tag = validate_tag(args.tag)
    model_root = repo / "tiara" / "models"
    destination = model_root / f"hierarchical-models-{tag}"
    checkpoint = source / args.checkpoint_name
    tfidf = resolve_repo_path(repo, args.tfidf).resolve()

    if not (repo / "tiara").is_dir():
        raise PublishError(f"not a Tiara2 repository: {repo}")
    if not model_root.is_dir():
        raise PublishError(f"model root is missing: {model_root}")
    if not source.is_dir():
        raise PublishError(f"training output directory is missing: {source}")
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise PublishError(f"trained checkpoint is missing or empty: {checkpoint}")
    if not (tfidf / "model.npy").is_file() or not (tfidf / "params.txt").is_file():
        raise PublishError(f"TF-IDF folder is incomplete: {tfidf}")
    if not inside(destination, model_root):
        raise PublishError(f"destination escapes tiara/models: {destination}")
    if source == destination:
        raise PublishError("source and destination are the same directory")

    if args.skip_checkpoint_validation:
        checkpoint_meta = {"validation": "skipped by explicit flag"}
    else:
        checkpoint_meta = validate_checkpoint(checkpoint, tag)

    files = [checkpoint]
    for name in OPTIONAL_FILES:
        candidate = source / name
        if candidate.is_file():
            if candidate.suffix == ".json":
                read_json_if_present(candidate)
            files.append(candidate)
    if args.include_resume:
        resume = source / "last_checkpoint.pt"
        if not resume.is_file():
            raise PublishError(f"--include-resume requested but missing: {resume}")
        files.append(resume)

    plan = {
        "source": str(source),
        "destination": str(destination),
        "files": [path.name for path in files],
        "tfidf": str(tfidf),
        "tag": tag,
        "k": int(args.k),
        "set_current": bool(args.set_current),
    }
    print(json.dumps(plan, indent=2, sort_keys=True))
    if args.dry_run:
        return {"dry_run": True, **plan}

    model_root.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp.", dir=model_root))
    backup = None
    try:
        for path in files:
            shutil.copy2(path, temp / path.name)

        file_records = []
        for path in sorted(temp.iterdir()):
            if path.is_file():
                file_records.append({
                    "name": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                })

        try:
            tfidf_relative = str(tfidf.relative_to(repo))
        except ValueError:
            tfidf_relative = str(tfidf)
        manifest = {
            "manifest_version": 1,
            "model_family": "tiara2_hierarchical",
            "model_tag": tag,
            "published_at": now_iso(),
            "source": str(source),
            "checkpoint": args.checkpoint_name,
            "checkpoint_metadata": checkpoint_meta,
            "feature": {
                "k": int(args.k),
                "tfidf": tfidf_relative,
                "tfidf_model_sha256": sha256(tfidf / "model.npy"),
                "tfidf_params_sha256": sha256(tfidf / "params.txt"),
            },
            "output_columns": [
                "record_id", "root", "leaf", "root_probability", "leaf_probability"
            ],
            "files": file_records,
        }
        (temp / "model_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        checksums = []
        for path in sorted(p for p in temp.iterdir() if p.is_file()):
            checksums.append(f"{sha256(path)}  {path.name}")
        (temp / "SHA256SUMS").write_text("\n".join(checksums) + "\n")

        if destination.exists() or destination.is_symlink():
            if not args.force:
                raise PublishError(
                    f"destination already exists: {destination}; rerun with --force"
                )
            backup = destination.with_name(
                destination.name + ".backup." + time.strftime("%Y%m%d-%H%M%S")
            )
            destination.rename(backup)
        temp.rename(destination)

        current_link = model_root / "hierarchical-models-current"
        if args.set_current:
            atomic_symlink(current_link, destination.name)

        receipt = {
            "published": str(destination),
            "manifest": str(destination / "model_manifest.json"),
            "checkpoint": str(destination / args.checkpoint_name),
            "current": str(current_link) if args.set_current else None,
            "backup": str(backup) if backup else None,
        }
        print("[published]")
        print(json.dumps(receipt, indent=2, sort_keys=True))
        print("\nInference command:")
        print(
            "python -m tiara.hierarchical classify "
            f"--checkpoint {receipt['checkpoint']} "
            f"--tfidf {tfidf} -i input.fasta -o predictions.tsv"
        )
        return receipt
    except Exception:
        if temp.exists():
            shutil.rmtree(temp, ignore_errors=True)
        if backup is not None and backup.exists() and not destination.exists():
            backup.rename(destination)
        raise


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="/data/shouhanyu/Tiara2_v2_3_0/models/v2.3.0",
        help="training output directory",
    )
    parser.add_argument(
        "--repo-root", default="/home/shouhanyu/tiara2_pipeline"
    )
    parser.add_argument("--tag", default="v2.3.0")
    parser.add_argument("--checkpoint-name", default="hierarchical_model.pt")
    parser.add_argument(
        "--tfidf",
        default="tiara/models/tfidf-models-v2.2.0/k7-first-stage",
    )
    parser.add_argument("--k", type=int, default=7)
    parser.add_argument("--set-current", action="store_true")
    parser.add_argument("--include-resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-checkpoint-validation",
        action="store_true",
        help="testing/emergency only; normal publication must validate with PyTorch",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        publish(args)
    except PublishError as exc:
        print(f"[publish error] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

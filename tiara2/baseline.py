"""Freeze the v2.2.0 baseline and reclaim storage without losing reproducibility.

Destructive cleanup is deliberately a separate command.  A cleanup is refused
unless a verified freeze manifest exists, the exact protected paths are still
present, every deletion target is under base/fast_base, and the user supplies
both --apply and --yes.  Releasing upstream corpora requires one more explicit
acknowledgement.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

SCHEMA_VERSION = 1
FREEZE_FILE = "freeze_manifest.json"
PLAN_FILE = "cleanup_plan.json"

class BaselineError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _json_sha(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def _within(path: Path, root: Path) -> bool:
    try:
        path.absolute().relative_to(root.absolute())
        return True
    except ValueError:
        return False


def _same_or_parent(parent: Path, child: Path) -> bool:
    return parent.absolute() == child.absolute() or _within(child, parent)


def _tree_records(root: Path, *, hash_files: bool, splits: Iterable[str] | None = None) -> list[dict]:
    roots = [root / s for s in splits] if splits else [root]
    rows: list[dict] = []
    for start in roots:
        if not start.exists() and not start.is_symlink():
            rows.append({"path": str(start.relative_to(root)), "missing": True})
            continue
        if start.is_file() or start.is_symlink():
            paths = [start]
        else:
            paths = sorted(p for p in start.rglob("*") if p.is_file() or p.is_symlink())
        for path in paths:
            rel = str(path.relative_to(root))
            if path.is_symlink():
                rows.append({"path": rel, "type": "symlink", "target": os.readlink(path)})
                continue
            st = path.stat()
            row = {"path": rel, "type": "file", "bytes": st.st_size,
                   "mtime_ns": st.st_mtime_ns}
            if hash_files:
                row["sha256"] = _sha256(path)
            rows.append(row)
    return rows


def _copy_if_file(src, dest: Path, copied: list[dict]) -> None:
    if not src:
        return
    path = Path(str(src))
    if not path.is_file():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dest)
    copied.append({"source": str(path), "copy": str(dest), "sha256": _sha256(dest)})


def _published_dirs(cfg: dict, repo_root: Path) -> tuple[Path, Path]:
    pub = cfg.get("publish", {}) or {}
    def resolve(value):
        p = Path(str(value))
        return p if p.is_absolute() else repo_root / p
    return resolve(pub.get("nnet_dest", "")), resolve(pub.get("tfidf_dest", ""))


def _required_small_files(cfg: dict) -> list[tuple[str, object]]:
    pub = cfg.get("publish", {}) or {}
    report = pub.get("report", {}) or {}
    ver = cfg.get("versioning", {}) or {}
    train = cfg.get("train", {}) or {}
    return [
        ("config_snapshot.json", ver.get("snapshot")),
        ("run_report.json", report.get("out_json")),
        ("run_report.md", report.get("out_md")),
        ("runs_index.csv", report.get("index_csv")),
        ("versions.json", ver.get("registry")),
        ("training_manifest.json", Path(str(train.get("out_models", ""))) / "training_manifest.json"),
        ("tfidf_manifest.json", Path(str(train.get("tfidf_dir", ""))) / "tfidf_manifest.json"),
    ]


def freeze(cfg: dict, repo_root: str | Path, *, tag: str = "v2.2.0",
           freeze_dir: str | Path | None = None, metadata_only: bool = False,
           force: bool = False) -> dict:
    """Create a tamper-evident baseline record. No data is deleted."""
    repo_root = Path(repo_root)
    section = cfg.get("baseline_freeze", {}) or {}
    out = Path(str(freeze_dir or section.get("freeze_dir")
                   or Path(str(cfg["base"])) / "baselines" / tag))
    manifest_path = out / FREEZE_FILE
    if manifest_path.exists() and not force:
        raise BaselineError(f"baseline already frozen: {manifest_path}; use --force only intentionally")
    out.mkdir(parents=True, exist_ok=True)

    train = cfg.get("train", {}) or {}
    train_ready = Path(str(train.get("train_ready", "")))
    splits = list(section.get("training_splits") or ["train", "validation"])
    if not train_ready.is_dir():
        raise BaselineError(f"training input is missing: {train_ready}")
    for split in splits:
        if not (train_ready / split).is_dir():
            raise BaselineError(f"required training split is missing: {train_ready / split}")

    nnet, tfidf = _published_dirs(cfg, repo_root)
    if not nnet.is_dir() or not list(nnet.glob("*.pkl")):
        raise BaselineError(f"published neural models missing: {nnet}")
    if not tfidf.is_dir() or not list(tfidf.glob("k*-*-stage/model.npy")):
        raise BaselineError(f"published TF-IDF models missing: {tfidf}")

    copied: list[dict] = []
    small = out / "metadata"
    small.mkdir(parents=True, exist_ok=True)
    (small / "resolved_config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True, default=str) + "\n")
    copied.append({"source": "resolved runtime config", "copy": str(small / "resolved_config.json"),
                   "sha256": _sha256(small / "resolved_config.json")})
    for name, src in _required_small_files(cfg):
        _copy_if_file(src, small / name, copied)
    for raw in section.get("extra_metadata_files", []) or []:
        path = Path(str(raw))
        _copy_if_file(path, small / "extra" / path.name, copied)

    training_records = _tree_records(train_ready, hash_files=not metadata_only, splits=splits)
    if not any(r.get("type") == "file" for r in training_records):
        raise BaselineError("training input contains no regular files")
    model_records = {
        "nnet": _tree_records(nnet, hash_files=True),
        "tfidf": _tree_records(tfidf, hash_files=True),
    }
    protected = protected_paths(cfg, repo_root, out)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "tag": tag,
        "frozen_at": _now(),
        "model_tag": cfg.get("model_tag"),
        "version_tag": cfg.get("version_tag"),
        "corpus_tag": cfg.get("corpus_tag"),
        "repo_root": str(repo_root),
        "config_sha256": _json_sha(cfg),
        "metadata_only": bool(metadata_only),
        "training_input": {"root": str(train_ready), "splits": splits,
                           "records": training_records},
        "published_models": {"nnet_root": str(nnet), "tfidf_root": str(tfidf),
                             "records": model_records},
        "metadata_copies": copied,
        "protected_paths": [str(p) for p in protected],
    }
    payload["freeze_id"] = _json_sha(payload)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (out / "KEEP_PATHS.txt").write_text("\n".join(payload["protected_paths"]) + "\n")
    (out / "FROZEN").write_text(f"{tag}\n{payload['freeze_id']}\n")
    return payload


def load_freeze(path: str | Path) -> dict:
    root = Path(path)
    manifest = root / FREEZE_FILE if root.is_dir() else root
    if not manifest.is_file():
        raise BaselineError(f"freeze manifest missing: {manifest}")
    payload = json.loads(manifest.read_text())
    if payload.get("schema_version") != SCHEMA_VERSION or not payload.get("freeze_id"):
        raise BaselineError(f"invalid freeze manifest: {manifest}")
    expected = payload["freeze_id"]
    unsigned = dict(payload)
    unsigned.pop("freeze_id", None)
    if _json_sha(unsigned) != expected:
        raise BaselineError(f"freeze manifest integrity check failed: {manifest}")
    return payload


def protected_paths(cfg: dict, repo_root: Path, freeze_dir: Path) -> list[Path]:
    train = cfg.get("train", {}) or {}
    pub = cfg.get("publish", {}) or {}
    rep = pub.get("report", {}) or {}
    ver = cfg.get("versioning", {}) or {}
    section = cfg.get("baseline_freeze", {}) or {}
    nnet, tfidf = _published_dirs(cfg, repo_root)
    values = [freeze_dir, train.get("train_ready"), train.get("log_dir"),
              cfg.get("results_root"), cfg.get("log_dir"), nnet, tfidf,
              rep.get("out_json"), rep.get("out_md"), rep.get("index_csv"),
              ver.get("snapshot"), ver.get("registry")]
    values += list(section.get("benchmark_paths", []) or [])
    values += list(section.get("extra_keep_paths", []) or [])
    seen, out = set(), []
    for value in values:
        if not value:
            continue
        p = Path(str(value)).absolute()
        if str(p) not in seen:
            seen.add(str(p)); out.append(p)
    return out


@dataclass
class CleanupItem:
    key: str
    path: str
    reason: str
    bytes: int = 0
    exists: bool = False


def _size_no_follow(root: Path) -> int:
    if root.is_symlink():
        return 0
    if root.is_file():
        return root.stat().st_size
    total = 0
    if root.is_dir():
        for p in root.rglob("*"):
            try:
                if p.is_file() and not p.is_symlink():
                    total += p.stat().st_size
            except OSError:
                pass
    return total


def _candidate_paths(cfg: dict, profile: str, release_upstream: bool) -> list[tuple[str, object, str]]:
    train = cfg.get("train", {}) or {}
    section = cfg.get("baseline_freeze", {}) or {}
    rows = [
        ("feature_cache", train.get("feature_cache"), "dense derived k-mer matrices"),
        ("seq_pack", train.get("seq_pack"), "derived 2-bit sequence pack"),
        ("flat_data", train.get("flat_data"), "unused concatenated corpus"),
        ("checkpoint_root", cfg.get("checkpoint_root"), "old pipeline checkpoints"),
        ("work_root", cfg.get("work_root"), "rebuildable stage work tree"),
    ]
    if profile == "reproducible":
        rows += [
            ("source_model_dir", train.get("out_models"), "published model copy is frozen"),
            ("source_tfidf_dir", train.get("tfidf_dir"), "published TF-IDF copy is frozen"),
        ]
    if release_upstream:
        fbp = cfg.get("fragment_bp_balance", {}) or {}
        rows += [
            ("source_ready", cfg.get("source_ready"), "exact train/validation input is retained"),
            ("corpus_ready", cfg.get("corpus_ready"), "exact train/validation input is retained"),
            ("balance_source", fbp.get("source_root"), "exact train/validation input is retained"),
        ]
    for i, value in enumerate(section.get("extra_delete_paths", []) or []):
        rows.append((f"extra_{i}", value, "explicit baseline_freeze.extra_delete_paths"))
    return rows


def cleanup_plan(cfg: dict, repo_root: str | Path, freeze_dir: str | Path, *,
                 profile: str = "safe", release_upstream: bool = False) -> dict:
    if profile not in ("safe", "reproducible"):
        raise BaselineError("profile must be safe or reproducible")
    frozen = load_freeze(freeze_dir)
    if release_upstream and frozen.get("metadata_only"):
        raise BaselineError("cannot release upstream data after a metadata-only freeze; rerun freeze-baseline without --metadata-only")
    repo_root, freeze_dir = Path(repo_root), Path(freeze_dir)
    protected = [Path(p) for p in frozen["protected_paths"]]
    roots = [Path(str(cfg.get("base", ""))).absolute(),
             Path(str(cfg.get("fast_base") or cfg.get("base", ""))).absolute()]
    train_ready = Path(frozen["training_input"]["root"]).absolute()
    items, seen = [], set()
    for key, raw, reason in _candidate_paths(cfg, profile, release_upstream):
        if not raw:
            continue
        path = Path(str(raw)).absolute()
        if str(path) in seen:
            continue
        seen.add(str(path))
        if not any(_within(path, root) for root in roots if str(root) not in ("", ".")):
            continue
        if any(path == root for root in roots):
            continue
        if _same_or_parent(path, train_ready):
            continue
        if any(_same_or_parent(path, keep) or _same_or_parent(keep, path)
               for keep in protected):
            continue
        items.append(CleanupItem(key, str(path), reason,
                                 _size_no_follow(path), path.exists() or path.is_symlink()))
    payload = {"schema_version": SCHEMA_VERSION, "created_at": _now(),
               "freeze_id": frozen["freeze_id"], "freeze_dir": str(freeze_dir),
               "profile": profile, "release_upstream": release_upstream,
               "allowed_roots": [str(r) for r in roots],
               "protected_paths": [str(p) for p in protected],
               "items": [asdict(x) for x in items]}
    payload["plan_id"] = _json_sha(payload)
    (freeze_dir / PLAN_FILE).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def load_plan(freeze_dir: str | Path) -> dict:
    path = Path(freeze_dir) / PLAN_FILE
    if not path.is_file():
        raise BaselineError(f"cleanup plan missing: {path}; run cleanup-baseline without --apply first")
    payload = json.loads(path.read_text())
    expected = payload.get("plan_id")
    unsigned = dict(payload)
    unsigned.pop("plan_id", None)
    if not expected or _json_sha(unsigned) != expected:
        raise BaselineError(f"cleanup plan integrity check failed: {path}")
    return payload


def render_plan(plan: dict) -> str:
    lines = [f"[baseline cleanup] profile={plan['profile']} plan={plan['plan_id'][:12]}"]
    total = 0
    for item in plan["items"]:
        mark = "x" if item["exists"] else " "
        total += item["bytes"] if item["exists"] else 0
        lines.append(f"  [{mark}] {item['key']:<20} {item['bytes'] / 1024**3:9.2f} GiB  {item['path']}")
        lines.append(f"      {item['reason']}")
    lines.append(f"  total reclaimable: {total / 1024**3:.2f} GiB")
    return "\n".join(lines)


def apply_cleanup(plan: dict, *, apply: bool = False, yes: bool = False,
                  acknowledge_upstream_loss: bool = False) -> dict:
    if not apply:
        return {"dry_run": True, "removed": [], "freed_bytes": 0}
    if not yes:
        raise BaselineError("deletion requires --apply --yes")
    if plan.get("release_upstream") and not acknowledge_upstream_loss:
        raise BaselineError("upstream deletion also requires --acknowledge-upstream-loss")
    frozen = load_freeze(plan["freeze_dir"])
    if frozen["freeze_id"] != plan["freeze_id"]:
        raise BaselineError("cleanup plan does not match the freeze manifest")
    allowed = [Path(p) for p in plan["allowed_roots"]]
    protected = [Path(p) for p in plan["protected_paths"]]
    removed, freed = [], 0
    for item in plan["items"]:
        path = Path(item["path"])
        if not item["exists"] or not (path.exists() or path.is_symlink()):
            continue
        if not any(_within(path, root) and path != root for root in allowed):
            raise BaselineError(f"unsafe deletion target: {path}")
        if any(_same_or_parent(path, keep) or _same_or_parent(keep, path)
               for keep in protected):
            raise BaselineError(f"target intersects protected data: {path}")
        size = _size_no_follow(path)
        if path.is_symlink() or path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)
        removed.append(str(path)); freed += size
    receipt = {"applied_at": _now(), "freeze_id": plan["freeze_id"],
               "plan_id": plan["plan_id"], "removed": removed,
               "freed_bytes": freed}
    Path(plan["freeze_dir"], "cleanup_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt

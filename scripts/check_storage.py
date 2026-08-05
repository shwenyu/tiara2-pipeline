#!/usr/bin/env python3
"""Storage-tier preflight: is the hot tier actually hot, and big enough?

The pipeline splits its paths across two tiers (see `storage:` in
config/config.yaml):

  COLD  /data   135 T HDD array -- raw assemblies, metadata, published models,
                logs, results. Read once, or must survive.
  HOT   /ssd     14 T NVMe      -- chopped corpus, length-bin shards, sequence
                pack, feature cache. Read over and over.

This script answers three questions before a long run starts:

1. Does every hot key resolve onto the NVMe, or did one silently stay on /data?
2. Is there enough free space on each tier to finish?
3. Are there symlinks on the hot tier pointing back into the cold tier?
   That is the subtle one: with ``dedup.enabled: false`` the regroup stage
   mirrors source_ready into corpus_ready using SYMLINKS, so a corpus_ready on
   NVMe whose links all target /data is exactly as slow as /data.

Usage::

    python3 scripts/check_storage.py
    python3 scripts/check_storage.py --config config/config.yaml --json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tiara2 import config as config_mod  # noqa: E402
from tiara2 import paths as paths_mod  # noqa: E402

GB = 1024 ** 3
DEFAULT_HOT = ("source_ready", "corpus_ready", "work_root", "train.flat_data",
               "train.tfidf_dir", "train.feature_cache", "train.seq_pack")
DEFAULT_COLD = ("results_root", "log_dir", "checkpoint_root", "train.out_models",
                "train.log_dir")


def dotted_get(cfg: dict, dotted: str):
    """Look up ``a.b`` in a nested config dict; ``None`` when absent."""
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def mount_of(path: str | os.PathLike) -> str:
    """Deepest existing ancestor of ``path`` -- what disk_usage can stat."""
    p = Path(path).resolve()
    while not p.exists() and p != p.parent:
        p = p.parent
    return str(p)


def device_of(path: str | os.PathLike):
    try:
        return os.stat(mount_of(path)).st_dev
    except OSError:
        return None


def free_gb(path: str | os.PathLike) -> float | None:
    try:
        return shutil.disk_usage(mount_of(path)).free / GB
    except OSError:
        return None


def dir_size_gb(path: str | os.PathLike, follow_symlinks: bool = False) -> float:
    """Apparent size of a tree, skipping symlinks so mirrored corpora are not
    double-counted."""
    total = 0
    root = Path(path)
    if not root.exists():
        return 0.0
    if root.is_file():
        return root.stat().st_size / GB
    for dirpath, dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
        for name in filenames:
            fp = Path(dirpath) / name
            if fp.is_symlink() and not follow_symlinks:
                continue
            try:
                total += fp.stat().st_size
            except OSError:
                pass
    return total / GB


def count_cross_tier_symlinks(path: str | os.PathLike, hot_device, limit: int = 2000):
    """How many symlinks under ``path`` point off the hot device."""
    crossing = 0
    checked = 0
    root = Path(path)
    if not root.exists() or hot_device is None:
        return 0, 0
    for dirpath, dirnames, filenames in os.walk(root):
        for name in filenames:
            fp = Path(dirpath) / name
            if not fp.is_symlink():
                continue
            checked += 1
            try:
                if os.stat(fp).st_dev != hot_device:
                    crossing += 1
            except OSError:
                pass
            if checked >= limit:
                return crossing, checked
    return crossing, checked


def inspect(cfg: dict, measure: bool = True) -> dict:
    storage = cfg.get("storage", {}) or {}
    hot_keys = list(storage.get("hot_keys") or DEFAULT_HOT)
    cold_keys = list(storage.get("cold_keys") or DEFAULT_COLD)
    fast_base = cfg.get("fast_base") or cfg.get("base")
    base = cfg.get("base")

    hot_device = device_of(fast_base)
    cold_device = device_of(base)
    collapsed = hot_device is not None and hot_device == cold_device

    entries = []
    for tier, keys, want_device in (("hot", hot_keys, hot_device),
                                   ("cold", cold_keys, cold_device)):
        for key in keys:
            value = dotted_get(cfg, key)
            if not isinstance(value, str) or not value:
                entries.append({"key": key, "tier": tier, "path": "",
                                "exists": False, "on_expected_tier": None,
                                "size_gb": 0.0})
                continue
            dev = device_of(value)
            entries.append({
                "key": key,
                "tier": tier,
                "path": value,
                "exists": Path(value).exists(),
                "on_expected_tier": None if collapsed or dev is None else dev == want_device,
                "size_gb": round(dir_size_gb(value), 2) if measure else None,
            })

    corpus_ready = cfg.get("corpus_ready", "")
    crossing, checked = (count_cross_tier_symlinks(corpus_ready, hot_device)
                         if measure and isinstance(corpus_ready, str) else (0, 0))

    report = {
        "base": base,
        "fast_base": fast_base,
        "tiers_collapsed": collapsed,
        "free_gb": {"hot": free_gb(fast_base), "cold": free_gb(base)},
        "min_free_gb": {"hot": storage.get("min_fast_free_gb", 0),
                        "cold": storage.get("min_cold_free_gb", 0)},
        "entries": entries,
        "cross_tier_symlinks": {"path": corpus_ready, "crossing": crossing,
                                "checked": checked},
    }

    problems = []
    if not Path(str(fast_base)).exists():
        problems.append(f"hot tier does not exist: {fast_base}")
    if collapsed and fast_base != base:
        problems.append(f"{fast_base} and {base} are the same device -- the hot tier is not separate")
    for tier in ("hot", "cold"):
        have = report["free_gb"][tier]
        want = report["min_free_gb"][tier] or 0
        if have is not None and want and have < want:
            problems.append(f"{tier} tier has {have:,.0f} GB free, needs {want:,.0f} GB")
    for entry in entries:
        if entry["on_expected_tier"] is False:
            problems.append(f"{entry['key']} is on the wrong tier: {entry['path']}")
    if crossing:
        problems.append(
            f"{crossing} of {checked} symlinks under corpus_ready point off the hot tier "
            "-- reads will hit the slow disk anyway")
    report["problems"] = problems
    return report


def render(report: dict) -> str:
    lines = ["storage tiers", ""]
    lines.append(f"  hot   {report['fast_base']}")
    lines.append(f"  cold  {report['base']}")
    if report["tiers_collapsed"]:
        lines.append("  (same device -- tiers collapsed)")
    lines.append("")
    for tier in ("hot", "cold"):
        have = report["free_gb"][tier]
        want = report["min_free_gb"][tier] or 0
        shown = f"{have:,.0f}" if have is not None else "?"
        mark = "" if have is None or not want or have >= want else "   << BELOW FLOOR"
        lines.append(f"  {tier:5} free {shown:>10} GB   floor {want:>7,} GB{mark}")
    lines.append("")
    lines.append(f"  {'key':24} {'tier':5} {'GB':>10}  path")
    for entry in report["entries"]:
        size = entry["size_gb"]
        shown = f"{size:,.1f}" if isinstance(size, (int, float)) else "-"
        flag = "" if entry["on_expected_tier"] is not False else "  !! wrong tier"
        lines.append(f"  {entry['key']:24} {entry['tier']:5} {shown:>10}  {entry['path']}{flag}")
    if report["problems"]:
        lines.append("")
        lines.append("  !! problems:")
        for problem in report["problems"]:
            lines.append(f"        {problem}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(paths_mod.DEFAULT_CONFIG))
    ap.add_argument("--set", action="append", default=[], dest="overrides")
    ap.add_argument("--no-measure", action="store_true",
                    help="skip the tree walk; only check devices and free space")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    cfg = config_mod.load(args.config, args.overrides)
    report = inspect(cfg, measure=not args.no_measure)
    print(json.dumps(report, indent=2, ensure_ascii=False) if args.json else render(report))
    return 2 if report["problems"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

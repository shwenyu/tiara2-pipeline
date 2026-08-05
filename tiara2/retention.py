"""Retention: decide which DERIVED artifacts are safe to delete, then delete
them safely.

Background
----------
One training round materialises several TB of derived data:

  flat_data       train+validation concatenated into 5 flat FASTA   (multi-TB)
  seq_pack        2-bit shared sequence pack                        (~0.8 TiB)
  feature_cache   dense fp32 k-mer matrices, per stage x k x split  (~6 TB)
  binned_shards   chop output; only needed until dedup+regroup ran
  scratch         *.tmp.* / *.partial left by an interrupted step

None of it is source data -- every item is reproducible from ``source_ready``.
What differs is the REBUILD COST, and that is the only thing that should drive
a keep/delete decision:

  DEAD       nothing reads it any more        -> always safe to delete
  CHEAP      rebuilt without re-reading the   -> delete by default
             multi-TB corpus
  EXPENSIVE  rebuild needs a full corpus pass -> keep unless explicitly asked

Why ``flat_data`` is DEAD
-------------------------
It used to be the input of the final trainer. Today ``train_tfidf``,
``build_features`` and ``hp_search`` all read ``train_ready``, and
``train_models`` is always invoked with ``--feature-cache``, which makes it
skip ``load_stage_sequences`` entirely. So the concatenated copy is written
every round and read by nobody.

Safety
------
Deletion is refused unless the resolved path lives INSIDE ``base`` and is not
one of the protected roots (source_ready / train_ready / tfidf / models /
results / logs / registry). A retention bug must never be able to eat the
corpus, so the guard is a whitelist of locations plus a blacklist of roots,
checked immediately before each ``rmtree``.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .inventory import DirStat, dir_stat, human

CAT_DEAD = "dead"
CAT_CHEAP = "cheap"
CAT_EXPENSIVE = "expensive"

# Order matters only for display.
CATEGORY_ORDER = (CAT_DEAD, CAT_CHEAP, CAT_EXPENSIVE)


@dataclass
class Item:
    """One deletable (or explicitly protected) derived artifact."""
    key: str
    path: str
    category: str
    reason: str
    rebuild: str
    stat: DirStat = field(default_factory=lambda: DirStat(path=""))
    selected: bool = False

    @property
    def exists(self) -> bool:
        return self.stat.exists

    def as_dict(self) -> dict:
        return {
            "key": self.key, "path": self.path, "category": self.category,
            "reason": self.reason, "rebuild": self.rebuild,
            "selected": self.selected, **self.stat.as_dict(),
        }


def _protected_roots(cfg: dict) -> list[str]:
    """Paths that must NEVER be removed by retention, at any policy."""
    train = cfg.get("train", {}) or {}
    publish = cfg.get("publish", {}) or {}
    candidates = [
        cfg.get("base"), cfg.get("data_home"), cfg.get("source_ready"),
        # corpus_ready is the deduped corpus: cheap to *delete*, but rebuilding
        # it means re-running chop_bin + dedup (tens of hours of mmseqs).
        cfg.get("corpus_ready"),
        cfg.get("results_root"), cfg.get("log_dir"), cfg.get("checkpoint_root"),
        train.get("train_ready"), train.get("tfidf_dir"), train.get("log_dir"),
        train.get("out_models"), publish.get("nnet_src"), publish.get("tfidf_src"),
    ]
    out = []
    for item in candidates:
        if item:
            out.append(str(Path(str(item))))
    return out


def is_safe_target(cfg: dict, path: str | Path) -> tuple[bool, str]:
    """Whitelist (inside base) + blacklist (protected roots) guard."""
    base = cfg.get("base")
    if not base:
        return False, "config has no 'base'"
    target = Path(str(path))
    root = Path(str(base))
    try:
        target.relative_to(root)
    except ValueError:
        return False, f"outside base ({root})"
    if str(target) == str(root):
        return False, "refusing to delete base itself"
    for guarded in _protected_roots(cfg):
        if str(target) == guarded:
            return False, f"protected root ({guarded})"
        # Also refuse anything that CONTAINS a protected root.
        try:
            Path(guarded).relative_to(target)
            return False, f"contains protected root ({guarded})"
        except ValueError:
            pass
    return True, ""


def _corpus_stage_done(cfg: dict, stage: str) -> bool:
    """True when ``stage`` has a manifest, i.e. it completed at least once."""
    work_root = cfg.get("work_root")
    if not work_root:
        return False
    return (Path(str(work_root)) / stage / "manifest.json").exists()


def _scratch_paths(cfg: dict) -> list[Path]:
    """Interrupted-step leftovers: ``*.tmp.*`` siblings and stale scratch dirs.

    ``TrainStage._build_flat`` and the publish step both write to a sibling
    ``<name>.tmp.<pid>.<ts>`` directory and rename it into place, so a crash
    leaves one behind. They are unreachable by name from any config key, hence
    this glob.
    """
    train = cfg.get("train", {}) or {}
    out: list[Path] = []
    seen: set[str] = set()
    roots = [cfg.get("base"), cfg.get("work_root"), cfg.get("results_root")]
    for candidate in (train.get("flat_data"), train.get("feature_cache"),
                      train.get("seq_pack")):
        if candidate:
            roots.append(str(Path(str(candidate)).parent))
    for root in roots:
        if not root:
            continue
        base = Path(str(root))
        if not base.is_dir():
            continue
        for pattern in ("*.tmp.*", "*.partial", "*.tmp"):
            try:
                for hit in base.glob(pattern):
                    key = str(hit)
                    if key not in seen:
                        seen.add(key)
                        out.append(hit)
            except OSError:
                continue
    return out


def plan(cfg: dict, *, max_entries: int | None = None) -> list[Item]:
    """Build the full retention plan for the CURRENT config.

    Sizes are probed with the capped walker, so this stays fast even when the
    feature cache holds terabytes across thousands of files.
    """
    train = cfg.get("train", {}) or {}
    kwargs = {} if max_entries is None else {"max_entries": max_entries}
    items: list[Item] = []

    def add(key: str, path, category: str, reason: str, rebuild: str) -> None:
        if not path:
            return
        items.append(Item(key=key, path=str(path), category=category,
                          reason=reason, rebuild=rebuild,
                          stat=dir_stat(str(path), **kwargs)))

    uses_cache = bool(train.get("feature_cache"))
    add("flat_data", train.get("flat_data"),
        CAT_DEAD if uses_cache else CAT_EXPENSIVE,
        "nothing reads it: every step uses train_ready or the feature cache"
        if uses_cache else "legacy no-cache path still reads this",
        "concatenate train+validation again (~1 corpus copy)")

    has_pack = bool(train.get("seq_pack"))
    add("feature_cache", train.get("feature_cache"), CAT_CHEAP,
        "dense fp32 matrices; regenerated on demand and signature-checked",
        "one pass over the 2-bit pack" if has_pack
        else "one full pass over the raw FASTA")

    add("seq_pack", train.get("seq_pack"), CAT_EXPENSIVE,
        "shared 2-bit pack; the durable resume artifact for featurization",
        "re-read the entire multi-TB FASTA corpus")

    work_root = cfg.get("work_root")
    if work_root:
        shards = Path(str(work_root)) / "chop_bin" / "binned"
        regrouped = _corpus_stage_done(cfg, "regroup")
        add("binned_shards", shards,
            CAT_DEAD if regrouped else CAT_EXPENSIVE,
            "regroup already rebuilt the FASTA from removed-ID sets"
            if regrouped else "dedup/regroup have not completed yet",
            "re-run chop_bin over the corpus")

    for path in _scratch_paths(cfg):
        add(f"scratch:{path.name}", path, CAT_DEAD,
            "leftover temporary directory from an interrupted step",
            "nothing to rebuild")

    return items


def select(cfg: dict, items: list[Item]) -> list[Item]:
    """Mark items the configured policy would delete.

    ``retention.delete`` lists keys to remove, ``retention.keep`` always wins.
    A ``scratch:*`` entry is matched by the bare ``scratch`` key so the list
    stays readable in YAML.
    """
    policy = cfg.get("retention", {}) or {}
    delete = set(policy.get("delete", []) or [])
    keep = set(policy.get("keep", []) or [])
    for item in items:
        family = item.key.split(":", 1)[0]
        wanted = item.key in delete or family in delete
        blocked = item.key in keep or family in keep
        safe, _ = is_safe_target(cfg, item.path)
        item.selected = bool(wanted and not blocked and item.exists and safe)
    return items


def render(items: list[Item], *, title: str = "retention plan") -> str:
    """Human-readable table, grouped by category, biggest first."""
    lines = [f"[{title}]"]
    if not items:
        return lines[0] + "\n  (nothing derived on disk yet)"
    total_sel = 0
    for category in CATEGORY_ORDER:
        group = [i for i in items if i.category == category]
        if not group:
            continue
        group.sort(key=lambda i: i.stat.bytes, reverse=True)
        label = {CAT_DEAD: "DEAD (safe to delete)",
                 CAT_CHEAP: "CHEAP to rebuild",
                 CAT_EXPENSIVE: "EXPENSIVE to rebuild"}[category]
        lines.append(f"  {label}")
        for item in group:
            if not item.exists:
                lines.append(f"    [ ] {item.key:<22} (absent)")
                continue
            mark = "x" if item.selected else " "
            if item.selected:
                total_sel += item.stat.bytes
            lines.append(
                f"    [{mark}] {item.key:<22} {item.stat.size_h:>12}"
                f"  {item.stat.files:>9,} files  {item.path}")
            lines.append(f"        why: {item.reason}")
            lines.append(f"        rebuild: {item.rebuild}")
    lines.append(f"  => marked for deletion: {human(total_sel)}")
    return "\n".join(lines)


def reclaimable(items: list[Item]) -> int:
    return sum(i.stat.bytes for i in items if i.selected)


def apply(cfg: dict, items: list[Item], *, dry_run: bool = True,
          log=None) -> dict:
    """Delete every selected item, re-checking the safety guard per path.

    The guard is re-evaluated here (not just at plan time) because the plan may
    have been built minutes earlier, and because ``apply`` is the last line of
    defence between a config typo and the corpus.
    """
    removed, skipped = [], []
    freed = 0
    for item in items:
        if not item.selected:
            continue
        safe, why = is_safe_target(cfg, item.path)
        if not safe:
            skipped.append({"path": item.path, "reason": why})
            if log:
                log.warning("retention: refusing %s (%s)", item.path, why)
            continue
        if dry_run:
            removed.append(item.path)
            freed += item.stat.bytes
            continue
        target = Path(item.path)
        try:
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
        except OSError as exc:
            skipped.append({"path": item.path, "reason": str(exc)})
            if log:
                log.warning("retention: failed to remove %s: %s", item.path, exc)
            continue
        removed.append(item.path)
        freed += item.stat.bytes
        if log:
            log.info("retention: removed %s (%s)", item.path,
                     human(item.stat.bytes))
    return {"removed": removed, "skipped": skipped, "freed_bytes": freed,
            "dry_run": dry_run}

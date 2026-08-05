"""Unified pipeline entrypoint.

    python -m tiara2.cli list
    python -m tiara2.cli status --data
    python -m tiara2.cli census                   # per-class records/bp/fragments
    python -m tiara2.cli census --exact --json census.json
    python -m tiara2.cli gc                       # report only, deletes nothing
    python -m tiara2.cli gc --apply
    python -m tiara2.cli version
    python -m tiara2.cli run --config config/config.yaml
    python -m tiara2.cli run --version-tag v2_1_hybrid
    python -m tiara2.cli run --only chop_bin
    python -m tiara2.cli run --from chop_bin --to regroup --dry-run
    python -m tiara2.cli run --set dedup.min_seq_id=0.97 --debug

One CLI, one config, every stage behind the same contract. --dry-run,
--from/--to, --only, --force and --set give a full debug + upgrade surface
without editing code.

NON-INTERACTIVE BY DEFAULT
--------------------------
Real runs are launched under nohup, where stdin is not a terminal. Nothing here
ever blocks on input unless stdin AND stdout are both a TTY; otherwise prompts
resolve to their documented default and the run proceeds. Destructive work
(``gc``) additionally requires an explicit ``--apply``, so a redirected session
can never silently delete terabytes.
"""
from __future__ import annotations

import argparse
import os
import importlib
import json
import sys
from pathlib import Path

from . import census as _census
from . import config as _config
from . import paths as _paths
from . import inventory as _inventory
from . import retention as _retention
from . import baseline as _baseline
from . import stages as _stages_pkg  # noqa: F401  (imports register stages)
from . import versioning as _versioning
from .stage import build_context, registry
from .progress import add_progress_args, progress_from_args

# Canonical order. Editing this list is the ONLY place stage order lives.
DEFAULT_ORDER = [
    "acquire",       # wraps ncbi_pipeline.py (fetch/qc/select/download/split)
    "curate",        # genus-level genome selection + per-genome bp cap (a PLAN)
    "chop_bin",      # chop + write directly into length-bin shards + metadata
    "dedup",         # per-bin cross-split double-95 / short-cov
    "regroup",       # rebuild per-species/per-split FASTA from surviving IDs
    "train",         # wraps 05_train (TF-IDF + dynamic-GPU HP + final NNet)
    "publish",       # closed loop: copy trained params from /data into tiara/models/
    "infer",         # classify with the pack publish just installed (opt-in)
]

# Which layer each stage belongs to -- see tiara2/versioning.py for why the
# corpus and training layers carry separate tags.
CORPUS_STAGES = ("acquire", "curate", "chop_bin", "dedup", "regroup")
TRAINING_STAGES = ("train", "publish", "infer")


def _load_stages():
    """Import stage modules so their @register decorators run."""
    for name in DEFAULT_ORDER:
        try:
            importlib.import_module(f"tiara2.stages.{name}")
        except ModuleNotFoundError:
            pass


def _resolve_order(args) -> list[str]:
    only = getattr(args, "only", None)
    if only:
        # `--only chop_bin,regroup` is the natural way to ask for two adjacent
        # stages and used to be accepted silently and then looked up as one
        # stage named "chop_bin,regroup", which fell through to
        # "stage not implemented yet ... (skipping)" -- a no-op run that looks
        # like a successful one. Split, validate, and always emit in canonical
        # order so `--only regroup,chop_bin` cannot invert the DAG.
        wanted = [part.strip() for part in str(only).split(",") if part.strip()]
        unknown = [w for w in wanted if w not in DEFAULT_ORDER]
        if unknown:
            raise SystemExit(
                f"[tiara2] unknown stage(s): {', '.join(unknown)}\n"
                f"         known stages: {', '.join(DEFAULT_ORDER)}")
        return [s for s in DEFAULT_ORDER if s in set(wanted)]
    order = [s for s in DEFAULT_ORDER if s in registry()]
    if getattr(args, "from_stage", None):
        order = order[order.index(args.from_stage):]
    if getattr(args, "to_stage", None):
        order = order[: order.index(args.to_stage) + 1]
    return order


def _resolve_config_path(path):
    """Resolve --config against the cwd, then against the repo.

    The default is the relative string "config/config.yaml", which only works
    when the cwd happens to be the checkout. Running from the DATA directory
    is a completely reasonable thing to do and used to fail with a bare
    FileNotFoundError, so fall back to the repo root before giving up.
    """
    p = Path(os.path.expanduser(str(path)))
    if p.exists():
        return str(p)
    if not p.is_absolute():
        alt = _paths.repo_root() / p
        if alt.exists():
            return str(alt)
    raise SystemExit(
        f"[tiara2] config not found: {path}\n"
        f"         tried {p.resolve()}\n"
        f"         and   {_paths.repo_root() / p}\n"
        f"         pass an absolute path with --config, or cd into "
        f"{_paths.repo_root()}")


def _load_cfg(args) -> dict:
    """Load the config, folding --version-tag / --corpus-tag into --set.

    The tags must be injected BEFORE templating: nearly every path in the
    config is derived from them via ``{version_tag}`` / ``{corpus_tag}``, so
    overriding them after resolution would leave the paths pointing at the old
    round.
    """
    overrides = list(getattr(args, "overrides", []) or [])
    if getattr(args, "version_tag", None):
        tag = _versioning.validate_tag(args.version_tag)
        overrides.append(f"version_tag={tag}")
    if getattr(args, "corpus_tag", None):
        tag = _versioning.validate_tag(args.corpus_tag)
        overrides.append(f"corpus_tag={tag}")
    return _config.load(_resolve_config_path(args.config),
                        overrides=overrides)


def _tag_banner(cfg: dict) -> str:
    previous = _versioning.last_run(cfg)
    lines = [
        "[tiara2] corpus_tag  = %s   (acquire/chop_bin/dedup/regroup)"
        % _versioning.corpus_tag(cfg),
        "[tiara2] version_tag = %s   (train/publish)"
        % _versioning.version_tag(cfg),
        "[tiara2] model_tag   = %s" % cfg.get("model_tag", "?"),
    ]
    if previous:
        lines.append("[tiara2] last run    = %s at %s"
                     % (previous.get("version_tag", "?"),
                        previous.get("started_at", "?")))
    else:
        lines.append("[tiara2] last run    = (none recorded)")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# subcommands
# --------------------------------------------------------------------------

def _cmd_list(args) -> int:
    _load_stages()
    for name in DEFAULT_ORDER:
        mark = "o" if name in registry() else " "
        layer = "corpus" if name in CORPUS_STAGES else "training"
        print(f"  [{mark}] {name:<10} ({layer})")
    try:
        from .model_backend import available_backends
        print(f"  model backends: {', '.join(available_backends())}")
    except Exception:
        pass
    return 0


def _stage_status(cfg: dict) -> list[dict]:
    """Read each stage manifest: has it run, when, on which fingerprint.

    Six small JSON reads -- no walking, safe to call on every status.
    """
    work_root = Path(str(cfg.get("work_root", ".")))
    rows = []
    for name in DEFAULT_ORDER:
        path = work_root / name / "manifest.json"
        row = {"stage": name, "ran": False, "finished_at": "-",
               "fingerprint": "-"}
        if path.exists():
            try:
                data = json.loads(path.read_text())
                row["ran"] = True
                row["finished_at"] = str(data.get("finished_at", "-"))
                row["fingerprint"] = str(data.get("fingerprint", "-"))[:12]
            except (OSError, json.JSONDecodeError):
                row["finished_at"] = "(unreadable manifest)"
        rows.append(row)
    return rows


def _fragment_bp_balance_status(cfg: dict) -> dict:
    """Cheap status for the external fragment-bp balancing checkpoint.

    The sampler is intentionally a data-only bridge between ``regroup`` and
    ``train``, not a corpus stage: changing its priors must not re-run chop or
    dedup. Completion is therefore defined by its own atomic outputs rather
    than a work_root stage manifest.
    """
    section = cfg.get("fragment_bp_balance", {}) or {}
    out = Path(str(section.get("output_root", "")))
    # v2.1.3: EVERY balanced split must exist, not just train. v2.1.2 reported
    # "done" while validation was still an unbalanced symlink to the raw
    # corpus, which is precisely the state that produced the collapse.
    balanced = [str(s) for s in (section.get("balance_splits")
                                 or ["train", "validation"])]
    row = {"enabled": bool(section.get("enabled", False)), "state": "disabled",
           "output_root": str(out), "detail": "-",
           "balanced_splits": balanced}
    if not row["enabled"]:
        return row
    row["state"] = "pending"
    plan = out / str(section.get("plan_file", "sampling_plan.json"))
    report = out / str(section.get("report_file", "sampling_report.tsv"))
    classes = list(cfg.get("classes", []))
    expected = [out / split / f"{c}.fasta"
                for split in balanced for c in classes]
    present = sum(p.is_file() for p in expected)
    if not out.exists():
        row["detail"] = "output directory missing"
        return row
    if not plan.is_file() or not report.is_file():
        row["state"] = "incomplete"
        row["detail"] = "missing sampling_plan.json or sampling_report.tsv"
        return row
    if present != len(expected):
        missing = [str(p.relative_to(out)) for p in expected if not p.is_file()]
        row["state"] = "incomplete"
        row["detail"] = (f"FASTAs {present}/{len(expected)} "
                         f"({'+'.join(balanced)}); missing "
                         f"{', '.join(missing[:4])}"
                         + (" ..." if len(missing) > 4 else ""))
        return row
    # A balanced split must be a real directory of real files. If it is a
    # symlink, it is still pointing at the raw corpus.
    linked = [s for s in balanced if (out / s).is_symlink()]
    if linked:
        row["state"] = "incomplete"
        row["detail"] = (f"{', '.join(linked)} is a symlink to the unbalanced "
                         f"corpus; re-run `tiara2 bp-balance --force`")
        return row
    try:
        data = json.loads(plan.read_text())
        target = int(data.get("target_total_bp", 0))
        kept = sum(int(v.get("kept_bp", 0))
                   for v in (data.get("strata", {}) or {}).values())
        row["state"] = "done"
        row["detail"] = (f"FASTAs {present}/{len(expected)} "
                         f"({'+'.join(balanced)}); "
                         f"kept {kept:,} / target {target:,} bp")
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        row["state"] = "incomplete"
        row["detail"] = "sampling plan unreadable"
    return row


def _cmd_bp_balance(args) -> int:
    """Run the fragment bp-balancing checkpoint from the config.

    The sampler used to be launched by a hand-maintained shell wrapper whose
    flags could (and did) drift from config/config.yaml -- notably the split
    selection. Everything now comes from ONE resolved config.
    """
    cfg = _load_cfg(args)
    section = cfg.get("fragment_bp_balance", {}) or {}
    if not section.get("enabled", False) and not args.force:
        print("fragment_bp_balance.enabled is false; nothing to do "
              "(pass --force to run anyway)")
        return 0

    source = Path(str(section.get("source_root", ""))).expanduser()
    out = Path(str(section.get("output_root", ""))).expanduser()
    if not source.is_dir():
        print(f"[FATAL] source_root does not exist: {source}", file=sys.stderr)
        return 2
    if not str(out):
        print("[FATAL] fragment_bp_balance.output_root is empty",
              file=sys.stderr)
        return 2
    # Refuse to write into the corpus we are reading, symlinks included.
    if source.resolve() == out.resolve():
        print(f"[FATAL] output_root resolves to source_root ({source.resolve()})",
              file=sys.stderr)
        return 2

    balanced = [str(s) for s in (section.get("balance_splits")
                                 or ["train", "validation"])]
    for split in balanced:
        for cls in cfg.get("classes", []):
            src = source / split / f"{cls}.fasta"
            if not src.exists():          # .exists() follows symlinks
                print(f"[FATAL] missing input (or broken symlink): {src}",
                      file=sys.stderr)
                return 2

    script = _paths.repo_root() / "scripts" / "sample_fragments_by_bp.py"
    cmd = [sys.executable, "-u", str(script),
           "--source-root", str(source),
           "--out-root", str(out),
           "--balance-splits", ",".join(balanced),
           "--eval-mode", str(section.get("eval_mode", "symlink")),
           "--seed", str(int(section.get("seed", 42))),
           "--min-len", str(int(section.get("min_len", 1000))),
           "--min-acgt-purity", str(float(section.get("min_acgt_purity", 0.9))),
           "--workers", str(int(section.get("workers", 16))),
           "--euk-workers", str(int(section.get("euk_workers", 12))),
           "--io-buffer-mb", str(int(section.get("io_buffer_mb", 32))),
           "--plan-file", str(section.get("plan_file", "sampling_plan.json")),
           "--report-file", str(section.get("report_file",
                                            "sampling_report.tsv"))]
    # Priors/weights live in the config; pass them through verbatim so the
    # script's built-in defaults can never silently disagree with it.
    priors = section.get("class_priors")
    if priors:
        cmd += ["--class-priors", json.dumps(priors)]
    euk_weights = section.get("euk_weights")
    if euk_weights:
        cmd += ["--euk-weights", json.dumps(euk_weights)]
    if section.get("target_total_bp"):
        cmd += ["--target-total-bp", str(int(section["target_total_bp"]))]
    taxdump = str(section.get("taxdump_dir", "") or "")
    if taxdump:
        cmd += ["--taxdump-dir", taxdump]
    acc_tsv = str(section.get("accession_taxid_tsv", "") or "")
    if acc_tsv:
        cmd += ["--accession-taxid-tsv", acc_tsv]
    if section.get("require_clades", False):
        cmd.append("--require-clades")
    if args.force:
        cmd.append("--force")

    print("[bp-balance] source        : %s" % source)
    print("[bp-balance] output        : %s" % out)
    print("[bp-balance] balanced      : %s" % ",".join(balanced))
    print("[bp-balance] other splits  : %s" % section.get("eval_mode",
                                                          "symlink"))
    print("[bp-balance] clade source  : %s" % (taxdump or "header sg= only"))
    if args.dry_run:
        print("[dry-run] " + " ".join(cmd))
        return 0
    import subprocess
    return subprocess.run(cmd).returncode


def _cmd_status(args) -> int:
    cfg = _load_cfg(args)
    print(_tag_banner(cfg))
    print("")

    free = _inventory.disk_free(cfg.get("base", "."))
    if free.get("ok"):
        print("[disk] %s  free %s of %s (%.1f%% used)"
              % (cfg.get("base"), _inventory.human(free["free"]),
                 _inventory.human(free["total"]), free["pct_used"]))
        print("")

    print("[corpus] %s" % cfg.get("source_ready", "?"))
    inv = _inventory.corpus_inventory(cfg)
    print(_inventory.render_corpus(inv))
    print("  total: %s across %d/%d files"
          % (_inventory.human(inv["total_bytes"]), inv["present"],
             inv["expected"]))
    print("")

    print("[stages] work_root = %s" % cfg.get("work_root", "?"))
    for row in _stage_status(cfg):
        mark = "done" if row["ran"] else "----"
        print("  %-4s %-10s %-26s %s"
              % (mark, row["stage"], row["finished_at"], row["fingerprint"]))
    print("")

    bp = _fragment_bp_balance_status(cfg)
    print("[fragment_bp_balance] %s  (balanced: %s)"
          % (bp["output_root"], ",".join(bp.get("balanced_splits", []) or ["-"])))
    print("  %-10s %s" % (bp["state"], bp["detail"]))
    print("")

    if args.data:
        home = cfg.get("data_home")
        if home:
            print("[data_home] %s  (one level, no recursive sizing)" % home)
            children = _inventory.shallow_children(home)
            if not children:
                print("  (empty or unreadable)")
            for row in children:
                kind = "dir " if row["is_dir"] else "file"
                count = f"{row['children']:,} entries" if row["is_dir"] else ""
                print("  %s %-28s %s" % (kind, row["name"][:28], count))
            print("")

    if args.sizes:
        items = _retention.select(cfg, _retention.plan(cfg))
        print(_retention.render(items, title="derived artifacts"))
    else:
        print("[derived] pass --sizes to measure flat_data / feature_cache / "
              "seq_pack (walks the tree), or run: tiara2 gc")
    return 0


def _cmd_census(args) -> int:
    """Per-class data VOLUME -- the pre-deployment check.

    ``status`` answers "how many bytes are on disk" in 15 stat() calls. That is
    the wrong unit for a curation or subsampling decision: the training budget
    is denominated in FRAGMENTS, and bytes -> fragments runs through record
    count and sequence length. This command measures those, projects fragments
    at the configured chop length, compares each class against its target
    prior, and reports whether a download is currently writing -- because a
    census taken mid-download is a moving target.

    Default mode samples the head of each FASTA and extrapolates by byte ratio
    (a full scan is a multi-TB read). ``--exact`` reads everything.
    """
    cfg = _load_cfg(args)
    print(_tag_banner(cfg))
    print("")
    cen_cfg = (cfg.get("census", {}) or {})
    sample_mb = args.sample_mb or int(cen_cfg.get("sample_mb",
                                                  _census.DEFAULT_SAMPLE_MB))
    window = float(cen_cfg.get("active_window_s",
                               _census.DEFAULT_ACTIVE_WINDOW))
    progress = progress_from_args(args, label="census")
    snapshot = _census.run_census(cfg, exact=args.exact, sample_mb=sample_mb,
                                  window_s=window, progress=progress)
    print(_census.render_census(snapshot["census"], snapshot["projection"],
                                snapshot["activity"]))
    print("")
    if args.json:
        Path(args.json).write_text(json.dumps(snapshot, indent=2,
                                              sort_keys=True))
        print("[census] wrote %s" % args.json)
    if snapshot["activity"]["active"]:
        # Exit 0 either way: this is an observation, not a failure. The message
        # is what matters, and a non-zero exit would break `tiara2 census` in
        # any wrapper script that checks return codes.
        print("[census] a download looks ACTIVE -- treat these numbers as a "
              "lower bound, and do not freeze a curation plan yet "
              "(curate defers automatically).")
    else:
        print("[census] corpus looks settled; safe to run: tiara2 run --only curate")
    return 0


def _cmd_gc(args) -> int:
    cfg = _load_cfg(args)
    print(_tag_banner(cfg))
    print("")
    items = _retention.plan(cfg)
    if args.all_rebuildable:
        # Widen the selection to everything reproducible, including the
        # expensive pack -- an explicit "I am done with this corpus" switch.
        for item in items:
            safe, _ = _retention.is_safe_target(cfg, item.path)
            item.selected = item.exists and safe
    else:
        _retention.select(cfg, items)
    print(_retention.render(items))
    print("")

    if not args.apply:
        print("[gc] report only. Re-run with --apply to delete the [x] items.")
        return 0

    total = _retention.reclaimable(items)
    if total <= 0:
        print("[gc] nothing selected; no action taken.")
        return 0
    approved = args.yes or _versioning.confirm(
        f"[gc] permanently delete the {sum(1 for i in items if i.selected)} "
        f"marked items ({_inventory.human(total)})?", default=False)
    if not approved:
        print("[gc] not confirmed (non-interactive sessions need --yes); "
              "nothing deleted.")
        return 0
    result = _retention.apply(cfg, items, dry_run=False)
    print("[gc] removed %d paths, freed %s"
          % (len(result["removed"]), _inventory.human(result["freed_bytes"])))
    for skip in result["skipped"]:
        print("[gc] SKIPPED %s (%s)" % (skip["path"], skip["reason"]))
    return 0


def _cmd_freeze_baseline(args) -> int:
    cfg = _load_cfg(args)
    try:
        result = _baseline.freeze(
            cfg, _paths.repo_root(), tag=args.tag, freeze_dir=args.freeze_dir,
            metadata_only=args.metadata_only, force=args.force)
    except _baseline.BaselineError as exc:
        print(f"[freeze-baseline] ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"[freeze-baseline] frozen {result['tag']}")
    print(f"  freeze_id : {result['freeze_id']}")
    directory = args.freeze_dir or (cfg.get('baseline_freeze', {}) or {}).get('freeze_dir')
    print(f"  directory : {directory}")
    print("  No data was deleted. Run cleanup-baseline without --apply next.")
    return 0


def _cmd_cleanup_baseline(args) -> int:
    cfg = _load_cfg(args)
    freeze_dir = args.freeze_dir or (cfg.get("baseline_freeze", {}) or {}).get("freeze_dir")
    if not freeze_dir:
        print("[cleanup-baseline] --freeze-dir is required", file=sys.stderr)
        return 2
    try:
        if args.apply:
            plan = _baseline.load_plan(freeze_dir)
            if plan.get("profile") != args.profile or bool(plan.get("release_upstream")) != bool(args.release_upstream):
                raise _baseline.BaselineError(
                    "apply flags do not match the reviewed cleanup_plan.json")
        else:
            plan = _baseline.cleanup_plan(
                cfg, _paths.repo_root(), freeze_dir, profile=args.profile,
                release_upstream=args.release_upstream)
        print(_baseline.render_plan(plan))
        print(f"  plan file: {Path(freeze_dir) / _baseline.PLAN_FILE}")
        if args.apply:
            receipt = _baseline.apply_cleanup(
                plan, apply=True, yes=args.yes,
                acknowledge_upstream_loss=args.acknowledge_upstream_loss)
            print(f"[cleanup-baseline] removed {len(receipt['removed'])} paths; "
                  f"freed {receipt['freed_bytes'] / 1024**3:.2f} GiB")
        else:
            print("[cleanup-baseline] dry plan only; nothing deleted")
    except _baseline.BaselineError as exc:
        print(f"[cleanup-baseline] ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


def _cmd_version(args) -> int:
    cfg = _load_cfg(args)
    print(_tag_banner(cfg))
    print("")
    print("[registry] %s" % _versioning.registry_path(cfg))
    print(_versioning.render_registry(cfg))
    print("")
    print("[next] suggested version_tag: %s"
          % _versioning.suggest_next(_versioning.version_tag(cfg)))
    return 0


# --------------------------------------------------------------------------
# plan-subsample
# --------------------------------------------------------------------------

def _load_seqpack():
    """Load seqpack by file path, bypassing the tiara package __init__.

    Going through `from tiara.training import seqpack` would import the whole
    vendored package (tqdm/Bio/torch), which is far more than this command
    needs and is not importable in a bare environment.
    """
    import importlib.util
    from . import paths

    path = paths.repo_root() / "tiara" / "training" / "seqpack.py"
    spec = importlib.util.spec_from_file_location("_tiara2_seqpack", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load seqpack from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _parse_counts(text: str) -> dict:
    counts = {}
    for part in str(text or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(f"bad --counts entry {part!r}; expected class=rows")
        cls, _, num = part.partition("=")
        try:
            counts[cls.strip().lower()] = int(float(num))
        except ValueError:
            raise SystemExit(f"bad row count in {part!r}")
    if not counts:
        raise SystemExit("--counts is empty")
    return counts


def _cmd_plan_subsample(args) -> int:
    seqpack = _load_seqpack()
    counts = _parse_counts(args.counts)
    rates = seqpack.balance_rates(
        counts, mode=args.mode, target_total=args.target_rows,
        base_rate=args.base_rate, max_rate=args.max_rate,
        min_rows=args.min_rows)
    rows = seqpack.expected_rows(counts, rates)
    total_in = sum(counts.values()) or 1
    total_out = sum(rows.values()) or 1

    print(f"[plan-subsample] mode={args.mode} target_rows="
          f"{args.target_rows if args.target_rows else '(base_rate)'}")
    print("")
    print(f"  {'class':<16}{'available':>14}{'rate':>10}"
          f"{'kept':>14}{'before':>9}{'after':>9}")
    for cls in sorted(counts, key=lambda c: -counts[c]):
        before = 100.0 * counts[cls] / total_in
        after = 100.0 * rows.get(cls, 0) / total_out
        print(f"  {cls:<16}{counts[cls]:>14,}{rates.get(cls, 1.0):>10.4f}"
              f"{rows.get(cls, 0):>14,}{before:>8.2f}%{after:>8.2f}%")
    print(f"  {'TOTAL':<16}{total_in:>14,}{'':>10}{total_out:>14,}")
    print("")
    if any(r >= 1.0 for r in rates.values()):
        capped = [c for c, r in rates.items() if r >= 1.0]
        print(f"  note: {', '.join(sorted(capped))} capped at rate 1.0 "
              "(cannot oversample; rows are only ever dropped)")
    print("  paste into config.yaml under train.subsample.train:")
    print("    class_rates:")
    for cls in sorted(rates):
        print(f"      {cls}: {rates[cls]}")
    print("")
    print("  NOTE: changing class_rates invalidates the TRAIN feature cache "
          "(val/test are untouched).")
    return 0


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------

def _preflight(cfg: dict, args, order: list[str]) -> None:
    """Report tags and derived-data footprint before any stage runs.

    Policy (``retention.policy``):
      off     -- skip entirely (no tree walking at all)
      report  -- print the plan, delete nothing            [default]
      prompt  -- print, then ask; a non-TTY session declines
      auto    -- print, then delete the configured selection
    """
    policy = str((cfg.get("retention", {}) or {}).get("policy", "report")).lower()
    if policy == "off":
        return
    items = _retention.select(cfg, _retention.plan(cfg))
    total = _retention.reclaimable(items)
    print("")
    print(_retention.render(items, title="derived data from previous rounds"))
    print("")
    if total <= 0 or policy == "report":
        if total > 0:
            print("[retention] policy=report: keeping everything. "
                  "Run `tiara2 gc --apply` to reclaim %s."
                  % _inventory.human(total))
        return
    if args.dry_run:
        print("[retention] dry-run: would reclaim %s" % _inventory.human(total))
        return
    approved = args.yes
    if not approved and policy == "prompt":
        approved = _versioning.confirm(
            "[retention] delete the marked stale artifacts (%s) before this "
            "round?" % _inventory.human(total), default=False)
    elif not approved and policy == "auto":
        approved = True
    if not approved:
        print("[retention] keeping everything for this round.")
        return
    result = _retention.apply(cfg, items, dry_run=False)
    print("[retention] freed %s" % _inventory.human(result["freed_bytes"]))


def _cmd_run(args) -> int:
    cfg = _load_cfg(args)
    order = _resolve_order(args)

    # A full run starting at the first stage is the moment to pin down which
    # version this is. Only then, only on a TTY, and only if not already given
    # explicitly on the command line.
    wants_prompt = bool((cfg.get("versioning", {}) or {}).get("prompt", True))
    full_run = order and order[0] == DEFAULT_ORDER[0]
    if wants_prompt and full_run and not args.version_tag and not args.dry_run:
        chosen = _versioning.prompt_for_version(cfg, assume_yes=args.yes)
        if chosen and chosen != _versioning.version_tag(cfg):
            args.version_tag = chosen
            cfg = _load_cfg(args)
            print("[tiara2] version_tag set to %s" % chosen)

    print(_tag_banner(cfg))
    print(f"[tiara2] plan: {' -> '.join(order)}")
    _preflight(cfg, args, order)

    if not args.dry_run:
        _versioning.register_run(cfg, stages=order, note=args.note or "")
        snapshot = _versioning.snapshot_config(cfg)
        if snapshot:
            print("[tiara2] config snapshot -> %s" % snapshot)

    reg = registry()
    for name in order:
        if name not in reg:
            print(f"[tiara2] !! stage not implemented yet: {name} (skipping)")
            continue
        stage = reg[name]()
        ctx = build_context(cfg, name, dry_run=args.dry_run,
                            force=args.force, debug=args.debug)
        stage.execute(ctx)
    return 0


def _cmd_classify(args) -> int:
    """Unified inference entry point: one command, one resolved model set."""
    from . import inference as _inference

    if args.manifest:
        plan = _inference.InferencePlan.from_json(Path(args.manifest).read_text())
        if args.threads:
            plan.threads = int(args.threads)
        if args.batch_records:
            plan.batch_records = int(args.batch_records)
        if args.device:
            plan.device = args.device
        if args.min_len:
            plan.min_len = int(args.min_len)
    else:
        cutoffs = {}
        if args.prob_cutoff_first is not None:
            cutoffs["first"] = float(args.prob_cutoff_first)
        if args.prob_cutoff_second is not None:
            cutoffs["second"] = float(args.prob_cutoff_second)
        kwargs = dict(
            k_first=args.k_first,
            k_second=args.k_second,
            cutoffs=cutoffs,
            min_len=args.min_len or 3000,
            threads=args.threads or 1,
            batch_records=args.batch_records or 512,
            device=args.device,
        )
        if args.nnet_dir or args.tfidf_dir:
            if not (args.nnet_dir and args.tfidf_dir):
                print("--nnet-dir and --tfidf-dir must be given together",
                      file=sys.stderr)
                return 2
            plan = _inference.build_plan(
                args.nnet_dir, args.tfidf_dir,
                model_tag=args.model_tag or "", **kwargs)
        else:
            cfg = _load_cfg(args)
            if args.model_tag:
                cfg.setdefault("publish", {})["model_tag"] = args.model_tag
            plan = _inference.plan_from_config(cfg, **kwargs)

    for name in _inference.STAGES:
        spec = plan.stages[name]
        print(f"[{name}] k={spec.k} cutoff={spec.prob_cutoff:.6f} "
              f"weights={Path(spec.weights).name} tfidf={Path(spec.tfidf).name}"
              + (f" mean_f1={spec.mean_f1:.4f}" if spec.mean_f1 is not None else ""))

    if args.emit_manifest:
        Path(args.emit_manifest).write_text(plan.to_json())
        print(f"wrote inference manifest: {args.emit_manifest}")

    if not args.input:
        if not args.emit_manifest:
            print("nothing to do: pass -i/--input or --emit-manifest",
                  file=sys.stderr)
            return 2
        return 0

    out = args.output or (str(Path(args.input).with_suffix("")) + ".tiara.tsv")
    log_path = args.log
    if log_path is None and not args.no_log:
        out_p = Path(out)
        log_path = str(out_p.with_name("log_" + out_p.stem + ".txt"))
    counts = _inference.run(
        plan, args.input, out,
        probabilities=args.probabilities,
        verbose=args.verbose,
        to_fasta=args.to_fasta,
        gzip_output=args.gzip,
        log_path=log_path,
    )
    print(f"wrote {counts['records']} records to {counts['output']}")
    for name, value in sorted((counts.get("per_class") or {}).items(),
                              key=lambda kv: -kv[1]):
        print(f"  {name:<14} {value:>12,}")
    for path in counts.get("fasta_paths") or []:
        print(f"  wrote fasta: {path}")
    if counts.get("log"):
        print(f"  wrote log: {counts['log']}")
    return 0


def _add_config_args(parser) -> None:
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--set", action="append", default=[], dest="overrides",
                        help="config override a.b=value (repeatable)")
    parser.add_argument("--version-tag", dest="version_tag",
                        help="training-layer tag (train/publish paths)")
    parser.add_argument("--corpus-tag", dest="corpus_tag",
                        help="corpus-layer tag (acquire/chop/dedup/regroup)")


def main(argv=None) -> int:
    _load_stages()
    ap = argparse.ArgumentParser(prog="tiara2", description="Tiara2 unified pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list registered stages in run order")

    st = sub.add_parser("status", help="fast inventory of corpus + stages")
    _add_config_args(st)
    st.add_argument("--data", action="store_true",
                    help="also list data_home one level deep")
    st.add_argument("--sizes", action="store_true",
                    help="measure derived artifacts (walks the tree)")

    cen = sub.add_parser("census",
                         help="per-class records/bp/fragments + download guard")
    _add_config_args(cen)
    cen.add_argument("--exact", action="store_true",
                     help="read every byte instead of sampling (slow)")
    cen.add_argument("--sample-mb", dest="sample_mb", type=int, default=0,
                     help="MiB read per FASTA in fast mode (default: config)")
    cen.add_argument("--json", help="also write the full snapshot as JSON")
    add_progress_args(cen)

    gc = sub.add_parser("gc", help="report / reclaim derived data")
    _add_config_args(gc)
    gc.add_argument("--apply", action="store_true", help="actually delete")
    gc.add_argument("--all-rebuildable", action="store_true",
                    help="also select expensive-to-rebuild items (seq_pack)")
    gc.add_argument("--yes", action="store_true", help="skip confirmation")

    fr = sub.add_parser("freeze-baseline",
                        help="freeze a verified reproducible model baseline")
    _add_config_args(fr)
    fr.add_argument("--tag", default="v2.2.0")
    fr.add_argument("--freeze-dir")
    fr.add_argument("--metadata-only", action="store_true",
                    help="skip hashing train/validation FASTA (not for upstream release)")
    fr.add_argument("--force", action="store_true")

    cb = sub.add_parser("cleanup-baseline",
                        help="plan/apply cleanup after a verified baseline freeze")
    _add_config_args(cb)
    cb.add_argument("--freeze-dir")
    cb.add_argument("--profile", choices=["safe", "reproducible"], default="safe")
    cb.add_argument("--release-upstream", action="store_true")
    cb.add_argument("--apply", action="store_true")
    cb.add_argument("--yes", action="store_true")
    cb.add_argument("--acknowledge-upstream-loss", action="store_true")

    bpb = sub.add_parser("bp-balance",
                         help="run the fragment bp-balancing checkpoint "
                              "(between regroup and train) from the config")
    _add_config_args(bpb)
    bpb.add_argument("--force", action="store_true",
                     help="rebuild the output root from scratch")
    bpb.add_argument("--dry-run", action="store_true",
                     help="print the command instead of running it")

    ver = sub.add_parser("version", help="show tags and the run registry")
    _add_config_args(ver)

    ps = sub.add_parser("plan-subsample",
                        help="compute per-class train rates from row counts")
    ps.add_argument("--counts", required=True,
                    help="class=rows pairs, e.g. eukarya=26000000,archaea=46000")
    ps.add_argument("--mode", default="sqrt",
                    choices=["none", "sqrt", "linear"],
                    help="rebalancing strength (default: sqrt)")
    ps.add_argument("--target-rows", dest="target_rows", type=int, default=None,
                    help="total rows to aim for (default: base_rate x available)")
    ps.add_argument("--base-rate", dest="base_rate", type=float, default=1.0,
                    help="uniform rate used when --target-rows is omitted")
    ps.add_argument("--max-rate", dest="max_rate", type=float, default=1.0,
                    help="per-class ceiling (never oversample above 1.0)")
    ps.add_argument("--min-rows", dest="min_rows", type=int, default=0,
                    help="floor on target rows per class")

    cl = sub.add_parser("classify",
                        help="classify a fasta with the published model pack")
    _add_config_args(cl)
    cl.add_argument("-i", "--input", help="fasta (optionally .gz) to classify")
    cl.add_argument("-o", "--output", help="output TSV (default: alongside input)")
    cl.add_argument("--model-tag", dest="model_tag",
                    help="published pack to use (default: publish.model_tag)")
    cl.add_argument("--nnet-dir", dest="nnet_dir",
                    help="use this nnet dir instead of the published pack")
    cl.add_argument("--tfidf-dir", dest="tfidf_dir",
                    help="use this tf-idf dir instead of the published pack")
    cl.add_argument("--k-first", dest="k_first", type=int,
                    help="pin the first-stage k (default: best recorded mean_f1)")
    cl.add_argument("--k-second", dest="k_second", type=int,
                    help="pin the second-stage k (default: best recorded mean_f1)")
    cl.add_argument("--prob-cutoff-first", dest="prob_cutoff_first", type=float,
                    help="first-stage threshold; RE-CALIBRATE per model pack")
    cl.add_argument("--prob-cutoff-second", dest="prob_cutoff_second", type=float,
                    help="second-stage threshold; RE-CALIBRATE per model pack")
    cl.add_argument("--min-len", dest="min_len", type=int,
                    help="skip sequences shorter than this (default 3000)")
    cl.add_argument("--threads", type=int, help="featurisation workers")
    cl.add_argument("--batch-records", dest="batch_records", type=int,
                    help="records per prediction batch (default 512)")
    cl.add_argument("--device", help="cuda | cpu (default: auto-detect)")
    cl.add_argument("--probabilities", action="store_true",
                    help="append per-class probabilities to the TSV")
    cl.add_argument("--manifest", help="reuse a previously emitted plan")
    cl.add_argument("--emit-manifest", dest="emit_manifest",
                    help="write the resolved plan as JSON and exit if no input")
    # Original tiara's --to_fasta / --tf vocabulary, kept verbatim.
    cl.add_argument("--to-fasta", "--tf", dest="to_fasta", nargs="+",
                    metavar="CLASS",
                    help="also write per-class fasta files; classes: "
                         "mit pla bac arc euk unk pro org all")
    cl.add_argument("--gz", "--gzip", dest="gzip", action="store_true",
                    help="gzip every output (adds .gz)")
    cl.add_argument("--log", help="model params + summary log "
                                  "(default: log_<output>.txt)")
    cl.add_argument("--no-log", dest="no_log", action="store_true",
                    help="do not write the summary log")
    cl.add_argument("--verbose", action="store_true")

    r = sub.add_parser("run", help="run stages")
    _add_config_args(r)
    r.add_argument("--only",
                   help="run only these stages (comma-separated, e.g. "
                        "chop_bin,regroup); always executed in DAG order")
    r.add_argument("--from", dest="from_stage", help="start at this stage")
    r.add_argument("--to", dest="to_stage", help="stop after this stage")
    r.add_argument("--note", default="", help="free-text note for the registry")
    r.add_argument("--yes", action="store_true",
                   help="assume yes for retention prompts (non-interactive)")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--force", action="store_true", help="ignore resume manifests")
    r.add_argument("--debug", action="store_true")

    args = ap.parse_args(argv)

    if args.cmd == "list":
        return _cmd_list(args)
    if args.cmd == "status":
        return _cmd_status(args)
    if args.cmd == "census":
        return _cmd_census(args)
    if args.cmd == "gc":
        return _cmd_gc(args)
    if args.cmd == "freeze-baseline":
        return _cmd_freeze_baseline(args)
    if args.cmd == "cleanup-baseline":
        return _cmd_cleanup_baseline(args)
    if args.cmd == "bp-balance":
        return _cmd_bp_balance(args)
    if args.cmd == "version":
        return _cmd_version(args)
    if args.cmd == "plan-subsample":
        return _cmd_plan_subsample(args)
    if args.cmd == "classify":
        return _cmd_classify(args)
    return _cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())

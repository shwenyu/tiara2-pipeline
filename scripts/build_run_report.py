#!/usr/bin/env python3
"""Build the per-run RECORD DOCUMENT for one training run.

Why this exists
---------------
After a run finishes, the facts that decide whether a model version is any good
are scattered across half a dozen artifacts written by different processes:

  * ``config.yaml``                     -- the data SELECTION strategy
  * ``<seq_pack>/<stage>/<split>/meta.json``   -- records kept after subsampling
  * ``<feature_cache>/<stage>/k*/meta.json``   -- rows x dims actually featurized
  * ``<tfidf_dir>/tfidf_manifest.json``        -- TF-IDF seconds + signature
  * ``<log_dir>/hp_<stage>_k<k>.json``         -- per-candidate GPU seconds + F1
  * ``<log_dir>/step_timings.jsonl``           -- per-step WALL time
  * ``<work_root>/<stage>/manifest.json``      -- stage fingerprints

Reading seven files by hand to compare v2.0 against v2.1 does not scale. This
script harvests all of them into ONE record, in three shapes:

  * Markdown -- for reading and archiving next to the model
  * JSON     -- for exact machine diffing between versions
  * CSV      -- ONE ROW PER RUN, appended, so versions line up side by side

Design constraints (matching scripts/publish_trained_models.py):
  * STDLIB ONLY. No PyYAML, no numpy. The resolved config is handed over as
    JSON by the publish stage, so this runs in any env.
  * NEVER FATAL on missing artifacts. A run that stopped after TF-IDF must
    still produce a report -- absent pieces are recorded as null / "-" rather
    than raising. The report is documentation, and documentation must not be
    able to fail a pipeline that already succeeded.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

REPORT_VERSION = 1

# featurize_cache writes its two splits under these tags (NOT 'validation').
FEATURE_TAGS = ("train", "val")
STAGES = ("first", "second")


# --------------------------------------------------------------------------- #
# tiny helpers -- every one of them returns a sentinel instead of raising
# --------------------------------------------------------------------------- #
def read_json(path) -> Any:
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def stat_of(path) -> dict | None:
    try:
        st = Path(path).stat()
    except OSError:
        return None
    return {"bytes": int(st.st_size), "mtime": iso(st.st_mtime)}


def iso(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(ts))


def human_bytes(n) -> str:
    if n is None:
        return "-"
    x = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if x < 1024.0 or unit == "PiB":
            return f"{int(x):,} B" if unit == "B" else f"{x:,.1f} {unit}"
        x /= 1024.0
    return "-"


def human_dur(seconds) -> str:
    if seconds is None:
        return "-"
    total = int(float(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def num(value, digits=4) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def count(value) -> str:
    if value is None:
        return "-"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def git_info(repo_root) -> dict:
    def run(*argv):
        try:
            out = subprocess.run(["git", "-C", str(repo_root), *argv],
                                 capture_output=True, text=True, timeout=15)
        except Exception:
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    porcelain = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": (bool(porcelain) if porcelain is not None else None),
    }


# --------------------------------------------------------------------------- #
# harvest: identity
# --------------------------------------------------------------------------- #
def harvest_identity(cfg: dict, repo_root: str) -> dict:
    return {
        "input_tag": cfg.get("input_tag"),
        "model_tag": cfg.get("model_tag"),
        "output_tag": cfg.get("output_tag"),
        "base": cfg.get("base"),
        "data_home": cfg.get("data_home"),
        "conda_env": cfg.get("conda_env"),
        "backend": (cfg.get("train") or {}).get("backend"),
        "host": os.uname().nodename if hasattr(os, "uname") else None,
        "git": git_info(repo_root),
    }


# --------------------------------------------------------------------------- #
# harvest: data SELECTION strategy (the "why these records" half of the record)
# --------------------------------------------------------------------------- #
def harvest_selection(cfg: dict) -> dict:
    train = cfg.get("train") or {}
    gpu = train.get("gpu") or {}
    return {
        "splits": cfg.get("splits"),
        "classes": cfg.get("classes"),
        "mag": cfg.get("mag"),
        "chop": cfg.get("chop"),
        "dedup": cfg.get("dedup"),
        "subsample": train.get("subsample"),
        "k_first": train.get("k_first"),
        "k_second": train.get("k_second"),
        "tfidf": {
            "workers": train.get("tfidf_workers"),
            "batch": train.get("tfidf_batch"),
            "fragment_len": train.get("tfidf_fragment_len"),
        },
        "hp_budget": {
            "epochs": gpu.get("hp_epochs"),
            "batch": gpu.get("hp_batch"),
            "max_train_rows": gpu.get("hp_max_train_rows"),
            "max_val_rows": gpu.get("hp_max_val_rows"),
            "gpus": gpu.get("hp_gpus"),
            "max_parallel": gpu.get("hp_maxpar"),
            "feat_workers": gpu.get("hp_feat_workers"),
            "feat_chunk": gpu.get("hp_feat_chunk"),
        },
        "final_model": {
            "epochs": gpu.get("model_epochs"),
            "batch": gpu.get("model_batch"),
            "gpus": gpu.get("model_gpus"),
            "max_parallel": gpu.get("model_maxpar"),
        },
    }


# --------------------------------------------------------------------------- #
# harvest: data VOLUMES actually used
# --------------------------------------------------------------------------- #
def harvest_volumes(cfg: dict) -> dict:
    train = cfg.get("train") or {}
    splits = cfg.get("splits") or []
    classes = cfg.get("classes") or []
    out: dict[str, Any] = {
        "train_ready": {},
        "seqpack": {},
        "feature_cache": {},
        "feature_cache_total_bytes": None,
    }

    # 1) raw inputs actually on disk (size only -- never read multi-TiB FASTA)
    ready = train.get("train_ready")
    if ready:
        for split in splits:
            per_class = {}
            total = 0
            for cls in classes:
                st = stat_of(Path(ready) / split / f"{cls}.fasta")
                if st:
                    per_class[cls] = st["bytes"]
                    total += st["bytes"]
            if per_class:
                out["train_ready"][split] = {"classes": per_class,
                                             "total_bytes": total}

    # 2) the 2-bit pack: the record count AFTER subsampling, plus the exact
    #    per-split spec that produced it (this is the ground truth for "how much
    #    data did this version actually train on").
    pack_root = train.get("seq_pack")
    if pack_root:
        for stage in STAGES:
            for split in splits:
                pdir = Path(pack_root) / stage / split
                meta = read_json(pdir / "meta.json")
                if not meta:
                    continue
                codes = stat_of(pdir / "codes.bin")
                out["seqpack"][f"{stage}/{split}"] = {
                    "records": meta.get("n"),
                    "subsample": meta.get("subsample"),
                    "codes_bytes": codes["bytes"] if codes else None,
                    "built_at": codes["mtime"] if codes else None,
                }

    # 3) the dense feature cache: rows x dim per (stage, k, tag) and the bytes
    #    it cost. This is where the disk actually goes.
    feat_root = train.get("feature_cache")
    if feat_root:
        grand = 0
        seen_any = False
        for stage, ks in (("first", train.get("k_first") or []),
                          ("second", train.get("k_second") or [])):
            for k in ks:
                kdir = Path(feat_root) / stage / f"k{int(k)}"
                meta = read_json(kdir / "meta.json") or {}
                for tag in FEATURE_TAGS:
                    entry = meta.get(tag)
                    if not entry:
                        continue
                    xst = stat_of(kdir / f"{tag}_X.f32")
                    yst = stat_of(kdir / f"{tag}_y.i64")
                    nbytes = (xst["bytes"] if xst else 0) + (yst["bytes"] if yst else 0)
                    grand += nbytes
                    seen_any = True
                    node = out["feature_cache"].setdefault(stage, {})
                    node.setdefault(f"k{int(k)}", {})[tag] = {
                        "rows": entry.get("n"),
                        "dim": entry.get("dim"),
                        "bytes": nbytes or None,
                        "built_at": xst["mtime"] if xst else None,
                    }
        if seen_any:
            out["feature_cache_total_bytes"] = grand
    return out


# --------------------------------------------------------------------------- #
# harvest: TIMING
# --------------------------------------------------------------------------- #
def harvest_timing(cfg: dict) -> dict:
    train = cfg.get("train") or {}
    log_dir = Path(train.get("log_dir") or ".")
    out: dict[str, Any] = {
        "steps": [],
        "tfidf": None,
        "hp_search": {},
        "stage_manifests": {},
        "totals": {},
    }

    # per-step WALL time, recorded by ModelBackend._run_module. Append-only, so
    # a --resume run contributes extra lines; we keep them all and also sum the
    # successful ones.
    timings = log_dir / "step_timings.jsonl"
    if timings.is_file():
        try:
            lines = timings.read_text().splitlines()
        except OSError:
            lines = []
        for line in lines:
            row = read_json_line(line)
            if row:
                out["steps"].append(row)

    # TF-IDF records its own duration in its manifest
    tfidf_dir = train.get("tfidf_dir")
    if tfidf_dir:
        man = read_json(Path(tfidf_dir) / "tfidf_manifest.json")
        if man:
            out["tfidf"] = {
                "status": man.get("status"),
                "signature": man.get("signature"),
                "seconds": man.get("seconds"),
                "workers": man.get("workers"),
                "config": man.get("config"),
            }

    # HP search: per-candidate GPU seconds + the winning architecture
    for stage, ks in (("first", train.get("k_first") or []),
                      ("second", train.get("k_second") or [])):
        for k in ks:
            path = log_dir / f"hp_{stage}_k{int(k)}.json"
            data = read_json(path)
            if not data:
                continue
            results = data.get("results") or []
            secs = [float(r.get("seconds") or 0.0) for r in results]
            best = max(results, key=lambda r: r.get("mean_f1", float("-inf"))) if results else None
            st = stat_of(path)
            out["hp_search"][f"{stage}_k{int(k)}"] = {
                "stage": stage,
                "k": int(k),
                "status": data.get("status"),
                "signature": data.get("signature"),
                "labels": data.get("labels"),
                "budget": data.get("budget"),
                "candidates": len(results),
                "gpu_seconds": sum(secs) if secs else None,
                "slowest_candidate_seconds": max(secs) if secs else None,
                "median_candidate_seconds": (sorted(secs)[len(secs) // 2]
                                             if secs else None),
                "finished_at": st["mtime"] if st else None,
                "best": ({
                    "mean_f1": best.get("mean_f1"),
                    "accuracy": best.get("accuracy"),
                    "hid1": best.get("hid1"),
                    "hid2": best.get("hid2"),
                    "learning_rate": best.get("learning_rate"),
                    "dropout": best.get("dropout"),
                    "seconds": best.get("seconds"),
                    "f1": best.get("f1"),
                } if best else None),
            }

    # stage-level provenance from the orchestrator
    work_root = cfg.get("work_root")
    if work_root:
        try:
            entries = sorted(Path(work_root).iterdir())
        except OSError:
            entries = []
        for child in entries:
            man = read_json(child / "manifest.json")
            if not man:
                continue
            out["stage_manifests"][child.name] = {
                "finished_at": man.get("finished_at"),
                "fingerprint": (man.get("fingerprint") or "")[:12] or None,
                "counts": man.get("counts"),
            }

    wall = sum(float(s.get("seconds") or 0.0) for s in out["steps"]
               if s.get("status") == "ok")
    hp_gpu = sum(float(v["gpu_seconds"] or 0.0)
                 for v in out["hp_search"].values())
    out["totals"] = {
        "recorded_wall_seconds": wall or None,
        "hp_gpu_seconds": hp_gpu or None,
        "tfidf_seconds": (out["tfidf"] or {}).get("seconds"),
    }
    return out


def read_json_line(line: str):
    line = line.strip()
    if not line:
        return None
    try:
        row = json.loads(line)
    except Exception:
        return None
    return row if isinstance(row, dict) else None


# --------------------------------------------------------------------------- #
# harvest: published artifacts
# --------------------------------------------------------------------------- #
def harvest_published(nnet_dest: str | None, tfidf_dest: str | None) -> dict:
    out: dict[str, Any] = {}
    for label, dest in (("nnet", nnet_dest), ("tfidf", tfidf_dest)):
        if not dest:
            continue
        root = Path(dest)
        files: list[dict] = []
        total = 0
        if root.is_dir():
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    size = path.stat().st_size
                    total += size
                    files.append({"path": str(path.relative_to(root)),
                                  "bytes": size})
        out[label] = {
            "dest": str(root),
            "exists": root.is_dir(),
            "file_count": len(files),
            "total_bytes": total or None,
            "files": files,
        }
    return out


# --------------------------------------------------------------------------- #
# render: Markdown
# --------------------------------------------------------------------------- #
def table(rows: list[list[str]], header: list[str]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def render_markdown(report: dict) -> str:
    ident = report["identity"]
    sel = report["selection"]
    vol = report["volumes"]
    tim = report["timing"]
    pub = report["published"]
    out: list[str] = []
    add = out.append

    add(f"# Run record: {ident.get('output_tag')}")
    add("")
    add(f"Generated {report['generated_at']} "
        f"(report schema v{report['report_version']}).")
    add("")

    # ---- identity ----
    add("## 1. Run identity")
    add("")
    git = ident.get("git") or {}
    dirty = git.get("dirty")
    rows = [
        ["input_tag", str(ident.get("input_tag"))],
        ["model_tag", str(ident.get("model_tag"))],
        ["output_tag", str(ident.get("output_tag"))],
        ["backend", str(ident.get("backend"))],
        ["base", str(ident.get("base"))],
        ["host", str(ident.get("host"))],
        ["git commit", str(git.get("commit") or "-")],
        ["git branch", str(git.get("branch") or "-")],
        ["working tree", "dirty" if dirty else ("clean" if dirty is False else "-")],
    ]
    out.extend(table(rows, ["field", "value"]))
    add("")

    # ---- selection strategy ----
    add("## 2. Data selection strategy")
    add("")
    add("### 2.1 Subsampling (which records entered training)")
    add("")
    sub = sel.get("subsample") or {}
    if sub:
        add(f"`enabled: {sub.get('enabled')}`, `seed: {sub.get('seed')}` "
            "-- selection is a deterministic per-record content hash, so class "
            "proportions are preserved and a restart reselects the identical "
            "subset.")
        add("")
        rows = []
        for split in ("train", "validation", "test"):
            spec = sub.get(split) or {}
            if not spec:
                continue
            rows.append([split, str(spec.get("rate")),
                         str(spec.get("min_acgt_purity",
                                      spec.get("min_purity", "-"))),
                         str(spec.get("min_len"))])
        if rows:
            out.extend(table(rows, ["split", "rate", "min ACGT purity",
                                    "min length"]))
    else:
        add("_No subsample config recorded (every record kept)._")
    add("")

    add("### 2.2 Deduplication")
    add("")
    ded = sel.get("dedup") or {}
    if ded:
        rows = [[str(key), str(value)] for key, value in sorted(ded.items())
                if not isinstance(value, dict)]
        out.extend(table(rows, ["setting", "value"]))
        varlen = ded.get("varlen")
        if isinstance(varlen, dict):
            add("")
            add(f"Variable-length bin `{ded.get('varlen_bin')}`: "
                + ", ".join(f"`{k}={v}`" for k, v in sorted(varlen.items())))
    else:
        add("_No dedup config recorded._")
    add("")

    add("### 2.3 Fragmentation / MAG policy")
    add("")
    rows = []
    for key, value in sorted((sel.get("chop") or {}).items()):
        rows.append([f"chop.{key}", str(value)])
    for key, value in sorted((sel.get("mag") or {}).items()):
        rows.append([f"mag.{key}", str(value)])
    if rows:
        out.extend(table(rows, ["setting", "value"]))
    else:
        add("_Not recorded._")
    add("")

    add("### 2.4 Model / search budget")
    add("")
    hp = sel.get("hp_budget") or {}
    fin = sel.get("final_model") or {}
    rows = [
        ["k (first stage)", str(sel.get("k_first"))],
        ["k (second stage)", str(sel.get("k_second"))],
        ["HP epochs", str(hp.get("epochs"))],
        ["HP batch", str(hp.get("batch"))],
        ["HP max train rows", count(hp.get("max_train_rows"))],
        ["HP max val rows", count(hp.get("max_val_rows"))],
        ["HP parallel tasks", str(hp.get("max_parallel"))],
        ["final epochs", str(fin.get("epochs"))],
        ["final batch", str(fin.get("batch"))],
    ]
    out.extend(table(rows, ["knob", "value"]))
    add("")
    add("> HP search only RANKS architectures; the winner is retrained on the "
        "full data by the final trainer. A row cap of 0 means no cap.")
    add("")

    # ---- volumes ----
    add("## 3. Data volume actually used")
    add("")
    add("### 3.1 Source corpus on disk")
    add("")
    ready = vol.get("train_ready") or {}
    if ready:
        classes = sorted({c for v in ready.values() for c in v["classes"]})
        header = ["split"] + classes + ["total"]
        rows = []
        for split, node in ready.items():
            row = [split]
            for cls in classes:
                row.append(human_bytes(node["classes"].get(cls)))
            row.append(human_bytes(node.get("total_bytes")))
            rows.append(row)
        out.extend(table(rows, header))
    else:
        add("_Source FASTA not found._")
    add("")

    add("### 3.2 Records kept after selection (2-bit pack)")
    add("")
    pack = vol.get("seqpack") or {}
    if pack:
        rows = []
        for key in sorted(pack):
            node = pack[key]
            spec = node.get("subsample") or {}
            rows.append([key, count(node.get("records")),
                         str(spec.get("rate", "-")),
                         str(spec.get("min_purity", "-")),
                         str(spec.get("min_len", "-")),
                         human_bytes(node.get("codes_bytes"))])
        out.extend(table(rows, ["stage/split", "records", "rate",
                                "min purity", "min len", "packed size"]))
    else:
        add("_No pack metadata found._")
    add("")

    add("### 3.3 Featurized matrices")
    add("")
    feat = vol.get("feature_cache") or {}
    if feat:
        rows = []
        for stage in STAGES:
            for kname in sorted((feat.get(stage) or {}),
                                key=lambda s: int(s[1:])):
                for tag in FEATURE_TAGS:
                    node = (feat[stage][kname] or {}).get(tag)
                    if not node:
                        continue
                    rows.append([stage, kname, tag,
                                 count(node.get("rows")),
                                 str(node.get("dim")),
                                 human_bytes(node.get("bytes"))])
        out.extend(table(rows, ["stage", "k", "split", "rows", "dim", "size"]))
        add("")
        add(f"**Feature cache total: "
            f"{human_bytes(vol.get('feature_cache_total_bytes'))}**")
    else:
        add("_No feature cache metadata found._")
    add("")

    # ---- timing ----
    add("## 4. Time spent")
    add("")
    totals = tim.get("totals") or {}
    rows = [
        ["recorded wall time (all steps)",
         human_dur(totals.get("recorded_wall_seconds"))],
        ["TF-IDF", human_dur(totals.get("tfidf_seconds"))],
        ["HP search (summed GPU time)",
         human_dur(totals.get("hp_gpu_seconds"))],
    ]
    out.extend(table(rows, ["measure", "duration"]))
    add("")
    add("> Wall time counts each step once per invocation, so a `--resume` "
        "restart adds its own rows. Summed GPU time exceeds wall time because "
        "candidates run concurrently across GPUs.")
    add("")

    steps = tim.get("steps") or []
    if steps:
        add("### 4.1 Per-step wall time")
        add("")
        rows = []
        for step in steps:
            rows.append([str(step.get("log") or step.get("module")),
                         str(step.get("status")),
                         str(step.get("started_at")),
                         human_dur(step.get("seconds"))])
        out.extend(table(rows, ["step", "status", "started", "duration"]))
        add("")
    else:
        add("### 4.1 Per-step wall time")
        add("")
        add("_No `step_timings.jsonl` found. Per-step wall time is recorded "
            "from this version onward; runs started before it only report "
            "TF-IDF and HP search durations._")
        add("")

    hps = tim.get("hp_search") or {}
    if hps:
        add("### 4.2 Hyperparameter search")
        add("")
        rows = []
        for key in sorted(hps):
            node = hps[key]
            budget = node.get("budget") or {}
            rows.append([key, str(node.get("status")),
                         count(node.get("candidates")),
                         str(budget.get("epochs", "-")),
                         count(budget.get("hp_max_train_rows")),
                         count(budget.get("hp_max_val_rows")),
                         human_dur(node.get("gpu_seconds")),
                         human_dur(node.get("median_candidate_seconds"))])
        out.extend(table(rows, ["search", "status", "candidates", "epochs",
                                "train rows", "val rows", "GPU time",
                                "median/candidate"]))
        add("")

    # ---- results ----
    add("## 5. Best architecture per search")
    add("")
    if hps:
        rows = []
        for key in sorted(hps):
            best = (hps[key] or {}).get("best")
            if not best:
                continue
            rows.append([key, num(best.get("mean_f1")),
                         num(best.get("accuracy")),
                         str(best.get("hid1")), str(best.get("hid2")),
                         str(best.get("learning_rate")),
                         str(best.get("dropout"))])
        if rows:
            out.extend(table(rows, ["search", "mean F1", "accuracy", "hid1",
                                    "hid2", "lr", "dropout"]))
        else:
            add("_No completed searches._")
    else:
        add("_No search results found._")
    add("")

    # ---- published ----
    add("## 6. Published artifacts")
    add("")
    if pub:
        rows = []
        for label in sorted(pub):
            node = pub[label]
            rows.append([label, str(node.get("dest")),
                         "yes" if node.get("exists") else "no",
                         count(node.get("file_count")),
                         human_bytes(node.get("total_bytes"))])
        out.extend(table(rows, ["which", "destination", "present", "files",
                                "size"]))
    else:
        add("_Nothing published._")
    add("")

    # ---- stage provenance ----
    stages = tim.get("stage_manifests") or {}
    if stages:
        add("## 7. Stage provenance")
        add("")
        rows = [[name, str(node.get("finished_at")),
                 str(node.get("fingerprint"))]
                for name, node in sorted(stages.items())]
        out.extend(table(rows, ["stage", "finished at", "fingerprint"]))
        add("")

    # ---- full config ----
    add("## 8. Resolved configuration snapshot")
    add("")
    add("<details><summary>Full resolved config (click to expand)</summary>")
    add("")
    add("```json")
    add(json.dumps(report.get("config_snapshot") or {}, indent=2,
                   sort_keys=True, default=str))
    add("```")
    add("")
    add("</details>")
    add("")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- #
# render: one CSV row per run (the cross-version comparison surface)
# --------------------------------------------------------------------------- #
CSV_FIELDS = [
    "generated_at", "output_tag", "input_tag", "model_tag", "git_commit",
    "git_dirty", "dedup_enabled", "dedup_within_split", "dedup_min_seq_id",
    "dedup_min_cov", "subsample_enabled", "sub_train_rate",
    "sub_train_min_purity", "sub_train_min_len", "sub_val_rate",
    "k_first", "k_second", "hp_epochs", "hp_max_train_rows",
    "hp_max_val_rows", "hp_batch", "model_epochs",
    "first_train_records", "first_val_records", "second_train_records",
    "second_val_records", "feature_cache_bytes", "tfidf_seconds",
    "hp_gpu_hours", "recorded_wall_hours", "best_f1_first", "best_f1_second",
    "best_by_search", "report_md",
]


def _hours(seconds):
    if not seconds:
        return None
    return round(float(seconds) / 3600.0, 3)


def build_csv_row(report: dict, md_path: str) -> dict:
    ident = report["identity"]
    sel = report["selection"]
    vol = report["volumes"]
    tim = report["timing"]
    sub = sel.get("subsample") or {}
    sub_train = sub.get("train") or {}
    sub_val = sub.get("validation") or {}
    ded = sel.get("dedup") or {}
    hp = sel.get("hp_budget") or {}
    fin = sel.get("final_model") or {}
    pack = vol.get("seqpack") or {}
    totals = tim.get("totals") or {}
    hps = tim.get("hp_search") or {}

    def records(stage, split):
        node = pack.get(f"{stage}/{split}") or {}
        return node.get("records")

    def best_for(stage):
        values = [(v.get("best") or {}).get("mean_f1")
                  for key, v in hps.items() if v.get("stage") == stage]
        values = [v for v in values if v is not None]
        return round(max(values), 6) if values else None

    compact = ";".join(
        f"{key}={num((hps[key].get('best') or {}).get('mean_f1'))}"
        for key in sorted(hps) if hps[key].get("best")
    )
    git = ident.get("git") or {}
    return {
        "generated_at": report["generated_at"],
        "output_tag": ident.get("output_tag"),
        "input_tag": ident.get("input_tag"),
        "model_tag": ident.get("model_tag"),
        "git_commit": (git.get("commit") or "")[:12] or None,
        "git_dirty": git.get("dirty"),
        "dedup_enabled": ded.get("enabled"),
        "dedup_within_split": ded.get("within_split"),
        "dedup_min_seq_id": ded.get("min_seq_id"),
        "dedup_min_cov": ded.get("min_cov"),
        "subsample_enabled": sub.get("enabled"),
        "sub_train_rate": sub_train.get("rate"),
        "sub_train_min_purity": sub_train.get("min_acgt_purity"),
        "sub_train_min_len": sub_train.get("min_len"),
        "sub_val_rate": sub_val.get("rate"),
        "k_first": ",".join(str(k) for k in (sel.get("k_first") or [])),
        "k_second": ",".join(str(k) for k in (sel.get("k_second") or [])),
        "hp_epochs": hp.get("epochs"),
        "hp_max_train_rows": hp.get("max_train_rows"),
        "hp_max_val_rows": hp.get("max_val_rows"),
        "hp_batch": hp.get("batch"),
        "model_epochs": fin.get("epochs"),
        "first_train_records": records("first", "train"),
        "first_val_records": records("first", "validation"),
        "second_train_records": records("second", "train"),
        "second_val_records": records("second", "validation"),
        "feature_cache_bytes": vol.get("feature_cache_total_bytes"),
        "tfidf_seconds": totals.get("tfidf_seconds"),
        "hp_gpu_hours": _hours(totals.get("hp_gpu_seconds")),
        "recorded_wall_hours": _hours(totals.get("recorded_wall_seconds")),
        "best_f1_first": best_for("first"),
        "best_f1_second": best_for("second"),
        "best_by_search": compact or None,
        "report_md": md_path,
    }


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS,
                                extrasaction="ignore")
        if is_new:
            writer.writeheader()
        writer.writerow(row)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def build_report(cfg: dict, *, repo_root: str, nnet_dest=None,
                 tfidf_dest=None) -> dict:
    """Assemble the whole record from a RESOLVED config dict."""
    return {
        "report_version": REPORT_VERSION,
        "generated_at": iso(time.time()),
        "identity": harvest_identity(cfg, repo_root),
        "selection": harvest_selection(cfg),
        "volumes": harvest_volumes(cfg),
        "timing": harvest_timing(cfg),
        "published": harvest_published(nnet_dest, tfidf_dest),
        "config_snapshot": cfg,
    }


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-json", required=True,
                        help="resolved pipeline config, dumped as JSON")
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--index-csv", default=None,
                        help="append one comparison row per run here")
    parser.add_argument("--nnet-dest", default=None)
    parser.add_argument("--tfidf-dest", default=None)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = read_json(args.config_json)
    if not isinstance(cfg, dict):
        print(f"[run-report] cannot read config json: {args.config_json}")
        return 1

    report = build_report(cfg, repo_root=args.repo_root,
                         nnet_dest=args.nnet_dest,
                         tfidf_dest=args.tfidf_dest)
    markdown = render_markdown(report)
    row = build_csv_row(report, args.out_md)

    if args.dry_run:
        print(f"[run-report] would write {args.out_md}")
        print(f"[run-report] would write {args.out_json}")
        if args.index_csv:
            print(f"[run-report] would append a row to {args.index_csv}")
        return 0

    write_atomic(Path(args.out_md), markdown)
    write_atomic(Path(args.out_json),
                 json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    if args.index_csv:
        append_csv(Path(args.index_csv), row)

    totals = report["timing"].get("totals") or {}
    print(f"[run-report] wrote {args.out_md}")
    print(f"[run-report] wrote {args.out_json}")
    if args.index_csv:
        print(f"[run-report] appended comparison row to {args.index_csv}")
    print(f"[run-report] wall={human_dur(totals.get('recorded_wall_seconds'))} "
          f"tfidf={human_dur(totals.get('tfidf_seconds'))} "
          f"hp_gpu={human_dur(totals.get('hp_gpu_seconds'))} "
          f"features={human_bytes(report['volumes'].get('feature_cache_total_bytes'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
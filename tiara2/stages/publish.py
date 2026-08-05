"""Stage: publish --- the closed loop back into the package.

Code lives at ~/ (the repo); runs/data live at /data. The train stage writes
TF-IDF + NNet params under /data. When training is fully done, this stage
snapshots those params INTO the code package at ``tiara/models/...`` so the repo
carries a runnable copy.

Why in-package: later, when tiara2 is packaged, inference points straight at
these in-package params (``tiara/models/nnet-models-<tag>`` /
``tiara/models/tfidf-models-<tag>``), the classic tiara model layout that
``tiara.src.classification`` already knows how to load.

All heavy logic (validation, backup, atomic copy, checksums) lives in the
portable, dependency-free ``scripts/publish_trained_models.py`` so it can run in
any env, mirroring how other stages delegate to scripts/.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from .. import paths
from ..stage import Stage, register


@register("publish")
class PublishStage(Stage):
    param_keys = ("publish",)

    def _dests(self, ctx):
        p = ctx.cfg["publish"]
        # relative code paths -> resolved INSIDE the package (portable)
        nnet_dest = paths.resolve(p["nnet_dest"])
        tfidf_dest = paths.resolve(p["tfidf_dest"])
        return nnet_dest, tfidf_dest

    def inputs(self, ctx):
        p = ctx.cfg["publish"]
        # fingerprint off the trained source dirs so a re-train re-publishes
        return [str(Path(p["nnet_src"])), str(Path(p["tfidf_src"]))]

    def outputs(self, ctx):
        nnet_dest, tfidf_dest = self._dests(ctx)
        return [str(nnet_dest), str(tfidf_dest)]

    # ---- last line of defence ------------------------------------------- #
    @staticmethod
    def _iter_values(node, key):
        """Yield every value stored under ``key`` anywhere in a JSON tree."""
        if isinstance(node, dict):
            for k, v in node.items():
                if k == key:
                    yield v
                else:
                    yield from PublishStage._iter_values(v, key)
        elif isinstance(node, list):
            for item in node:
                yield from PublishStage._iter_values(item, key)

    def _check_quality(self, ctx):
        """HARD gate: never copy a collapsed model into the package.

        This is deliberately separate from the train-stage gate. Publish can be
        run on its own (`tiara2 run --only publish`) against models trained
        days earlier, which is exactly how the collapsed v2.1.2 pack reached
        ``tiara/models/nnet-models-v2.1.2`` and then the benchmark.

        Unlike the report writer below, a failure here is fatal.
        """
        t = ctx.cfg["train"]
        gates = t.get("gates") or {}
        floor = float(gates.get("min_first_mean_f1", 0.0) or 0.0)
        limit = float((ctx.cfg.get("train", {}).get("gpu", {}) or {})
                      .get("sanity_max_pred_share", 0.98) or 0.0)
        problems: list[str] = []

        log_dir = Path(t["log_dir"])
        for k in [int(x) for x in t["k_first"]]:
            path = log_dir / f"hp_first_k{k}.json"
            if not path.is_file():
                problems.append(f"{path.name}: missing (was the HP search run?)")
                continue
            try:
                results = (json.loads(path.read_text()).get("results") or [])
            except (OSError, ValueError) as exc:
                problems.append(f"{path.name}: unreadable ({exc})")
                continue
            healthy = [r for r in results if not r.get("collapsed")]
            if not healthy:
                problems.append(f"{path.name}: every candidate collapsed")
                continue
            best = max(healthy, key=lambda r: float(r.get("mean_f1", 0.0)))
            if floor and float(best.get("mean_f1", 0.0)) < floor:
                problems.append(f"{path.name}: best mean_f1 "
                                f"{float(best.get('mean_f1', 0.0)):.4f} < "
                                f"{floor:.4f}")

        manifest = Path(t["out_models"]) / "training_manifest.json"
        if not manifest.is_file():
            problems.append(f"{manifest} missing: refusing to publish models "
                            f"that carry no training provenance")
        elif limit > 0:
            try:
                payload = json.loads(manifest.read_text())
            except (OSError, ValueError) as exc:
                problems.append(f"{manifest.name}: unreadable ({exc})")
            else:
                bad = [float(v) for v in
                       self._iter_values(payload, "validation_pred_share")
                       if isinstance(v, (int, float)) and float(v) > limit]
                if bad:
                    problems.append(
                        f"{manifest.name}: {len(bad)} model(s) predict a single "
                        f"class for up to {max(bad):.4f} of validation rows "
                        f"(limit {limit:.2f})")

        if problems:
            raise RuntimeError(
                "publish quality gate failed -- NOT copying models into the "
                "package:\n  - " + "\n  - ".join(problems)
                + "\nThis gate exists because v2.1.2 published a collapsed "
                  "pack that answered `eukarya` with p=1.000000 for every "
                  "read. Fix the training run, or relax train.gates on "
                  "purpose.")

    def run(self, ctx):
        p = ctx.cfg["publish"]
        if not p.get("enabled", True):
            ctx.log.info("publish disabled (publish.enabled=false)")
            return {"counts": {}}
        t = ctx.cfg["train"]
        if not ctx.dry_run:
            self._check_quality(ctx)
        nnet_dest, tfidf_dest = self._dests(ctx)
        script = paths.repo_root() / "scripts" / "publish_trained_models.py"
        cmd = [
            sys.executable, str(script),
            "--which", p.get("which", "both"),
            "--nnet-src", str(Path(p["nnet_src"])),
            "--nnet-dest", str(nnet_dest),
            "--tfidf-src", str(Path(p["tfidf_src"])),
            "--tfidf-dest", str(tfidf_dest),
            "--first-count", str(len(t["k_first"])),
            "--second-count", str(len(t["k_second"])),
            "--k-first", *[str(k) for k in t["k_first"]],
            "--k-second", *[str(k) for k in t["k_second"]],
        ]
        if not p.get("backup", True):
            cmd.append("--no-backup")
        if ctx.dry_run:
            cmd.append("--dry-run")
        ctx.log.info("publish: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)
        counts = {"nnet_dest": str(nnet_dest), "tfidf_dest": str(tfidf_dest)}
        counts.update(self._write_report(ctx, nnet_dest, tfidf_dest))
        return {"counts": counts}

    # ---- run record document (final step) -------------------------------- #
    def _write_report(self, ctx, nnet_dest, tfidf_dest) -> dict:
        """Harvest this run's provenance into a versioned record document.

        Consolidates what is otherwise scattered across the run -- the data
        SELECTION strategy (dedup / subsample / chop), the data VOLUME actually
        used (pack records, featurized rows x dims, bytes on disk), the TIME
        each step took, and the winning architectures -- into Markdown (read),
        JSON (diff) and an appended CSV row (compare versions side by side).

        SOFT BY DESIGN, per the requirement: this documents a run that has
        ALREADY succeeded, so it must never fail the pipeline. Missing pieces --
        most importantly training TIME, which is only recorded from the version
        that introduced step_timings.jsonl onward -- are rendered as "-" by the
        harvester, and any unexpected error here degrades to a warning.
        """
        rep = ctx.cfg["publish"].get("report") or {}
        if not rep.get("enabled", True):
            ctx.log.info("run report disabled (publish.report.enabled=false)")
            return {}
        try:
            root = str(ctx.cfg.get("results_root") or ctx.cfg.get("base") or ".")
            tag = ctx.cfg.get("output_tag") or "run"
            out_md = paths.resolve(rep.get("out_md")
                                   or f"{root}/run_report_{tag}.md")
            out_json = paths.resolve(rep.get("out_json")
                                     or f"{root}/run_report_{tag}.json")
            # Hand the RESOLVED config over as JSON so the harvester script
            # stays dependency-free (no PyYAML needed in any env). The dump
            # doubles as provenance: it is the exact config this run used.
            cfg_json = ctx.work_dir / "resolved_config.json"
            cfg_json.parent.mkdir(parents=True, exist_ok=True)
            cfg_json.write_text(json.dumps(ctx.cfg, indent=2, sort_keys=True,
                                           default=str))
            script = paths.repo_root() / "scripts" / "build_run_report.py"
            cmd = [
                sys.executable, str(script),
                "--config-json", str(cfg_json),
                "--out-md", str(out_md),
                "--out-json", str(out_json),
                "--repo-root", str(paths.repo_root()),
                "--nnet-dest", str(nnet_dest),
                "--tfidf-dest", str(tfidf_dest),
            ]
            index_csv = rep.get("index_csv")
            if index_csv:
                cmd += ["--index-csv", str(paths.resolve(index_csv))]
            if ctx.dry_run:
                cmd.append("--dry-run")
            ctx.log.info("run report: %s", " ".join(cmd))
            subprocess.run(cmd, check=True)
            return {"report_md": str(out_md), "report_json": str(out_json)}
        except Exception as exc:  # never fail an already-successful publish
            ctx.log.warning("run report skipped (%s: %s)",
                            type(exc).__name__, exc)
            return {}

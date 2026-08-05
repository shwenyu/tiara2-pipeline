"""Stage: infer --- the pipeline's last step, immediately after publish.

WHY THIS IS A STAGE AND NOT A SCRIPT
------------------------------------
Publish copies the trained params INTO the package. Until now the run stopped
there, and whoever wanted predictions had to leave the pipeline, hand-assemble
the model set, and remember the thresholds. That hand-off is where v2.1.2's
collapsed pack slipped through to a benchmark.

As a stage, inference inherits the same contract as every other step:

  * same invocation      -- ``tiara2 run --only infer`` / ``--from publish``
  * same config surface  -- one ``infer:`` block, templated off the same tags
  * same resume semantics -- fingerprint over the published pack + the inputs,
                            so re-publishing a new pack re-runs inference and
                            an unchanged pack is skipped
  * same dry-run         -- prints the resolved model set and the planned
                            outputs without loading torch
  * same manifest        -- what ran, with what, producing what

It is DISABLED BY DEFAULT (``infer.enabled: false``). A training run should not
silently start classifying hundreds of GiB just because publish succeeded; you
turn it on once the pack has been checked, which is exactly the workflow asked
for -- "train -> publish -> confirm -> infer".

INPUT / OUTPUT INTERFACES (deliberately left open)
--------------------------------------------------
``infer.inputs`` accepts any mix of explicit fasta paths, directories and glob
patterns, so wiring in a new benchmark panel is a config edit, never a code
edit. Outputs land in ``infer.out_dir`` (templated off results_root/output_tag)
as ``<name>.tsv`` + ``log_<name>.txt``, plus optional per-class fasta.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

from .. import paths
from ..stage import Stage, register

FASTA_SUFFIXES = (".fa", ".fna", ".fasta", ".fa.gz", ".fna.gz", ".fasta.gz")


@register("infer")
class InferStage(Stage):
    param_keys = ("infer",)

    # ---- config helpers -------------------------------------------------- #
    @staticmethod
    def _cfg(ctx) -> dict:
        return ctx.cfg.get("infer", {}) or {}

    def _out_dir(self, ctx) -> Path:
        cfg = self._cfg(ctx)
        root = cfg.get("out_dir")
        if not root:
            base = ctx.cfg.get("results_root") or ctx.cfg.get("base") or "."
            tag = ctx.cfg.get("output_tag") or "run"
            root = f"{base}/predictions_{tag}"
        return Path(paths.resolve(str(root)))

    def _resolve_inputs(self, ctx) -> list[str]:
        """Expand paths / dirs / globs into a sorted, de-duplicated fasta list.

        Kept permissive on purpose: a benchmark panel is normally a directory,
        an ad-hoc check is a single file, and a sweep is a glob. All three go
        through the same key so nobody has to touch code to add a panel.
        """
        out: list[str] = []
        seen: set[str] = set()
        for raw in self._cfg(ctx).get("inputs") or []:
            entry = str(paths.resolve(str(raw)))
            candidates: list[str]
            path = Path(entry)
            if path.is_dir():
                candidates = [
                    str(child) for child in sorted(path.iterdir())
                    if child.is_file() and child.name.endswith(FASTA_SUFFIXES)
                ]
            elif any(ch in entry for ch in "*?["):
                candidates = sorted(glob.glob(entry))
            else:
                candidates = [entry]
            for candidate in candidates:
                if candidate not in seen:
                    seen.add(candidate)
                    out.append(candidate)
        return out

    def _plan(self, ctx):
        """Resolve the model set. Imports stay local so `list`/`status` are cheap."""
        from .. import inference as _inference

        cfg = self._cfg(ctx)
        manifest = cfg.get("manifest")
        if manifest:
            manifest_path = Path(paths.resolve(str(manifest)))
            if manifest_path.is_file():
                return _inference.InferencePlan.from_json(manifest_path.read_text())

        cutoffs = {}
        for stage in _inference.STAGES:
            value = (cfg.get("prob_cutoff") or {}).get(stage)
            if value is not None:
                cutoffs[stage] = float(value)
        return _inference.plan_from_config(
            ctx.cfg,
            k_first=cfg.get("k_first"),
            k_second=cfg.get("k_second"),
            cutoffs=cutoffs,
            min_len=int(cfg.get("min_len", 3000) or 3000),
            threads=int(cfg.get("threads")
                        or (ctx.cfg.get("resources", {}) or {}).get("threads", 1)),
            batch_records=int(cfg.get("batch_records", 512) or 512),
            device=cfg.get("device"),
        )

    # ---- Stage contract -------------------------------------------------- #
    def inputs(self, ctx):
        """Fingerprint over BOTH the fasta inputs and the published pack.

        Including the pack is the point: re-publishing new params must
        invalidate old predictions instead of silently resuming past them.
        """
        found = list(self._resolve_inputs(ctx))
        publish = ctx.cfg.get("publish", {}) or {}
        for key in ("nnet_dest", "tfidf_dest"):
            if publish.get(key):
                found.append(str(paths.resolve(publish[key])))
        return found

    def outputs(self, ctx):
        out_dir = self._out_dir(ctx)
        return [str(out_dir / (Path(f).name.split(".")[0] + ".tsv"))
                for f in self._resolve_inputs(ctx)] or [str(out_dir)]

    def run(self, ctx):
        from .. import inference as _inference

        cfg = self._cfg(ctx)
        if not cfg.get("enabled", False):
            ctx.log.info("infer disabled (infer.enabled=false); publish is the "
                         "last step of this run")
            return {"counts": {"status": "disabled"}}

        fastas = self._resolve_inputs(ctx)
        if not fastas:
            raise RuntimeError(
                "infer.enabled is true but infer.inputs resolved to no fasta "
                "file. Set infer.inputs to a file, a directory or a glob.")

        plan = self._plan(ctx)
        for name in _inference.STAGES:
            spec = plan.stages[name]
            ctx.log.info("model %s: k=%d cutoff=%.6f weights=%s tfidf=%s",
                         name, spec.k, spec.prob_cutoff,
                         Path(spec.weights).name, Path(spec.tfidf).name)

        out_dir = self._out_dir(ctx)
        emit = cfg.get("emit_manifest", True)
        probabilities = bool(cfg.get("probabilities", True))
        to_fasta = cfg.get("to_fasta") or []
        gzip_output = bool(cfg.get("gzip", False))

        if ctx.dry_run:
            ctx.log.info("DRY-RUN would classify %d file(s) into %s",
                         len(fastas), out_dir)
            return {"counts": {"inputs": len(fastas), "out_dir": str(out_dir)}}

        out_dir.mkdir(parents=True, exist_ok=True)
        if emit:
            manifest_path = out_dir / f"inference_manifest_{plan.model_tag}.json"
            manifest_path.write_text(plan.to_json())
            ctx.log.info("inference manifest: %s", manifest_path)

        totals = {"inputs": len(fastas), "records": 0, "out_dir": str(out_dir)}
        per_file = []
        for fasta in fastas:
            stem = Path(fasta).name
            for suffix in (".gz",) + FASTA_SUFFIXES:
                if stem.endswith(suffix):
                    stem = stem[: -len(suffix)]
            out_path = out_dir / f"{stem}.tsv"
            log_path = out_dir / f"log_{stem}.txt"
            ctx.log.info("classify %s -> %s", fasta, out_path)
            counts = _inference.run(
                plan, fasta, str(out_path),
                probabilities=probabilities,
                verbose=bool(cfg.get("verbose", False)),
                to_fasta=to_fasta,
                gzip_output=gzip_output,
                log_path=str(log_path),
            )
            totals["records"] += int(counts.get("records", 0))
            per_file.append(counts)

        summary = out_dir / "inference_summary.json"
        summary.write_text(json.dumps(
            {"model_tag": plan.model_tag, "totals": totals, "files": per_file},
            indent=2, sort_keys=True))
        ctx.log.info("inference summary: %s", summary)
        totals["summary"] = str(summary)
        return {"counts": totals}

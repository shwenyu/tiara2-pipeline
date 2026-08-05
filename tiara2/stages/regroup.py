"""Stage: rebuild per-species/per-split FASTA from surviving IDs.

Sequences are written exactly once here, from the immutable source, using the
removed-ID sets produced by dedup. This is the only stage that materializes
full FASTA output.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from ..stage import Stage, register


@register("regroup")
class RegroupStage(Stage):
    param_keys = ("splits", "classes")

    def inputs(self, ctx):
        return [str(Path(ctx.cfg["work_root"]) / "dedup" / ".dedup.done")]

    def outputs(self, ctx):
        outs = [str(Path(ctx.cfg.get("results_root", "")) / "regroup_summary.json")]
        corpus_ready = ctx.cfg.get("corpus_ready")
        if corpus_ready:
            outs.append(str(Path(corpus_ready)))
        return outs

    def run(self, ctx):
        scripts = Path(__file__).resolve().parents[2] / "scripts"
        dedup_root = Path(ctx.cfg["work_root"]) / "chop_bin"  # shards+removed live here
        removed = [str(p) for p in dedup_root.rglob("removed_*.json")]
        results = Path(ctx.cfg["results_root"])
        # The deduped FASTA is the CORPUS layer's product, so it is keyed on
        # corpus_tag and lives outside the per-version results dir. Writing it
        # into results_root (the old behaviour) meant training, which reads
        # train_ready, never saw it. Falls back to results_root only for
        # configs predating corpus_ready.
        corpus_ready = Path(ctx.cfg.get("corpus_ready") or results)
        corpus_ready.mkdir(parents=True, exist_ok=True)
        results.mkdir(parents=True, exist_ok=True)
        ctx.log.info("regroup: deduped corpus -> %s", corpus_ready)

        # FAST PATH: dedup removed nothing (or was skipped via
        # dedup.enabled=false). Streaming every FASTA through the filter would
        # copy the entire multi-TB corpus byte-for-byte to produce an identical
        # result, so mirror it with symlinks instead: instant, zero extra disk,
        # and training still reads exactly one path (train_ready=corpus_ready).
        if not removed:
            n = 0
            for split in ctx.cfg["splits"]:
                (corpus_ready / split).mkdir(parents=True, exist_ok=True)
                for cls in ctx.cfg["classes"]:
                    src = Path(ctx.cfg["source_ready"]) / split / f"{cls}.fasta"
                    dst = corpus_ready / split / f"{cls}.fasta"
                    if not src.exists():
                        continue
                    if dst.is_symlink() or dst.exists():
                        dst.unlink()
                    os.symlink(src, dst)
                    n += 1
            ctx.log.info(
                "regroup: no removed-id files -> mirrored %d source FASTA as "
                "symlinks (no copy). Run dedup (dedup.enabled=true) to "
                "actually filter cross-split leakage.", n)
            (results / "regroup_summary.json").write_text(json.dumps(
                {"mode": "mirror", "linked": n, "removed_files": 0}, indent=2))
            return {"counts": {"linked": n, "removed_files": 0}}

        cmd = [
            sys.executable, str(scripts / "regroup_by_metadata.py"),
            "--source-root", ctx.cfg["source_ready"],
            "--out", str(corpus_ready),
            "--classes", *ctx.cfg["classes"],
            "--splits", *ctx.cfg["splits"],
            "--summary", str(results / "regroup_summary.json"),
        ]
        if removed:
            cmd += ["--removed-json", *removed]
        ctx.log.info("regroup: %d removed-id files", len(removed))
        subprocess.run(cmd, check=True)
        return {"counts": {"removed_files": len(removed)}}

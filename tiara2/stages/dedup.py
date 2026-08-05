"""Stage: per-bin cross-split dedup (wraps run_dedup_bins.sh).

Fixed-length bins use symmetric double-95 (cov-mode 0); the single VAR bin uses
cov-mode 5 (short-seq coverage). Only cross-split leakage is removed; same-split
duplicates are preserved. Per-bin .done markers make it resumable.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from ..stage import Stage, register


@register("dedup")
class DedupStage(Stage):
    param_keys = ("dedup", "resources", "split_priority")

    def inputs(self, ctx):
        m = ctx.cfg.get("work_root", "")
        return [str(Path(m) / "chop_bin" / "binned" / "bin_manifest.json")]

    def outputs(self, ctx):
        return [str(ctx.work_dir / ".dedup.done")]

    def _marker(self, ctx, text: str):
        marker = ctx.work_dir / ".dedup.done"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(text)
        return marker

    def run(self, ctx):
        scripts = Path(__file__).resolve().parents[2] / "scripts"
        d = ctx.cfg["dedup"]
        # OPTIONAL STAGE. mmseqs is the single most expensive step in the whole
        # pipeline (tens of hours) and cannot use the GPUs, so it is opt-in.
        # The marker is still written so regroup's dependency is satisfied and
        # the pipeline stays runnable end-to-end; regroup notices that no
        # removed-id files exist and mirrors the corpus instead of filtering it.
        if not d.get("enabled", True):
            ctx.log.info(
                "dedup SKIPPED (dedup.enabled=false). No cross-split leakage "
                "removal will be performed; regroup will mirror the source "
                "corpus as-is. Enable with --set dedup.enabled=true.")
            self._marker(ctx, "skipped\n")
            return {"counts": {"skipped": 1}}
        r = ctx.cfg["resources"]
        # The chop_bin stage produced shards under its own work dir.
        work = Path(ctx.cfg["work_root"]) / "chop_bin"
        cmd = [
            "bash", str(scripts / "run_dedup_bins.sh"),
            str(work), ctx.cfg.get("mmseqs_bin", "mmseqs"),
            str(d["min_seq_id"]), str(d["min_cov"]), str(d.get("sensitivity", 2.0)),
            str(d.get("max_seqs", 100)), str(d.get("max_accept", 1)),
            str(r["threads"]), str(r.get("split_memory_limit", "100G")),
        ]
        ctx.log.info("dedup: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)
        self._marker(ctx, "ok\n")
        return {"counts": {}}

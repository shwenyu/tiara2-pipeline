"""Stage: chop + length-bin in a SINGLE pass.

Per the design decision, binning is folded into chopping: as each fragment is
cut it is written directly to its (partition, length_bin, split) shard and a row
is appended to the metadata table. This removes the extra full-corpus rewrite
that a separate binning pass would cost.

The heavy chopping logic still lives in the user's existing
``03_04_prepare_dataset_v2_evaluate.py`` (presets T0-T7, ProcessPool, resume).
This stage is the thin integration point: it maps the unified config into the
binned writer and parallelizes per-accession with the shared CPU pool.

Fragments whose length is not one of ``discrete_lengths`` go to a single ``VAR``
bin (see tiara2/../scripts/common.py::LengthBinner), deduped later with
cov-mode 5 (short-seq coverage).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from common import LengthBinner  # noqa: E402

from ..resources import cpu_map  # noqa: E402
from ..stage import Stage, register  # noqa: E402


def binner_from_cfg(cfg: dict) -> LengthBinner:
    d = cfg["dedup"]
    return LengthBinner(
        mode=d.get("length_mode", "discrete"),
        min_cov=d.get("min_cov", 0.95),
        discrete_lengths=tuple(d.get("discrete_lengths", (1000, 2000, 3000, 5000, 10000))),
        tolerance=d.get("tolerance", 0),
        varlen_single=True,
        varlen_bin=d.get("varlen_bin", "VAR"),
    )


@register("chop_bin")
class ChopBinStage(Stage):
    param_keys = ("chop", "dedup", "splits", "classes")

    def inputs(self, ctx):
        src = ctx.cfg.get("source_ready", "")
        found = []
        for split in ctx.cfg.get("splits", []):
            for cls in ctx.cfg.get("classes", []):
                p = os.path.join(src, split, f"{cls}.fasta")
                if os.path.exists(p):
                    found.append(p)
        return found

    def outputs(self, ctx):
        return [str(ctx.work_dir / "binned" / "bin_manifest.json")]

    def run(self, ctx):
        """Delegates to the tested bin_by_length.py writer.

        The standalone script already writes (partition, bin, split) shards +
        metadata TSV + bin_manifest.json in one streaming pass and is unit
        tested. Here we just wire the unified config to it.
        """
        import subprocess
        scripts = Path(__file__).resolve().parents[2] / "scripts"
        d = ctx.cfg["dedup"]
        out = ctx.work_dir / "binned"
        cmd = [
            sys.executable, str(scripts / "bin_by_length.py"),
            "--source-root", ctx.cfg["source_ready"],
            "--out", str(out),
            "--classes", *ctx.cfg["classes"],
            "--splits", *ctx.cfg["splits"],
            "--mode", d.get("length_mode", "discrete"),
            "--min-cov", str(d.get("min_cov", 0.95)),
            "--discrete-lengths", *[str(x) for x in d.get("discrete_lengths", [])],
            "--tolerance", str(d.get("tolerance", 0)),
            "--partition-by", d.get("partition_by", "class"),
        ]
        ctx.log.info("chop_bin: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)
        return {"counts": {"shards": len(list(out.rglob("*.fasta")))}}

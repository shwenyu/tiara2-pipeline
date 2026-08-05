"""Stage: training --- orchestrates the full 05_train flow via a ModelBackend.

Reproduces the proven pipeline
    canonical -> flat(train+validation) -> optimized TF-IDF
    -> dynamic-GPU HP search (first/second x k) -> dynamic-GPU final NNet
but driven by ONE config, using portable (relative) code paths, and behind the
swappable ModelBackend contract so the tiara algorithm can be optimized later
without touching this stage.

Data paths keep pointing at the user's existing /data/shouhanyu layout, so this
plugs straight into the current workflow and old data.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

from ..model_backend import get_backend
from ..stage import Stage, register

CLASSES = ("archaea", "bacteria", "eukarya", "mitochondria", "plastids")
# canonical class name -> flat file stem expected by train_models_gpu
FLAT_MAP = {
    "archaea": "archaea_fr",
    "bacteria": "bacteria_fr",
    "eukarya": "eukarya_fr",
    "mitochondria": "mitochondria_fr",
    "plastids": "plast_fr",
}


@register("train")
class TrainStage(Stage):
    param_keys = ("train",)

    def inputs(self, ctx):
        t = ctx.cfg["train"]
        return [str(Path(t["train_ready"]) / s / f"{c}.fasta")
                for s in ("train", "validation") for c in CLASSES]

    # ---- anti-collapse gate ---------------------------------------------- #
    @staticmethod
    def _check_hp_gates(cfg, ctx, stage: str, ks) -> None:
        """Refuse to spend days on final models when the HP search says no.

        Runs BETWEEN the search and the final fit. v2.1.2 had nothing here: the
        winning stage-1 candidates were constant predictors (mean_f1 0.43-0.50)
        and the pipeline trained, saved and published them anyway.
        """
        t = cfg["train"]
        gates = t.get("gates") or {}
        floor = float(gates.get("min_first_mean_f1", 0.0) or 0.0)
        log_dir = Path(t["log_dir"])
        problems: list[str] = []
        for k in ks:
            path = log_dir / f"hp_{stage}_k{k}.json"
            if not path.is_file():
                problems.append(f"{path.name}: missing")
                continue
            try:
                payload = json.loads(path.read_text())
            except (OSError, ValueError) as exc:
                problems.append(f"{path.name}: unreadable ({exc})")
                continue
            results = payload.get("results") or []
            healthy = [r for r in results if not r.get("collapsed")]
            if not healthy:
                problems.append(
                    f"{path.name}: all {len(results)} candidates collapsed to "
                    f"a single class")
                continue
            best = max(healthy, key=lambda r: float(r.get("mean_f1", 0.0)))
            score = float(best.get("mean_f1", 0.0))
            ctx.log.info("hp gate %s k=%s: best mean_f1=%.4f pred_share=%s "
                         "(%d/%d collapsed)", stage, k, score,
                         best.get("pred_share", "?"),
                         len(results) - len(healthy), len(results))
            if floor and score < floor:
                problems.append(
                    f"{path.name}: best mean_f1 {score:.4f} < required "
                    f"{floor:.4f}")
        if problems:
            raise RuntimeError(
                "HP quality gate failed before final training:\n  - "
                + "\n  - ".join(problems)
                + "\nThis is the v2.1.2 collapse signature. Check that the "
                  "bp-balance stage balanced BOTH train and validation, then "
                  "re-run the HP search. Override via train.gates in the "
                  "config only if you know why.")

    def outputs(self, ctx):
        return [str(Path(ctx.cfg["train"]["out_models"]))]

    @staticmethod
    def _needs_flat(cfg) -> bool:
        """Is the concatenated flat corpus actually read by anything?

        It is not, as soon as ``feature_cache`` is configured: train_tfidf,
        build_features and hp_search all read ``train_ready``, and
        train_models_gpu is invoked with ``--feature-cache``, which makes it
        skip ``load_stage_sequences`` entirely. Building it anyway costs a full
        multi-TB copy of train+validation on every round, written once and read
        never. Kept only for the legacy no-cache path.
        """
        return not bool(cfg["train"].get("feature_cache"))

    # --- Stage B from 05_train: build flat train+validation data ---
    def _build_flat(self, cfg, ctx):
        t = cfg["train"]
        flat = Path(t["flat_data"])
        train_ready = Path(t["train_ready"])
        wanted = [flat / f"{stem}.fasta" for stem in FLAT_MAP.values()]
        if all(p.exists() and p.stat().st_size > 0 for p in wanted):
            ctx.log.info("flat data already complete: %s", flat)
            return
        if ctx.dry_run:
            ctx.log.info("[dry-run] would build flat data at %s", flat)
            return
        tmp = flat.with_suffix(f".tmp.{os.getpid()}.{int(time.time())}")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        for cls, stem in FLAT_MAP.items():
            dst = tmp / f"{stem}.fasta"
            with open(dst, "wb") as out:
                for split in ("train", "validation"):
                    src = train_ready / split / f"{cls}.fasta"
                    with open(src, "rb") as fh:
                        shutil.copyfileobj(fh, out, length=16 * 1024 * 1024)
            if dst.stat().st_size == 0:
                raise RuntimeError(f"empty flat file: {dst}")
        if flat.exists():
            shutil.rmtree(flat)
        tmp.rename(flat)
        ctx.log.info("flat data published: %s", flat)

    def run(self, ctx):
        cfg = ctx.cfg
        t = cfg["train"]
        if not t.get("enabled", True):
            ctx.log.info("train disabled (train.enabled=false)")
            return {"counts": {}}
        backend = get_backend(t.get("backend", "tiara"))
        ctx.log.info("training backend = %s", backend.backend_name)

        # B) flat data -- only when something still reads it (see _needs_flat).
        if self._needs_flat(cfg):
            self._build_flat(cfg, ctx)
        else:
            ctx.log.info("flat data SKIPPED: feature_cache is configured, so "
                         "no step reads flat_data (saves a full train+"
                         "validation copy). Delete any stale copy with "
                         "`tiara2 gc --apply`.")
        # 0) TF-IDF
        backend.train_tfidf(cfg, ctx)
        # 1/2) HP search per stage x k. Build the shared feature cache ONCE per
        # stage first (a single parallel pass over the data covering all k), so
        # the per-k searches reuse it instead of re-reading the huge inputs.
        for stage, ks in (("first", t["k_first"]), ("second", t["k_second"])):
            kk = [int(k) for k in ks]
            backend.build_features(cfg, ctx, stage=stage, ks=kk)
            for k in kk:
                backend.hp_search(cfg, ctx, stage=stage, k=k)
            # Gate the stage-1 search only: stage 2 is a much easier 3-class
            # problem and has its own floor of 0 by design.
            if stage == "first" and not ctx.dry_run:
                self._check_hp_gates(cfg, ctx, stage, kk)
        # 3) final NNet
        backend.train_models(cfg, ctx)
        return {"counts": {"models_dir": str(t["out_models"])}}

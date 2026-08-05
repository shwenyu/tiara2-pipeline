"""Stage contract + registry.

Every pipeline step subclasses ``Stage`` and implements ``run``. The base class
gives every stage, for free and uniformly:

  * declared ``inputs``/``outputs``/``params`` (from config)
  * a manifest with an input fingerprint (provenance)
  * resume: skip when fingerprint is unchanged (unless --force)
  * dry-run: report what would happen without side effects
  * a scoped logger and a per-stage work dir
  * debug hooks: --debug drops into params dump; DEBUG env keeps tmp dirs

Adding a stage or optimizing one in isolation cannot drift from the pipeline,
because the orchestrator only ever talks to this contract.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import manifest as _manifest
from .logging import get_logger

_REGISTRY: dict[str, type["Stage"]] = {}


def register(name: str) -> Callable[[type["Stage"]], type["Stage"]]:
    def deco(cls: type["Stage"]) -> type["Stage"]:
        cls.name = name
        _REGISTRY[name] = cls
        return cls
    return deco


def registry() -> dict[str, type["Stage"]]:
    return dict(_REGISTRY)


@dataclass
class StageContext:
    """Everything a stage needs; passed to ``run``."""
    cfg: dict
    work_dir: Path
    log: Any
    dry_run: bool = False
    force: bool = False
    debug: bool = False
    extra: dict = field(default_factory=dict)


class Stage:
    """Base stage. Subclasses set ``name`` via @register and implement run()."""
    name: str = "base"
    #: config sub-keys this stage reads (for the params fingerprint)
    param_keys: tuple[str, ...] = ()

    def inputs(self, ctx: StageContext) -> list[str]:
        """Input file paths (used for the resume fingerprint)."""
        return []

    def outputs(self, ctx: StageContext) -> list[str]:
        """Primary output paths (informational / manifest)."""
        return []

    def params(self, ctx: StageContext) -> dict:
        out = {}
        for key in self.param_keys:
            if key in ctx.cfg:
                out[key] = ctx.cfg[key]
        return out

    # ---- override this ----
    def run(self, ctx: StageContext) -> dict:  # pragma: no cover - abstract
        raise NotImplementedError

    # ---- orchestration (do not override) ----
    def execute(self, ctx: StageContext) -> dict:
        log = ctx.log
        params = self.params(ctx)
        inputs = self.inputs(ctx)
        fp = _manifest.fingerprint(inputs, params)
        manifest_path = ctx.work_dir / "manifest.json"
        prev = _manifest.read(manifest_path)

        if ctx.debug:
            log.info("[debug] params=%s", params)
            log.info("[debug] inputs=%d files fingerprint=%s", len(inputs), fp[:12])

        if prev and prev.get("fingerprint") == fp and not ctx.force:
            log.info("skip (unchanged) fingerprint=%s", fp[:12])
            return {"status": "skipped", "manifest": prev}

        if ctx.dry_run:
            log.info("DRY-RUN would run; %d inputs -> %s",
                     len(inputs), ", ".join(self.outputs(ctx)) or "(outputs)")
            return {"status": "dry-run", "fingerprint": fp}

        ctx.work_dir.mkdir(parents=True, exist_ok=True)
        log.info("running ...")
        result = self.run(ctx) or {}
        _manifest.write(
            manifest_path, stage=self.name, fp=fp, params=params,
            inputs=inputs, outputs=self.outputs(ctx),
            counts=result.get("counts", {}), versions=result.get("versions", {}),
        )
        log.info("done: %s", result.get("counts", {}))
        return {"status": "ok", "result": result}


def build_context(cfg: dict, stage_name: str, *, dry_run=False, force=False,
                  debug=False, extra=None) -> StageContext:
    work_root = Path(cfg.get("work_root", os.path.join(cfg.get("base", "."), ".work")))
    return StageContext(
        cfg=cfg,
        work_dir=work_root / stage_name,
        log=get_logger(stage_name, cfg.get("log_dir")),
        dry_run=dry_run, force=force, debug=debug, extra=extra or {},
    )

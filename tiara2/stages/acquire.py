"""Stage: data acquisition -- full-stack wrapper over ncbi_pipeline.py.

The ~2900-line ncbi_pipeline.py is already config-driven, idempotent and
resumable, so it is NOT rewritten here. What this stage adds is the thing the
old wrapper lacked: the acquisition pipeline is no longer an opaque all-or-
nothing shell-out. Its ten steps are exposed individually so each one can be
turned on or off, logged separately, timed, and (for the irreversible network
step) confirmed before it runs.

Why run the steps as separate invocations rather than one ``run --stages a,b``:
  * a failure names the exact step instead of the whole pipeline;
  * each step gets its own log file, so a slow download does not bury the
    metadata diff that preceded it;
  * ``--class`` only applies to the download step in ncbi_pipeline.py's CLI,
    which a single combined ``run`` invocation cannot express;
  * steps can be re-run individually after a partial failure.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .. import inventory, versioning
from ..stage import Stage, register

# ncbi_pipeline.py's RUN_ORDER, in dependency order. Keep in sync with it.
STEP_ORDER = (
    "fetch-metadata",   # download assembly_summary.* from NCBI
    "build-index",      # parse summaries into the unified index
    "apply-qc",         # QC flags / contamination screening
    "select",           # one-isolate-per-species, weighted quality scoring
    "plan-downloads",   # build download_all.tsv manifest
    "download",         # <-- the slow, network-bound, irreversible step
    "verify",           # md5 / composition re-check of what landed
    "export",           # emit per-class FASTA
    "split",            # train / validation / test assignment
    "organize",         # materialise source_ready/<split>/<class>.fasta
)

# Steps that hit the network hard and can run for days. These are what the
# confirmation prompt guards.
NETWORK_STEPS = frozenset({"fetch-metadata", "download", "fetch-organelle"})

# Cheap, read-only steps -- safe to re-run at any time.
READONLY_STEPS = frozenset({"status"})

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off", ""}


def _as_bool(value, default=True):
    """Config values may arrive as real bools or as --set strings."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUTHY:
        return True
    if text in _FALSY:
        return False
    return default


def planned_steps(cfg: dict) -> list[str]:
    """The enabled acquisition steps, always in dependency order.

    Order comes from STEP_ORDER, never from the config, so that toggling steps
    can disable work but can never reorder it into something invalid.
    """
    acq = cfg.get("acquire", {}) or {}
    steps = acq.get("steps") or {}
    if not isinstance(steps, dict):
        steps = {}
    return [s for s in STEP_ORDER if _as_bool(steps.get(s), True)]


def skipped_steps(cfg: dict) -> list[str]:
    enabled = set(planned_steps(cfg))
    return [s for s in STEP_ORDER if s not in enabled]


def resolve_script(cfg: dict) -> Path:
    """Locate ncbi_pipeline.py.

    Since v2.2h the script and its ``config.json`` ship inside the package at
    ``scripts/``, so the default relative name resolves with no configuration.
    A copy left outside the package (repo parent, ``base``, ``data_home``) is
    still honoured, and an absolute ``acquire.script`` always wins -- that is
    how you point at a private working copy instead of the bundled one.

    Raises with the full candidate list rather than letting subprocess fail
    with a bare ENOENT that says nothing about where we looked.
    """
    acq = cfg.get("acquire", {}) or {}
    script = str(acq.get("script", "ncbi_pipeline.py"))
    given = Path(script).expanduser()
    if given.is_absolute():
        if given.exists():
            return given
        raise FileNotFoundError(f"acquire.script not found: {given}")

    candidates = []
    hint = acq.get("script_dir")
    if hint:
        candidates.append(Path(str(hint)).expanduser() / script)
    repo_root = Path(__file__).resolve().parents[2]
    candidates += [
        repo_root / "scripts" / script,
        repo_root / script,
        repo_root.parent / script,
        Path(str(cfg.get("base", "."))) / script,
        Path(str(cfg.get("data_home", "."))) / script,
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"cannot locate acquire.script {script!r}. Tried:\n  {tried}\n"
        "Set an absolute path with --set acquire.script=/path/to/ncbi_pipeline.py")


def resolve_legacy_config(cfg: dict, script: Path) -> Path:
    acq = cfg.get("acquire", {}) or {}
    legacy = str(acq.get("legacy_config", "config.json"))
    given = Path(legacy).expanduser()
    if given.is_absolute():
        return given
    # Default to the json sitting next to the script, which is what
    # ncbi_pipeline.py itself does when --config is omitted.
    return script.parent / legacy


@register("acquire")
class AcquireStage(Stage):
    param_keys = ("acquire",)

    def outputs(self, ctx):
        return [ctx.cfg.get("source_ready", "")]

    def _log_dir(self, ctx) -> Path:
        d = ctx.work_dir / "logs"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _report_existing_data(self, ctx):
        """Shallow inventory of what is already on disk, before fetching more.

        Deliberately uses inventory.corpus_inventory (stat-only, no du, no deep
        walk) so this stays sub-second on a 36 TB spinning array.
        """
        try:
            inv = inventory.corpus_inventory(ctx.cfg)
            for line in inventory.render_corpus(inv).splitlines():
                ctx.log.info("%s", line)
            free = inventory.disk_free(ctx.cfg.get("base", "."))
            if free.get("ok"):
                ctx.log.info("disk free: %s (%s%% used)",
                             inventory.human(free["free"]), free["pct_used"])
        except Exception as exc:  # inventory must never block acquisition
            ctx.log.warning("inventory unavailable: %s", exc)

    def _confirm_network(self, ctx, steps) -> bool:
        """Ask once before the long network steps. Never blocks under nohup."""
        acq = ctx.cfg.get("acquire", {}) or {}
        if not _as_bool(acq.get("confirm"), True):
            return True
        hot = [s for s in steps if s in NETWORK_STEPS]
        if not hot:
            return True
        assume_yes = bool((ctx.extra or {}).get("assume_yes"))
        return versioning.confirm(
            f"acquire will run network steps ({', '.join(hot)}). Continue?",
            default=True, assume_yes=assume_yes)

    def _run_step(self, ctx, step, script, legacy_cfg):
        acq = ctx.cfg.get("acquire", {}) or {}
        cmd = [sys.executable, str(script), step, "--config", str(legacy_cfg)]
        only_class = str(acq.get("only_class", "") or "").strip()
        if step == "download" and only_class:
            # --class is only honoured by the download subcommand.
            cmd += ["--class", only_class]
        log_path = self._log_dir(ctx) / f"acquire_{step.replace('-', '_')}.log"
        ctx.log.info("acquire[%s]: %s", step, " ".join(cmd))
        ctx.log.info("acquire[%s]: log -> %s", step, log_path)
        with open(log_path, "a") as fh:
            fh.write(f"\n=== {step} :: {' '.join(cmd)} ===\n")
            fh.flush()
            proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            raise RuntimeError(
                f"acquire step {step!r} failed (exit {proc.returncode}). "
                f"See {log_path}")

    def run(self, ctx):
        acq = ctx.cfg.get("acquire", {}) or {}
        if not _as_bool(acq.get("enabled"), False):
            ctx.log.info(
                "acquire SKIPPED (acquire.enabled=false); assuming %s is "
                "already populated. Enable with --set acquire.enabled=true",
                ctx.cfg.get("source_ready", "source_ready"))
            return {"counts": {"skipped": 1}}

        steps = planned_steps(ctx.cfg)
        skipped = skipped_steps(ctx.cfg)
        if not steps:
            ctx.log.warning("acquire: every step disabled; nothing to do")
            return {"counts": {"steps": 0}}

        ctx.log.info("acquire plan: %s", " -> ".join(steps))
        if skipped:
            ctx.log.info("acquire skipping: %s", ", ".join(skipped))
        self._report_existing_data(ctx)

        script = resolve_script(ctx.cfg)
        legacy_cfg = resolve_legacy_config(ctx.cfg, script)
        if not legacy_cfg.exists():
            raise FileNotFoundError(
                f"acquire.legacy_config not found: {legacy_cfg}")
        ctx.log.info("acquire: script=%s config=%s", script, legacy_cfg)

        if ctx.dry_run:
            ctx.log.info("acquire: dry-run, no steps executed")
            return {"counts": {"planned": len(steps), "dry_run": 1}}

        if not self._confirm_network(ctx, steps):
            ctx.log.info("acquire: declined at confirmation; nothing fetched.")
            return {"counts": {"declined": 1}}

        done = 0
        for step in steps:
            self._run_step(ctx, step, script, legacy_cfg)
            done += 1
            ctx.log.info("acquire[%s]: done (%d/%d)", step, done, len(steps))

        self._report_existing_data(ctx)
        return {"counts": {"steps": done, "skipped": len(skipped)}}

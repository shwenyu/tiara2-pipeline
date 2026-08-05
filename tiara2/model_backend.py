"""Model / training backend interface --- the reserved extension point.

You said you will keep optimizing Tiara's *algorithm* later. To make that safe
and non-disruptive, the pipeline never calls tiara internals directly. Instead
it talks to a small, stable ``ModelBackend`` contract. The default backend
(:class:`TiaraBackend`) simply drives the vendored ``tiara.training.*`` modules
via their existing CLIs, using portable (relative) code paths.

When you want to try a new algorithm you have two clean options, neither of
which touches the pipeline framework:

  A. Edit the vendored package in ``<repo>/tiara/`` in place. Because it is an
     editable install, changes take effect immediately and the default backend
     keeps working.

  B. Register a NEW backend (e.g. a transformer-based classifier) without
     removing the old one, then switch via config ``train.backend: my_backend``.
     Old and new stay side by side for comparison -- exactly the workflow the
     Tiara-v2 README asks for (label results by model version).

    from tiara2.model_backend import ModelBackend, register_backend

    @register_backend("my_backend")
    class MyBackend(ModelBackend):
        def train_tfidf(self, cfg, ctx): ...
        def hp_search(self, cfg, ctx, *, stage, k): ...
        def train_models(self, cfg, ctx): ...

The contract is deliberately tiny (3 verbs) so it stays stable while the
algorithm underneath changes freely.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

from . import paths

_BACKENDS: dict[str, type["ModelBackend"]] = {}


def register_backend(name: str) -> Callable[[type], type]:
    def deco(cls: type) -> type:
        _BACKENDS[name] = cls
        cls.backend_name = name
        return cls
    return deco


def get_backend(name: str) -> "ModelBackend":
    if name not in _BACKENDS:
        raise KeyError(f"unknown model backend {name!r}; have {sorted(_BACKENDS)}")
    return _BACKENDS[name]()


def available_backends() -> list[str]:
    return sorted(_BACKENDS)


class ModelBackend:
    """Stable 3-verb contract the pipeline depends on."""

    backend_name = "base"

    def train_tfidf(self, cfg: dict, ctx) -> None:
        raise NotImplementedError

    def hp_search(self, cfg: dict, ctx, *, stage: str, k: int) -> Path:
        raise NotImplementedError

    def train_models(self, cfg: dict, ctx) -> None:
        raise NotImplementedError

    def build_features(self, cfg: dict, ctx, *, stage: str, ks: list[int]) -> None:
        """Optional pre-pass: build the shared HP feature cache for all k of a
        stage in ONE pass over the data. Default is a no-op so a custom backend
        that featurizes its own way needs not implement it."""
        return None

    # shared helper: run a python -m module with the vendored package on path.
    #
    # This wrapper adds a WALL-CLOCK recorder around every step. Nothing else in
    # the pipeline knows how long a step actually took: TF-IDF stores only its
    # own `seconds`, HP search stores per-candidate GPU seconds (not wall time),
    # and the feature-cache builder / final NNet trainer stored NOTHING at all.
    # Since every training step goes through here, appending one JSON line per
    # step to <log_dir>/step_timings.jsonl gives the run report a single, honest
    # source of truth for timing. Append-only so it survives --resume restarts,
    # and strictly best-effort: failing to record a timing must never break a
    # multi-hour training run.
    def _run_module(self, module: str, args: list[str], ctx, log_name: str,
                    echo: bool = False) -> None:
        started = time.time()
        status = "ok"
        try:
            self._run_module_inner(module, args, ctx, log_name, echo=echo)
        except BaseException:
            # record the partial duration too -- a step that died after 6 h is
            # exactly the kind of thing the report needs to show.
            status = "failed"
            raise
        finally:
            if not ctx.dry_run:
                self._record_timing(ctx, module, log_name, started,
                                    time.time(), status)

    @staticmethod
    def _record_timing(ctx, module: str, log_name: str, started: float,
                       ended: float, status: str) -> None:
        try:
            log_dir = Path(ctx.cfg["train"]["log_dir"])
            log_dir.mkdir(parents=True, exist_ok=True)
            row = {
                "module": module,
                "log": log_name,
                "status": status,
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z",
                                            time.localtime(started)),
                "ended_at": time.strftime("%Y-%m-%dT%H:%M:%S%z",
                                          time.localtime(ended)),
                "seconds": round(ended - started, 3),
            }
            with open(log_dir / "step_timings.jsonl", "a") as fh:
                fh.write(json.dumps(row) + "\n")
        except Exception:  # pragma: no cover - telemetry must never be fatal
            pass

    def _run_module_inner(self, module: str, args: list[str], ctx,
                          log_name: str, echo: bool = False) -> None:
        env = paths.pythonpath_with_repo()
        cmd = [sys.executable, "-m", module, *[str(a) for a in args]]
        ctx.log.info("%s: %s", self.backend_name, " ".join(cmd))
        if ctx.dry_run:
            return
        log_dir = Path(ctx.cfg["train"]["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / log_name
        if not echo:
            # Quiet step: capture everything to the per-step log file only, so
            # the console stays clean for the high-level stage INFO lines.
            with open(log_path, "ab") as fh:
                subprocess.run(cmd, check=True, env=env, stdout=fh,
                               stderr=subprocess.STDOUT)
            return
        # Live step: stream the child's output line-by-line. On a TTY we render
        # the periodic progress bar (the "read+featurize" lines) IN PLACE with a
        # carriage return, so the bar advances on ONE line instead of scrolling
        # the same numbers over and over; milestone / error lines scroll
        # normally. The per-step log FILE keeps clean full-line history, but the
        # bar is snapshotted only periodically there so it does not bloat.
        #
        # Force unbuffered output in the CHILD. Its stdout is a pipe here, and
        # Python block-buffers pipes (~4-8 KiB), so low-volume output such as
        # skorch's one-line-per-epoch table would sit in the buffer for many
        # minutes and the console would look frozen while training is perfectly
        # fine. Our own prints pass flush=True, but third-party ones do not.
        env = {**env, "PYTHONUNBUFFERED": "1"}
        is_tty = sys.stdout.isatty()
        on_prog = False        # console cursor currently sits on a live bar
        last_len = 0           # width of the last bar (to clear leftovers)
        pending = None         # newest bar line not yet written to the file
        last_file = 0.0        # when we last snapshotted a bar into the file
        file_every = 30.0      # seconds between bar snapshots in the file

        def _is_bar(s: str) -> bool:
            return "read+featurize" in s

        with open(log_path, "ab") as fh:
            def _flush_pending() -> None:
                nonlocal pending
                if pending is not None:
                    fh.write(pending.encode("utf-8", errors="replace"))
                    fh.flush()
                    pending = None

            proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, bufsize=1,
                                    text=True)
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    if _is_bar(line):
                        pending = line
                        now = time.time()
                        if is_tty:
                            # Overwrite the current line IN PLACE. Critically,
                            # CLAMP the bar to the terminal width first: a line
                            # wider than the terminal wraps onto a 2nd physical
                            # row, and then "\r" only returns to the start of the
                            # LAST row -- the wrapped remainder is left behind
                            # and the screen scrolls. That is exactly what starts
                            # happening once the counters grow (big seq counts /
                            # elapsed / ETA make the line longer than the window).
                            # Truncating to (cols - 1) keeps it on ONE row so the
                            # bar advances in place for the whole run.
                            cols = shutil.get_terminal_size((100, 24)).columns
                            width = max(20, cols - 1)
                            text = line.rstrip("\n")
                            if len(text) > width:
                                text = text[:width]
                            out = "\r" + text
                            if len(text) < last_len:
                                out += " " * (last_len - len(text))
                            sys.stdout.write(out)
                            sys.stdout.flush()
                            last_len = len(text)
                            on_prog = True
                        if now - last_file >= file_every:
                            _flush_pending()
                            if not is_tty:
                                # Non-TTY (redirected/nohup): scroll snapshots
                                # at the same throttle so logs do not bloat.
                                sys.stdout.write(line)
                                sys.stdout.flush()
                            last_file = now
                    else:
                        # Milestone / normal line: pin the latest bar into the
                        # file first, then scroll this line on both sinks.
                        _flush_pending()
                        fh.write(line.encode("utf-8", errors="replace"))
                        fh.flush()
                        if on_prog and is_tty:
                            sys.stdout.write("\n")
                            on_prog = False
                            last_len = 0
                        sys.stdout.write(line)
                        sys.stdout.flush()
            finally:
                _flush_pending()
                if on_prog and is_tty:
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                if proc.stdout is not None:
                    proc.stdout.close()
                ret = proc.wait()
        if ret != 0:
            raise subprocess.CalledProcessError(ret, cmd)


@register_backend("tiara")
class TiaraBackend(ModelBackend):
    """Default: drive the vendored tiara.training.* CLIs (05_train logic).

    All code paths are relative/portable via tiara2.paths; only DATA paths come
    from config and keep pointing at the user's existing /data/shouhanyu layout.
    """

    def train_tfidf(self, cfg, ctx):
        t = cfg["train"]
        self._run_module(
            "tiara.training.train_tfidf_optimized",
            [Path(t["train_ready"]) / "train", t["tfidf_dir"],
             "--workers", t["tfidf_workers"],
             "--batch-size", t["tfidf_batch"],
             "--fragment-len", t["tfidf_fragment_len"],
             "--k-first", *[str(int(k)) for k in t["k_first"]],
             "--k-second", *[str(int(k)) for k in t["k_second"]]],
            # Stream TF-IDF document-count heartbeats to the parent train log
            # and terminal, while preserving the dedicated stage log.
            ctx, "10_tfidf.log", echo=True,
        )

    @staticmethod
    def _subsample_args(t: dict) -> list:
        """`--subsample <json>` iff a subsample block is configured.

        Passed as JSON so the leaf CLIs stay stringly-typed; the leaf calls
        seqpack.normalize_subsample which treats a disabled/no-op block as None,
        so an unused block never changes the feature-cache signature.
        """
        sub = t.get("subsample")
        if sub is None:
            return []
        return ["--subsample", json.dumps(sub)]

    @classmethod
    def _pack_args(cls, t: dict) -> list:
        """`--seq-pack <root>` (shared 2-bit pack) plus `--subsample`.

        Enabling seq_pack makes the multi-TB FASTA be read exactly once into a
        compact, resumable pack that is shared across all k.
        """
        out = []
        seq_pack = t.get("seq_pack")
        if seq_pack:
            out += ["--seq-pack", seq_pack]
        out += cls._subsample_args(t)
        return out

    def build_features(self, cfg, ctx, *, stage, ks):
        """One parallel pass over the stage's data building features for ALL k,
        written to a persistent cache reused by the per-k hp_search below."""
        t = cfg["train"]
        g = t["gpu"]
        self._run_module(
            "tiara.training.featurize_cache",
            ["--stage", stage,
             "--ks", ",".join(str(int(x)) for x in ks),
             "--tfidf-dir", t["tfidf_dir"],
             "--feature-cache", t["feature_cache"],
             "--workers", g["hp_feat_workers"],
             "--feat-chunk", g["hp_feat_chunk"],
             *self._pack_args(t),
             t["train_ready"]],
            ctx, f"09_featcache_{stage}.log", echo=True,
        )

    def hp_search(self, cfg, ctx, *, stage, k):
        t = cfg["train"]
        g = t["gpu"]
        result = Path(t["log_dir"]) / f"hp_{stage}_k{k}.json"
        self._run_module(
            "tiara.training.hyperparameter_search_gpu",
            ["--stage", stage, "--k", k, "--tfidf-dir", t["tfidf_dir"],
             "--gpus", g["hp_gpus"], "--max-parallel", g["hp_maxpar"],
             "--cpu-threads-per-task", g["hp_cpu_threads"],
             "--min-free-mib", g["min_free_mib"], "--max-gpu-util", g["max_gpu_util"],
             "--max-tasks-per-gpu", g["hp_max_tasks_per_gpu"],
             "--shared-min-free-mib", g["hp_shared_min_free_mib"],
             "--shared-max-gpu-util", g["hp_shared_max_gpu_util"],
             "--share-launch-delay", g["hp_share_launch_delay"],
             "--poll-seconds", g["poll_seconds"],
             "--batch-size", g["hp_batch"], "--epochs", g["hp_epochs"],
             "--feat-chunk", g["hp_feat_chunk"],
             "--feature-cache", t["feature_cache"],
             "--feat-workers", g["hp_feat_workers"], "--resume",
             "--hp-max-train-rows", g.get("hp_max_train_rows", 0),
             "--hp-max-val-rows", g.get("hp_max_val_rows", 0),
             # v2.1.3 anti-collapse controls. `equal` makes the ranking set
             # class-balanced, `balanced` weights the loss, and max-pred-share
             # demotes any candidate that answers one class for everything.
             "--val-balance", g.get("hp_val_balance", "equal"),
             "--class-weight", g.get("class_weight", "balanced"),
             "--max-pred-share", g.get("max_pred_share", 0.98),
             "--max-val-class-share",
             (t.get("gates", {}) or {}).get("max_val_class_share", 0.75),
             *self._pack_args(t),
             t["train_ready"], result],
            ctx, f"11_hp_{stage}_k{k}.log", echo=True,
        )
        return result

    @staticmethod
    def _model_data_dir(t: dict) -> str:
        """Positional data dir for train_models_gpu.

        That directory is only ever READ on the legacy path: with
        ``--feature-cache`` set (always, since the in-RAM full-corpus build got
        OOM-killed) the trainer builds its matrix from the cache and never calls
        load_stage_sequences. Pointing it at train_ready in that case means the
        multi-TB flat copy no longer has to exist at all -- the stage stopped
        building it, so requiring it here would break the run.
        """
        return str(t["train_ready"] if t.get("feature_cache") else t["flat_data"])

    def train_models(self, cfg, ctx):
        t = cfg["train"]
        g = t["gpu"]
        final_split_args = (["--include-validation"]
                            if t.get("final_include_validation", False)
                            else [])
        self._run_module(
            "tiara.training.train_models_gpu",
            [self._model_data_dir(t), t["out_models"], g["model_cpu_threads"],
             "--hp-dir", t["log_dir"], "--tfidf-dir", t["tfidf_dir"],
             "--k-first", *[str(int(k)) for k in t["k_first"]],
             "--k-second", *[str(int(k)) for k in t["k_second"]],
             "--epochs", g["model_epochs"], "--gpus", g["model_gpus"],
             "--max-parallel", g["model_maxpar"],
             "--max-tasks-per-gpu", g["model_max_tasks_per_gpu"],
             "--min-free-mib", g["model_min_free_mib"],
             "--max-gpu-util", g["model_max_gpu_util"],
             "--shared-min-free-mib", g["model_shared_min_free_mib"],
             "--shared-max-gpu-util", g["model_shared_max_gpu_util"],
             "--share-launch-delay", g["model_share_launch_delay"],
             "--poll-seconds", g["model_poll_seconds"],
             "--batch-size", g["model_batch"],
             # Train from the cached matrices instead of rebuilding a dense
             # full-corpus matrix in RAM (that path got OOM-killed, exit=-9).
             "--feature-cache", t["feature_cache"],
             "--model-max-rows", g.get("model_max_rows", 4000000),
             "--ram-budget-gib", g.get("model_ram_budget_gib", 48),
             "--progress-secs", g.get("model_progress_secs", 60),
             # v2.1.3: weighted loss, a minimum HP score, and a post-fit
             # collapse probe that prevents a constant model being written.
             "--class-weight", g.get("class_weight", "balanced"),
             "--min-mean-f1",
             (t.get("gates", {}) or {}).get("min_first_mean_f1", 0.0),
             "--sanity-max-pred-share",
             g.get("sanity_max_pred_share", 0.98),
             *final_split_args,
             "--resume",
             *self._subsample_args(t)],
            # echo=True: stream subset-build progress, scheduler heartbeats and
            # per-epoch lines to the console instead of only the log file.
            ctx, "20_train_models_gpu.log", echo=True,
        )

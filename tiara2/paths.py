"""Portable path resolution.

The whole point of this module: nothing in the pipeline hardcodes an absolute
code path anymore. Everything the *code* needs is found RELATIVE to this file,
so the package keeps working when moved to another machine or checked out under
a different home directory.

Two distinct roots:

1. repo_root  -- where the CODE lives. Discovered from __file__ (the parent of
   the ``tiara2`` package). This is what used to be hardcoded as
   ``$HOME/tiara-v2``. Now it is wherever this package actually sits, e.g.
   ``~/`` == /home/shouhanyu on the user's box.

2. data_root  -- where the DATA lives. This is genuinely external to the code
   and is intentionally kept configurable, defaulting to the exact locations
   from the user's existing scripts (/data/shouhanyu ...). Data paths are NOT
   derived from __file__ because data does not move with the code.

The defaults below reproduce the directory layout from ncbi_pipeline.py /
05_train so the refactored pipeline plugs straight into the existing data and
workflow without any path edits.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---- code roots (relative, portable) -------------------------------------
# paths.py -> tiara2/ -> <repo_root>
REPO_ROOT = Path(__file__).resolve().parents[1]
TIARA_PKG = REPO_ROOT / "tiara"          # vendored model+training package
SCRIPTS_DIR = REPO_ROOT / "scripts"      # helper scripts (dedup/bin/regroup...)
CONFIG_DIR = REPO_ROOT / "config"
DEFAULT_CONFIG = CONFIG_DIR / "config.yaml"


def repo_root() -> Path:
    return REPO_ROOT


def tiara_pkg() -> Path:
    return TIARA_PKG


def pythonpath_with_repo(env: dict | None = None) -> dict:
    """Return an env dict with REPO_ROOT prepended to PYTHONPATH.

    This lets ``python -m tiara.training.*`` resolve the vendored package even
    when it is not pip-installed, so the package is runnable straight after
    unzip on a fresh machine.
    """
    env = dict(os.environ if env is None else env)
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{prev}" if prev else str(REPO_ROOT)
    return env


# ---- data roots (configurable, default = user's existing layout) ---------
# Mirrors 05_train: BASE=/data/shouhanyu/Tiara2, ENV=/data/shouhanyu/envs/tiara2
DEFAULT_DATA_HOME = "/data/shouhanyu"
DEFAULT_BASE = "/data/shouhanyu/Tiara2"
DEFAULT_ENV = "/data/shouhanyu/envs/tiara2"
DEFAULT_MMSEQS = "/data/shouhanyu/envs/mmseqs2/bin/mmseqs"


def resolve(path_like: str | os.PathLike, *, base: str | os.PathLike | None = None) -> Path:
    """Resolve a possibly-relative path.

    Absolute paths (e.g. the user's /data/shouhanyu/... data) are returned as-is.
    Relative paths are resolved against ``base`` (default: repo_root), so config
    can use short relative code paths that stay portable.
    """
    p = Path(os.path.expanduser(str(path_like)))
    if p.is_absolute():
        return p
    root = Path(base) if base is not None else REPO_ROOT
    return (root / p).resolve()

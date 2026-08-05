"""Uniform logging: console + per-stage rotating file, with a DEBUG switch."""
from __future__ import annotations

import logging
import os
from pathlib import Path

_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def get_logger(name: str, log_dir: str | None = None) -> logging.Logger:
    logger = logging.getLogger(f"tiara2.{name}")
    if logger.handlers:
        return logger
    level = logging.DEBUG if os.environ.get("TIARA2_DEBUG") else logging.INFO
    logger.setLevel(level)
    fmt = logging.Formatter(_FMT)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(log_dir) / f"{name}.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger

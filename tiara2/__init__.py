"""Tiara2 pipeline framework.

A thin, uniform layer that every stage plugs into:

    config (single YAML)  ->  Stage contract  ->  manifest + resume
                                    |
                              resources (CPU pool / GPU scheduler)

Design goals
------------
* One config file (``config/config.yaml``) is the single source of truth,
  replacing the previous mix of config.json + argparse presets + train env vars.
* Every stage implements the same contract (inputs -> outputs + manifest),
  so new stages or per-stage optimizations never drift from the whole pipeline.
* Provenance + resume are built in: each stage writes a manifest with an input
  fingerprint, resolved params, tool versions and counts, and skips work whose
  fingerprint is unchanged.
* Compute efficiency is a first-class concern: shared helpers for CPU process
  pools and a reusable GPU scheduler.
"""
from .version import __version__

__all__ = ["__version__"]

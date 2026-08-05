"""Unified configuration: load, merge, template, validate.

Why this exists
---------------
The legacy pipeline had three config surfaces:
  * ``config.json``        (data acquisition stage)
  * argparse ``PRESETS``   (chop stage)
  * shell ``${ENV:-...}``  (train stage)

That made a single change ripple across three formats. This module makes ONE
YAML file authoritative. Precedence (highest wins):

    CLI --set overrides  >  environment (TIARA2_*)  >  config.yaml  >  DEFAULTS

Path templating: any string value may reference other resolved top-level keys
with ``{name}`` (e.g. ``"{base}/log/{output_tag}"``). Templating runs after the
merge so overrides are reflected.
"""
from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - yaml is in envs/environment.yaml
    yaml = None

# Minimal, explicit defaults. Anything a stage relies on should have a default
# here so a partial config.yaml still runs.
DEFAULTS: dict[str, Any] = {
    "splits": ["train", "validation", "test"],
    "split_priority": ["test", "validation", "train"],
    "resources": {"threads": 8, "split_memory_limit": "50G"},
    "dedup": {"enabled": True, "within_split": False, "length_mode": "discrete",
              "min_seq_id": 0.95, "min_cov": 0.95, "cov_mode": 0},
    "mmseqs_bin": "mmseqs",
    "genus_balance": {"enabled": False, "scope": "global",
                      "max_species_per_genus": 1},
}

_TEMPLATE = re.compile(r"\{([a-zA-Z0-9_]+)\}")


def deep_update(base: dict, extra: dict) -> dict:
    """Recursively merge ``extra`` into ``base`` (returns ``base``)."""
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _coerce(text: str) -> Any:
    """Best-effort scalar coercion for env / --set string values."""
    low = text.lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def _apply_dotted(cfg: dict, dotted: str, value: Any) -> None:
    """Set ``cfg['a']['b'] = value`` from key ``'a.b'``."""
    parts = dotted.split(".")
    node = cfg
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def _env_overrides(cfg: dict) -> None:
    """Apply TIARA2_a__b=value env vars (double underscore => nesting)."""
    for name, raw in os.environ.items():
        if not name.startswith("TIARA2_"):
            continue
        dotted = name[len("TIARA2_"):].lower().replace("__", ".")
        _apply_dotted(cfg, dotted, _coerce(raw))


def _resolve_templates(cfg: dict) -> None:
    """Expand ``{key}`` references using top-level string values (2 passes)."""
    scalars = {k: v for k, v in cfg.items() if isinstance(v, str)}
    for _ in range(3):  # a few passes resolve chained references
        changed = False

        def sub(value):
            nonlocal changed
            if isinstance(value, str):
                new = _TEMPLATE.sub(lambda m: str(scalars.get(m.group(1), m.group(0))), value)
                if new != value:
                    changed = True
                return new
            if isinstance(value, dict):
                return {k: sub(v) for k, v in value.items()}
            if isinstance(value, list):
                return [sub(v) for v in value]
            return value

        for key in list(cfg.keys()):
            cfg[key] = sub(cfg[key])
        scalars = {k: v for k, v in cfg.items() if isinstance(v, str)}
        if not changed:
            break


class ConfigError(ValueError):
    pass


def validate(cfg: dict) -> None:
    """Fail fast on structural problems. Extend as stages are added."""
    errors = []
    if not cfg.get("base"):
        errors.append("missing required key: base")
    for key in ("splits", "split_priority", "classes"):
        if key in cfg and not isinstance(cfg[key], list):
            errors.append(f"{key} must be a list")
    if set(cfg.get("split_priority", [])) != set(cfg.get("splits", [])):
        errors.append("split_priority must be a permutation of splits")
    d = cfg.get("dedup", {})
    if d.get("length_mode") not in ("discrete", "geometric", None):
        errors.append("dedup.length_mode must be 'discrete' or 'geometric'")
    for path_key in ("min_seq_id", "min_cov"):
        if path_key in d and not (0 < d[path_key] <= 1):
            errors.append(f"dedup.{path_key} must be in (0, 1]")
    gb = cfg.get("genus_balance", {}) or {}
    if gb.get("scope") not in ("global", "per_split"):
        errors.append("genus_balance.scope must be 'global' or 'per_split'")
    try:
        if int(gb.get("max_species_per_genus", 1)) < 1:
            errors.append("genus_balance.max_species_per_genus must be >= 1")
    except (TypeError, ValueError):
        errors.append("genus_balance.max_species_per_genus must be an integer")
    # `train` is optional in small acquisition-only configs. If it is present,
    # however, both k lists are required and become the single source of truth
    # for TF-IDF, cache, HP, final models and publish.
    if "train" in cfg:
        train = cfg.get("train", {}) or {}
        for key in ("k_first", "k_second"):
            values = train.get(key)
            if not isinstance(values, list) or not values:
                errors.append(f"train.{key} must be a non-empty list")
                continue
            if any(isinstance(k, bool) or not isinstance(k, int) for k in values):
                errors.append(f"train.{key} must contain integers")
            elif any(k < 1 or k > 8 for k in values):
                errors.append(f"train.{key} values must be in [1, 8]")
            if len(values) != len(set(values)):
                errors.append(f"train.{key} must not contain duplicates")
            if values != sorted(values):
                errors.append(f"train.{key} must be sorted ascending")
        publish = cfg.get("publish", {}) or {}
        for count_key, k_key in (("first_count", "k_first"),
                                 ("second_count", "k_second")):
            if count_key in publish and isinstance(train.get(k_key), list):
                try:
                    if int(publish[count_key]) != len(train[k_key]):
                        errors.append(
                            f"publish.{count_key} must equal len(train.{k_key})")
                except (TypeError, ValueError):
                    errors.append(f"publish.{count_key} must be an integer")
    # `infer` is the last stage and is opt-in, but a typo in it must fail here
    # rather than after a full training run has already finished.
    if "infer" in cfg:
        infer = cfg.get("infer", {}) or {}
        if not isinstance(infer, dict):
            errors.append("infer must be a mapping")
            infer = {}
        if "inputs" in infer and not isinstance(infer["inputs"], list):
            errors.append("infer.inputs must be a list of files, dirs or globs")
        if infer.get("enabled") and not (infer.get("inputs") or []):
            errors.append("infer.enabled is true but infer.inputs is empty")
        train_ks = (cfg.get("train", {}) or {})
        for key, k_key in (("k_first", "k_first"), ("k_second", "k_second")):
            value = infer.get(key)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                errors.append(f"infer.{key} must be an integer or null")
                continue
            available = train_ks.get(k_key)
            if isinstance(available, list) and available and value not in available:
                errors.append(
                    f"infer.{key}={value} is not in train.{k_key}={available}; "
                    f"only trained k can be used for inference")
        cutoffs = infer.get("prob_cutoff") or {}
        if not isinstance(cutoffs, dict):
            errors.append("infer.prob_cutoff must be a mapping with "
                          "'first' and 'second'")
        else:
            for key, value in cutoffs.items():
                if key not in ("first", "second"):
                    errors.append(f"infer.prob_cutoff.{key} is not a stage "
                                  "(use 'first' / 'second')")
                elif value is not None and not (0 < float(value) < 1):
                    errors.append(f"infer.prob_cutoff.{key} must be in (0, 1)")
        for key in ("min_len", "batch_records"):
            value = infer.get(key)
            if value is None:
                continue
            try:
                if int(value) < 1:
                    errors.append(f"infer.{key} must be >= 1")
            except (TypeError, ValueError):
                errors.append(f"infer.{key} must be an integer")
        device = infer.get("device")
        if device is not None and str(device) not in ("cpu", "cuda"):
            errors.append("infer.device must be 'cpu', 'cuda' or null")
        allowed = {"mit", "pla", "bac", "arc", "euk", "unk", "pro", "org", "all"}
        to_fasta = infer.get("to_fasta") or []
        if not isinstance(to_fasta, list):
            errors.append("infer.to_fasta must be a list")
        else:
            for item in to_fasta:
                if str(item).lower() not in allowed:
                    errors.append(
                        f"infer.to_fasta contains unknown class {item!r}; "
                        "allowed: " + " ".join(sorted(allowed)))
    if errors:
        raise ConfigError("invalid config:\n  - " + "\n  - ".join(errors))


def load(path: str | os.PathLike, overrides: list[str] | None = None,
         apply_env: bool = True) -> dict:
    """Load and finalize a config from ``path``.

    ``overrides`` is a list of ``a.b=value`` strings from the CLI (--set).
    """
    if yaml is None:
        raise ConfigError("PyYAML not installed (see envs/environment.yaml)")
    raw = yaml.safe_load(Path(path).read_text()) or {}
    cfg = copy.deepcopy(DEFAULTS)
    deep_update(cfg, raw)
    if apply_env:
        _env_overrides(cfg)
    for item in overrides or []:
        key, _, value = item.partition("=")
        _apply_dotted(cfg, key.strip(), _coerce(value.strip()))
    _resolve_templates(cfg)
    validate(cfg)
    return cfg

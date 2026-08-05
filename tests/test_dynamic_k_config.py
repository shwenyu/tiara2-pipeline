#!/usr/bin/env python3
"""Regression tests for v2.2.0 config-driven k lists."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tiara2 import config


def test_v220_k_only_config_resolves():
    cfg = config.load(ROOT / "config" / "config.yaml", apply_env=False)
    t = cfg["train"]
    assert cfg["version_tag"] == "v2_2_0"
    assert cfg["model_tag"] == "v2.2.0"
    assert t["k_first"] == [5, 6, 7]
    assert t["k_second"] == [5, 6, 7, 8]
    assert t["train_ready"] == cfg["fragment_bp_balance"]["output_root"]
    assert t["train_ready"].endswith("corpus_ready_v2_1_3_bpbalanced")
    assert "v2_2_0" in t["tfidf_dir"]
    assert "v2_2_0" in t["feature_cache"]
    assert "v2_2_0" in t["seq_pack"]
    assert cfg["publish"]["first_count"] == len(t["k_first"])
    assert cfg["publish"]["second_count"] == len(t["k_second"])


def _must_fail(cfg, text):
    try:
        config.validate(cfg)
    except config.ConfigError as exc:
        assert text in str(exc), str(exc)
    else:
        raise AssertionError(f"invalid config accepted; expected {text!r}")


def test_k_lists_reject_empty_duplicate_unsorted_and_range():
    base = config.load(ROOT / "config" / "config.yaml", apply_env=False)
    cases = [
        ("k_first", [], "non-empty"),
        ("k_first", [5, 5, 6], "duplicates"),
        ("k_second", [6, 5, 7], "sorted ascending"),
        ("k_second", [5, 6, 9], "in [1, 8]"),
    ]
    for key, value, message in cases:
        cfg = copy.deepcopy(base)
        cfg["train"][key] = value
        # Keep the mirror aligned where possible, so the k error remains clear.
        count_key = "first_count" if key == "k_first" else "second_count"
        cfg["publish"][count_key] = len(value)
        _must_fail(cfg, message)


def test_publish_count_mirror_cannot_drift():
    cfg = config.load(ROOT / "config" / "config.yaml", apply_env=False)
    cfg["publish"]["first_count"] = 99
    _must_fail(cfg, "publish.first_count")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("dynamic-k tests passed")

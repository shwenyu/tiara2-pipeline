import json
from pathlib import Path

import pytest

from tiara.hierarchical.multi_expert import expert_for_length, resolve_manifest_path


@pytest.mark.parametrize(
    ("length_bp", "expected"),
    [(1000, "short"), (2499, "short"), (2500, "long"), (10000, "long")],
)
def test_deterministic_length_router(length_bp, expected):
    assert expert_for_length(length_bp, 2500) == expected


def test_published_bundle_uses_single_automatic_router():
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads(
        (root / "tiara/models/hierarchical-models-v2.4.0-multi-expert/model_manifest.json").read_text()
    )
    assert manifest["format"] == "tiara2-multi-expert-v1"
    assert manifest["router"] == {
        "type": "deterministic_length",
        "short_if_length_lt_bp": 2500,
        "threshold_selection": "roadmap_fixed_candidate_pending_benchmark_gate",
    }
    assert set(manifest["experts"]) == {"long", "short"}


def test_bundle_accepts_directory_or_manifest(tmp_path):
    manifest = tmp_path / "model_manifest.json"
    manifest.write_text("{}")
    assert resolve_manifest_path(tmp_path) == manifest.resolve()
    assert resolve_manifest_path(manifest) == manifest.resolve()


def test_bundle_reports_missing_manifest(tmp_path):
    with pytest.raises(FileNotFoundError, match="bundle manifest not found"):
        resolve_manifest_path(tmp_path)

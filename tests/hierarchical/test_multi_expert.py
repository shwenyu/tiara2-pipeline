import json
from pathlib import Path

import pytest

from tiara.hierarchical.multi_expert import expert_for_length


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

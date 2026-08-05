"""Inference-layer resolution tests.

These run WITHOUT torch/skorch/numba/Bio: tiara2.inference deliberately keeps
the heavy imports inside make_classifier(), so plan resolution -- the part that
picked the wrong model set in the past -- is testable on its own.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tiara2 import inference  # noqa: E402


def _pack(tmp, first_ks, second_ks, manifest=None):
    nnet = tmp / "nnet-models-v2.2.0"
    tfidf = tmp / "tfidf-models-v2.2.0"
    nnet.mkdir(parents=True, exist_ok=True)
    tfidf.mkdir(parents=True, exist_ok=True)
    for stage, ks in (("first", first_ks), ("second", second_ks)):
        for k in ks:
            name = (f"{stage}_k-{k}_hidden_1-2048_hidden_2-1024_"
                    f"lr-0.001_dropout-0.2_epochs-50.pkl")
            (nnet / name).write_text("x")
            folder = tfidf / f"k{k}-{stage}-stage"
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "model.npy").write_text("x")
            (folder / "params.txt").write_text(f"k:{k}\nfragment_len:5000\n")
    if manifest is not None:
        (nnet / "training_manifest.json").write_text(json.dumps(manifest))
    return nnet, tfidf


def test_discovers_v220_k_lists_without_hardcoding(tmp_path):
    """v2.2.0 ships first 5,6,7 and second 5,6,7,8 -- read, never assumed."""
    nnet, tfidf = _pack(tmp_path, [5, 6, 7], [5, 6, 7, 8])
    found = inference.discover_models(nnet, tfidf)
    assert [item["k"] for item in found["first"]] == [5, 6, 7]
    assert [item["k"] for item in found["second"]] == [5, 6, 7, 8]


def test_v213_layout_still_resolves(tmp_path):
    """The previous pack (first 4,5,6 / second 4,5,6,7) must keep working."""
    nnet, tfidf = _pack(tmp_path, [4, 5, 6], [4, 5, 6, 7])
    plan = inference.build_plan(nnet, tfidf)
    assert plan.stages["first"].k == 6
    assert plan.stages["second"].k == 7


def test_manifest_mean_f1_beats_largest_k(tmp_path):
    manifest = {"models": [
        {"stage": "first", "k": 5, "mean_f1": 0.99},
        {"stage": "first", "k": 7, "mean_f1": 0.80},
        {"stage": "second", "k": 8, "mean_f1": 0.98},
    ]}
    nnet, tfidf = _pack(tmp_path, [5, 7], [8], manifest=manifest)
    plan = inference.build_plan(nnet, tfidf)
    assert plan.stages["first"].k == 5
    assert abs(plan.stages["first"].mean_f1 - 0.99) < 1e-9


def test_pinned_k_must_exist(tmp_path):
    nnet, tfidf = _pack(tmp_path, [5, 6, 7], [5, 6, 7, 8])
    try:
        inference.build_plan(nnet, tfidf, k_first=4)
    except inference.InferenceError as exc:
        assert "k=4 requested" in str(exc)
    else:
        raise AssertionError("pinning a k that was never trained must fail")


def test_net_without_matching_tfidf_is_not_offered(tmp_path):
    """A stage x k triple is only usable if BOTH halves are present."""
    nnet, tfidf = _pack(tmp_path, [5, 6], [5, 6])
    import shutil
    shutil.rmtree(tfidf / "k6-first-stage")
    found = inference.discover_models(nnet, tfidf)
    assert [item["k"] for item in found["first"]] == [5]


def test_second_stage_missing_is_fatal(tmp_path):
    nnet, tfidf = _pack(tmp_path, [5], [])
    try:
        inference.build_plan(nnet, tfidf)
    except inference.InferenceError as exc:
        assert "second-stage" in str(exc)
    else:
        raise AssertionError("a one-stage pack must not silently classify")


def test_manifest_roundtrip_is_byte_stable(tmp_path):
    nnet, tfidf = _pack(tmp_path, [5, 6, 7], [5, 6, 7, 8])
    plan = inference.build_plan(nnet, tfidf, cutoffs={"first": 0.938553})
    text = plan.to_json()
    again = inference.InferencePlan.from_json(text)
    assert again.to_json() == text
    assert abs(again.stages["first"].prob_cutoff - 0.938553) < 1e-9


def test_hidden_2_none_is_parsed(tmp_path):
    """v2.1.3 published second_k-6 with hidden_2-none."""
    nnet, tfidf = _pack(tmp_path, [5], [6])
    for path in nnet.glob("second_k-6_*.pkl"):
        path.rename(nnet / ("second_k-6_hidden_1-128_hidden_2-none_"
                            "lr-0.01_dropout-0.5_epochs-50.pkl"))
    plan = inference.build_plan(nnet, tfidf)
    spec = plan.stages["second"]
    assert spec.hidden_2 is None
    assert spec.params()["hidden_1"] == 128
    assert spec.params()["dim_out"] == 3

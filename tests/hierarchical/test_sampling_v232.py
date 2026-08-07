import numpy as np
import pytest

from tiara.hierarchical.sampling_v232 import (
    build_hierarchical_sampling_plan,
    cap_nested_donor_distribution,
    capped_target_distribution,
)
from tiara.hierarchical.schema import EUK_WEIGHTS, schema
from tiara.hierarchical.train import DistributedWeightedSampler


def test_capped_target_never_exceeds_four_x():
    natural = np.asarray([0.50, 0.20, 0.10, 0.08, 0.05, 0.03, 0.025, 0.015])
    requested = np.asarray([0.20, 0.15, 0.13, 0.12, 0.12, 0.10, 0.10, 0.08])
    effective = capped_target_distribution(natural, requested, 4.0)
    assert effective.sum() == pytest.approx(1.0)
    assert np.all(effective / natural <= 4.0 + 1e-9)
    assert effective[-1] == pytest.approx(0.06)


def test_hierarchical_plan_balances_root_and_shapes_euk():
    sc = schema("2.3.2")
    root = np.asarray([0] * 80 + [1] * 30 + [2] * 10, dtype=np.int64)
    leaves = np.full(len(root), -1, dtype=np.int64)
    leaves[:80] = np.repeat(np.arange(8), 10)
    plan = build_hierarchical_sampling_plan(root, leaves, sc, EUK_WEIGHTS, 4.0)
    weights = plan.sample_weights
    for root_id in range(3):
        assert weights[root == root_id].sum() == pytest.approx(1 / 3)
    for leaf_id, name in enumerate(sc.classes("euk")):
        conditional = weights[(root == 0) & (leaves == leaf_id)].sum() / weights[root == 0].sum()
        assert conditional == pytest.approx(plan.report["euk_effective"][name])
    assert plan.report["donor_caps_applied"] is False


def test_missing_leaf_is_rejected():
    sc = schema("2.3.2")
    root = np.asarray([0] * 7 + [1, 2])
    leaves = np.asarray(list(range(7)) + [-1, -1])
    with pytest.raises(ValueError, match="Euk leaves are empty"):
        build_hierarchical_sampling_plan(root, leaves, sc, EUK_WEIGHTS, 4.0)


def test_nested_donor_caps_limit_dominant_accession():
    base = np.ones(20)
    accession = np.asarray([0] * 10 + list(range(1, 11)))
    species = accession.copy()
    genus = accession.copy()
    weights, maxima = cap_nested_donor_distribution(
        base, [accession, species, genus], [0.20, 0.20, 0.20]
    )
    assert weights.sum() == pytest.approx(1.0)
    assert all(maximum <= 0.20 + 1e-8 for maximum in maxima)
    assert weights[accession == 0].sum() == pytest.approx(0.20)


def test_distributed_sampler_is_deterministic_and_evenly_sharded():
    weights = np.arange(1, 25, dtype=np.float64)
    rank0 = DistributedWeightedSampler(weights, 2, 0, seed=71, drop_last=True)
    rank1 = DistributedWeightedSampler(weights, 2, 1, seed=71, drop_last=True)
    rank0.set_epoch(3)
    rank1.set_epoch(3)
    first0, first1 = list(rank0), list(rank1)
    assert first0 == list(rank0)
    assert len(first0) == len(first1) == 12
    rank0.set_epoch(4)
    assert first0 != list(rank0)

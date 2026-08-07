"""Deterministic two-level sampling policy for Tiara2 v2.3.2."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class HierarchicalSamplingPlan:
    sample_weights: np.ndarray
    report: dict


def _normalized(values: Mapping[str, float], names: Sequence[str]) -> np.ndarray:
    result = np.asarray([float(values[name]) for name in names], dtype=np.float64)
    if not np.all(np.isfinite(result)) or np.any(result < 0) or result.sum() <= 0:
        raise ValueError("target weights must be finite, non-negative, and sum above zero")
    return result / result.sum()


def capped_target_distribution(natural, requested, max_oversample=4.0):
    """Project requested probabilities under q[i] <= max_oversample * p[i]."""
    natural = np.asarray(natural, dtype=np.float64)
    requested = np.asarray(requested, dtype=np.float64)
    if natural.ndim != 1 or requested.shape != natural.shape:
        raise ValueError("natural/requested shape mismatch")
    if not np.isclose(natural.sum(), 1.0) or not np.isclose(requested.sum(), 1.0):
        raise ValueError("natural and requested probabilities must sum to one")
    if np.any(natural <= 0):
        raise ValueError("all Euk leaves must be non-empty before branch balancing")
    if not np.isfinite(max_oversample) or max_oversample < 1:
        raise ValueError("max_oversample must be finite and >= 1")
    cap = natural * float(max_oversample)
    effective = np.minimum(requested, cap)
    for _ in range(len(effective) + 2):
        remaining = 1.0 - effective.sum()
        if remaining <= 1e-12:
            break
        room = np.maximum(cap - effective, 0.0)
        eligible = room > 1e-12
        if not np.any(eligible):
            raise ValueError("oversample caps cannot form a normalized distribution")
        preference = np.where(eligible, requested, 0.0)
        if preference.sum() <= 0:
            preference = room
        addition = remaining * preference / preference.sum()
        effective += np.minimum(addition, room)
    effective /= effective.sum()
    if np.any(effective / natural > float(max_oversample) + 1e-9):
        raise AssertionError("oversample projection exceeded cap")
    return effective


def cap_nested_donor_distribution(base_row_weights, donor_codes, max_shares, iterations=50):
    """Cap nested donor-group probability while preserving total probability."""
    weights = np.asarray(base_row_weights, dtype=np.float64).copy()
    if weights.ndim != 1 or weights.sum() <= 0:
        raise ValueError("base_row_weights must be a positive 1-D array")
    weights /= weights.sum()
    if len(donor_codes) != len(max_shares):
        raise ValueError("donor_codes/max_shares length mismatch")
    arrays = [np.asarray(codes, dtype=np.int64) for codes in donor_codes]
    if any(codes.shape != weights.shape or np.any(codes < 0) for codes in arrays):
        raise ValueError("donor codes must be aligned non-negative arrays")
    caps = [float(value) for value in max_shares]
    if any(not np.isfinite(value) or not 0 < value <= 1 for value in caps):
        raise ValueError("donor max shares must be in (0,1]")
    for _ in range(int(iterations)):
        changed = False
        for codes, cap in zip(arrays, caps):
            totals = np.bincount(codes, weights=weights)
            over = totals > cap + 1e-12
            if not np.any(over):
                continue
            changed = True
            scale = np.ones_like(totals)
            scale[over] = cap / totals[over]
            weights *= scale[codes]
            remaining = 1.0 - weights.sum()
            if remaining > 1e-12:
                totals = np.bincount(codes, weights=weights)
                eligible = totals[codes] < cap - 1e-12
                mass = weights[eligible].sum()
                if mass <= 0:
                    raise ValueError("donor caps are infeasible")
                weights[eligible] += remaining * weights[eligible] / mass
            weights /= weights.sum()
        if not changed:
            break
    violations = []
    for codes, cap in zip(arrays, caps):
        maximum = float(np.bincount(codes, weights=weights).max(initial=0.0))
        violations.append(maximum)
        if maximum > cap + 1e-8:
            raise ValueError(f"donor caps did not converge: maximum={maximum:.8f}, cap={cap:.8f}")
    return weights, violations


def build_hierarchical_sampling_plan(
    root_labels,
    euk_labels,
    schema,
    target_weights: Mapping[str, float],
    max_oversample=4.0,
    donor_codes=None,
    donor_max_shares=None,
):
    """Balance root branches, then shape only the Euk leaf distribution."""
    root = np.asarray(root_labels, dtype=np.int64)
    euk = np.asarray(euk_labels, dtype=np.int64)
    if root.shape != euk.shape or root.ndim != 1:
        raise ValueError("root/euk labels must be aligned 1-D arrays")
    root_names = tuple(schema.classes("root"))
    euk_names = tuple(schema.classes("euk"))
    root_index = schema.index("root")
    if "euk_nuclear" not in root_index:
        raise ValueError("schema has no euk_nuclear root branch")
    root_counts = np.bincount(root, minlength=len(root_names))
    if len(root_counts) != len(root_names) or np.any(root_counts <= 0):
        raise ValueError("all root branches must be non-empty")
    euk_mask = root == root_index["euk_nuclear"]
    if np.any((euk[euk_mask] < 0) | (euk[euk_mask] >= len(euk_names))):
        raise ValueError("Euk rows contain missing or invalid leaf labels")
    leaf_counts = np.bincount(euk[euk_mask], minlength=len(euk_names))
    if np.any(leaf_counts <= 0):
        missing = [euk_names[i] for i, count in enumerate(leaf_counts) if count <= 0]
        raise ValueError("Euk leaves are empty: " + ", ".join(missing))
    natural = leaf_counts / leaf_counts.sum()
    requested = _normalized(target_weights, euk_names)
    effective = capped_target_distribution(natural, requested, max_oversample)

    root_target = np.full(len(root_names), 1.0 / len(root_names), dtype=np.float64)
    weights = root_target[root] / root_counts[root]
    leaf_per_row = (root_target[root_index["euk_nuclear"]] * effective) / leaf_counts
    weights[euk_mask] = leaf_per_row[euk[euk_mask]]
    donor_report = {"donor_caps_applied": False}
    if donor_codes is not None or donor_max_shares is not None:
        if donor_codes is None or donor_max_shares is None:
            raise ValueError("donor_codes and donor_max_shares must be supplied together")
        names = ("accession", "species", "genus")
        arrays = [np.asarray(donor_codes[name], dtype=np.int64) for name in names]
        if any(values.shape != (int(euk_mask.sum()),) for values in arrays):
            raise ValueError("donor arrays must align to Euk rows in feature order")
        capped = np.empty(int(euk_mask.sum()), dtype=np.float64)
        achieved = {name: 0.0 for name in names}
        euk_values = euk[euk_mask]
        for leaf_id in range(len(euk_names)):
            mask = euk_values == leaf_id
            base = np.ones(int(mask.sum()), dtype=np.float64)
            leaf_codes = [values[mask] for values in arrays]
            adjusted, maxima = cap_nested_donor_distribution(
                base,
                leaf_codes,
                [donor_max_shares[name] for name in names],
            )
            capped[mask] = effective[leaf_id] * adjusted
            for name, maximum in zip(names, maxima):
                achieved[name] = max(achieved[name], maximum)
        weights[euk_mask] = root_target[root_index["euk_nuclear"]] * capped
        donor_report = {
            "donor_caps_applied": True,
            "donor_cap_scope": "within_each_euk_leaf",
            "donor_max_shares": {name: float(donor_max_shares[name]) for name in names},
            "donor_achieved_max_shares": achieved,
        }
    weights /= weights.sum()

    report = {
        "policy": "hierarchical_branch_balancing_v1",
        "root_target": {name: float(root_target[i]) for i, name in enumerate(root_names)},
        "root_counts": {name: int(root_counts[i]) for i, name in enumerate(root_names)},
        "euk_counts": {name: int(leaf_counts[i]) for i, name in enumerate(euk_names)},
        "euk_natural": {name: float(natural[i]) for i, name in enumerate(euk_names)},
        "euk_requested": {name: float(requested[i]) for i, name in enumerate(euk_names)},
        "euk_effective": {name: float(effective[i]) for i, name in enumerate(euk_names)},
        "euk_oversample_factor": {name: float(effective[i] / natural[i]) for i, name in enumerate(euk_names)},
        "max_euk_oversample": float(max_oversample),
        "inverse_frequency_loss": False,
        **donor_report,
    }
    return HierarchicalSamplingPlan(weights, report)


__all__ = ["HierarchicalSamplingPlan", "build_hierarchical_sampling_plan", "capped_target_distribution", "cap_nested_donor_distribution"]

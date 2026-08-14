import numpy as np

from tiara.hierarchical.biosignal_multi_expert import biosignal_features


def test_biosignal_features_are_deterministic_and_60d():
    sequence = "ATGAAACCCGGGTAG" * 100
    first = biosignal_features(sequence)
    second = biosignal_features(sequence)
    assert first.shape == (60,)
    assert first.dtype == np.float32
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()


def test_biosignal_features_handle_ambiguous_bases():
    result = biosignal_features("NNNNATGNNNTAA" * 100)
    assert result.shape == (60,)
    assert np.isfinite(result).all()

from __future__ import annotations

import numpy as np
import pytest

try:
    from ._helpers import call_with_supported_kwargs
except ModuleNotFoundError:
    from ._helpers import call_with_supported_kwargs


def test_estimate_distribution_shapes_and_basic_properties():
    fv = pytest.importorskip("fairvote")
    privacy = pytest.importorskip("fairvote.privacy")
    estimate_distribution = getattr(privacy, "estimate_distribution")
    privatize_many = getattr(privacy, "privatize_many")

    rng = np.random.default_rng(123)
    k = 5
    n = 800
    epsilon = 1.0

    truth = rng.integers(0, k, size=n, dtype=int)
    reported = call_with_supported_kwargs(privatize_many, truth=truth, categories=truth, epsilon=epsilon, k=k, seed=123)
    # Some versions name args differently; fall back to positional if needed
    if reported is None:
        reported = privatize_many(truth, epsilon=epsilon, k=k, seed=123)

    p_hat = call_with_supported_kwargs(estimate_distribution, reported=reported, reported_categories=reported, epsilon=epsilon, k=k)
    if p_hat is None:
        p_hat = estimate_distribution(reported, epsilon=epsilon, k=k)

    p_hat = np.asarray(p_hat, dtype=float)
    assert p_hat.shape == (k,)
    assert np.all(np.isfinite(p_hat))
    # Allow tiny negative due to numerical noise, but not meaningful negatives
    assert float(p_hat.min()) >= -1e-8
    s = float(p_hat.sum())
    assert abs(s - 1.0) < 1e-6


def test_invalid_k_raises():
    privacy = pytest.importorskip("fairvote.privacy")
    estimate_distribution = getattr(privacy, "estimate_distribution")

    with pytest.raises(ValueError):
        estimate_distribution([0, 1, 1], epsilon=1.0, k=1)


def test_rr_mechanism_counts_from_reports_round_trip():
    kary_rr = pytest.importorskip("fairvote.privacy.mechanisms.kary_rr")
    counts_from_reports = getattr(kary_rr, "counts_from_reports")

    reports = [0, 1, 1, 2, 2, 2]
    k = 5
    counts = counts_from_reports(reports, k=k)
    assert list(counts[:3]) == [1, 2, 3]
    assert counts.shape == (k,)

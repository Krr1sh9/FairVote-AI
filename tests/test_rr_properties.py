"""Property-based tests for RR and post-stratification invariants.

These tests are intentionally invariant-oriented rather than example-oriented:
randomly generated valid inputs must still produce valid probability objects,
stable debiasing outputs, and normalised post-stratification weights.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings, strategies as st

from fairvote.inference.mrp.poststratify import normalise_poststrat_weights
from fairvote.privacy.mechanisms.kary_rr import invert_rr_counts, privatize_many, rr_params, rr_transition_matrix


@st.composite
def _count_vector(draw):
    k = draw(st.integers(min_value=2, max_value=9))
    # Keep counts small enough for shrinking/debugging but allow sparse cells.
    counts = draw(st.lists(st.integers(min_value=0, max_value=500), min_size=k, max_size=k))
    if sum(counts) == 0:
        counts[0] = 1
    return k, np.asarray(counts, dtype=int)


@settings(max_examples=80, suppress_health_check=[HealthCheck.filter_too_much])
@given(
    epsilon=st.floats(min_value=1e-4, max_value=8.0, allow_nan=False, allow_infinity=False),
    k=st.integers(min_value=2, max_value=12),
)
@pytest.mark.property
def test_rr_transition_matrix_is_row_stochastic_for_generated_inputs(epsilon: float, k: int) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        matrix = rr_transition_matrix(epsilon, k)
    assert matrix.shape == (k, k)
    assert np.all(np.isfinite(matrix))
    assert np.all(matrix >= 0.0)
    assert np.all(matrix <= 1.0)
    assert np.allclose(matrix.sum(axis=1), 1.0, atol=1e-12)


@settings(max_examples=80)
@given(payload=_count_vector(), epsilon=st.floats(min_value=0.05, max_value=5.0, allow_nan=False, allow_infinity=False))
@pytest.mark.property
def test_debiased_distribution_is_valid_probability_simplex(payload, epsilon: float) -> None:
    k, counts = payload
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        estimate = invert_rr_counts(counts, epsilon, k, clip=True, renormalize=True)
    assert estimate.shape == (k,)
    assert np.all(np.isfinite(estimate))
    assert np.all(estimate >= 0.0)
    assert np.all(estimate <= 1.0)
    assert np.isclose(float(np.sum(estimate)), 1.0, atol=1e-12)


@settings(max_examples=80)
@given(
    epsilon=st.floats(min_value=0.05, max_value=5.0, allow_nan=False, allow_infinity=False),
    k=st.integers(min_value=2, max_value=10),
    n=st.integers(min_value=1, max_value=200),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
@pytest.mark.property
def test_privatize_many_outputs_valid_categories(epsilon: float, k: int, n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, k, size=n, dtype=int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        reports = privatize_many(truth, epsilon=epsilon, k=k, rng=rng)
    assert reports.shape == truth.shape
    assert reports.dtype.kind in {"i", "u"}
    assert np.all((reports >= 0) & (reports < k))


@settings(max_examples=80)
@given(payload=_count_vector())
@pytest.mark.property
def test_poststratification_weights_are_non_negative_and_sum_to_one(payload) -> None:
    _k, counts = payload
    weights = normalise_poststrat_weights(counts)
    assert weights.shape == counts.shape
    assert np.all(np.isfinite(weights))
    assert np.all(weights >= 0.0)
    assert np.isclose(float(np.sum(weights)), 1.0, atol=1e-12)
    if counts.sum() > 0:
        assert np.allclose(weights, counts / counts.sum())


@settings(max_examples=50)
@given(
    eps_low=st.floats(min_value=0.01, max_value=2.0, allow_nan=False, allow_infinity=False),
    gap=st.floats(min_value=0.01, max_value=4.0, allow_nan=False, allow_infinity=False),
    k=st.integers(min_value=2, max_value=12),
)
@pytest.mark.property
def test_p_keep_monotonic_property(eps_low: float, gap: float, k: int) -> None:
    eps_high = eps_low + abs(gap)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        low = rr_params(eps_low, k).p_keep
        high = rr_params(eps_high, k).p_keep
    assert high >= low

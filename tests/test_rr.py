from __future__ import annotations

import math

import numpy as np
import pytest

from fairvote.privacy.mechanisms.kary_rr import (
    counts_from_reports,
    invert_rr_counts,
    privatize_many,
    rr_params,
    rr_transition_matrix,
)


def test_rr_parameters_and_transition_matrix_match_definition() -> None:
    epsilon = 1.0
    k = 3
    params = rr_params(epsilon, k)
    matrix = rr_transition_matrix(epsilon, k)

    assert params.p / params.q == pytest.approx(math.exp(epsilon))
    assert params.p + (k - 1) * params.q == pytest.approx(1.0)
    assert params.p > params.q > 0.0
    np.testing.assert_allclose(np.diag(matrix), np.full(k, params.p))
    np.testing.assert_allclose(
        matrix[~np.eye(k, dtype=bool)],
        np.full(k * (k - 1), params.q),
    )
    np.testing.assert_allclose(matrix.sum(axis=1), np.ones(k))


def test_vectorised_randomised_response_returns_valid_categories() -> None:
    rng = np.random.default_rng(1)
    truth = np.array([0, 1, 2] * 200)

    reports = privatize_many(truth, 1.0, 3, rng)

    assert reports.shape == truth.shape
    assert reports.dtype.kind == "i"
    assert reports.min() >= 0
    assert reports.max() < 3


def test_counts_from_reports_matches_known_counts() -> None:
    counts = counts_from_reports([0, 0, 1, 2, 2, 2], 3)
    np.testing.assert_array_equal(counts, [2, 1, 3])


def test_analytical_inverse_recovers_a_hand_calculated_distribution() -> None:
    epsilon = math.log(4.0)
    counts = np.array([25, 19, 16])

    estimate = invert_rr_counts(
        counts,
        epsilon,
        3,
        clip=False,
        renormalize=False,
    )

    np.testing.assert_allclose(estimate, [0.5, 0.3, 0.2], atol=1e-12)


def test_clipping_and_renormalisation_produce_a_valid_distribution() -> None:
    counts = np.array([100, 0, 0])

    raw = invert_rr_counts(
        counts,
        0.2,
        3,
        clip=False,
        renormalize=False,
    )
    cleaned = invert_rr_counts(counts, 0.2, 3)

    assert raw.min() < 0.0
    assert cleaned.min() >= 0.0
    assert cleaned.sum() == pytest.approx(1.0)


def test_invalid_core_inputs_are_rejected() -> None:
    rng = np.random.default_rng(2)

    with pytest.raises(ValueError):
        rr_params(0.0, 3)
    with pytest.raises(ValueError):
        rr_params(1.0, 1)
    with pytest.raises(ValueError):
        privatize_many([0, 3], 1.0, 3, rng)
    with pytest.raises(ValueError):
        invert_rr_counts([0, 0, 0], 1.0, 3)

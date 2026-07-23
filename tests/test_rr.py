from __future__ import annotations

import math

import numpy as np
import pytest

from fairvote.privacy.mechanisms.kary_rr import (
    counts_from_reports,
    invert_rr_counts,
    privatize_many,
    privatize_one,
    rr_params,
    rr_transition_matrix,
)


@pytest.mark.parametrize("epsilon", [0.1, 0.5, 1.0, 3.0])
@pytest.mark.parametrize("k", [2, 3, 5])
def test_transition_matrix_rows_sum_to_one(epsilon: float, k: int) -> None:
    matrix = rr_transition_matrix(epsilon, k)
    assert matrix.shape == (k, k)
    np.testing.assert_allclose(matrix.sum(axis=1), np.ones(k), atol=1e-12)


@pytest.mark.parametrize("epsilon", [0.25, 1.0, 2.5])
def test_diagonal_is_p_and_off_diagonal_is_q(epsilon: float) -> None:
    k = 3
    params = rr_params(epsilon, k)
    matrix = rr_transition_matrix(epsilon, k)

    np.testing.assert_allclose(np.diag(matrix), np.full(k, params.p), atol=1e-12)
    off_diagonal = matrix[~np.eye(k, dtype=bool)]
    np.testing.assert_allclose(off_diagonal, np.full(off_diagonal.size, params.q), atol=1e-12)


@pytest.mark.parametrize("epsilon", [0.1, 1.0, 4.0])
@pytest.mark.parametrize("k", [2, 3, 7])
def test_p_over_q_equals_exp_epsilon(epsilon: float, k: int) -> None:
    params = rr_params(epsilon, k)
    assert params.p / params.q == pytest.approx(math.exp(epsilon), rel=1e-9)


@pytest.mark.parametrize("epsilon", [0.2, 1.0, 3.0])
def test_p_and_q_satisfy_the_simplex_constraint(epsilon: float) -> None:
    k = 4
    params = rr_params(epsilon, k)
    assert params.p + (k - 1) * params.q == pytest.approx(1.0, abs=1e-12)
    assert params.p > params.q > 0.0


def test_p_increases_with_epsilon() -> None:
    values = [rr_params(eps, 3).p for eps in (0.1, 0.5, 1.0, 2.0, 5.0)]
    assert all(a < b for a, b in zip(values, values[1:], strict=False))


def test_privatize_many_returns_valid_categories() -> None:
    rng = np.random.default_rng(1)
    truth = rng.integers(0, 4, size=500)
    reports = privatize_many(truth, 0.7, 4, rng)

    assert reports.shape == truth.shape
    assert reports.dtype.kind == "i"
    assert reports.min() >= 0
    assert reports.max() <= 3


def test_privatize_one_returns_valid_categories() -> None:
    rng = np.random.default_rng(2)
    reports = [privatize_one(1, 0.7, 3, rng) for _ in range(300)]
    assert set(reports) <= {0, 1, 2}


def test_single_and_vectorised_apis_agree_on_keep_rate() -> None:
    epsilon, k, n = 1.5, 3, 20_000
    expected_p = rr_params(epsilon, k).p

    rng_one = np.random.default_rng(11)
    single_keep = np.mean([privatize_one(2, epsilon, k, rng_one) == 2 for _ in range(n)])

    rng_many = np.random.default_rng(12)
    truth = np.full(n, 2)
    many_keep = float(np.mean(privatize_many(truth, epsilon, k, rng_many) == 2))

    assert single_keep == pytest.approx(expected_p, abs=0.02)
    assert many_keep == pytest.approx(expected_p, abs=0.02)


def test_privatize_many_on_empty_input() -> None:
    rng = np.random.default_rng(3)
    assert privatize_many([], 1.0, 3, rng).size == 0


def test_raw_inverse_is_approximately_unbiased_at_large_n() -> None:
    rng = np.random.default_rng(20)
    epsilon, k, n = 1.0, 3, 200_000
    truth = np.array([0.5, 0.3, 0.2])

    categories = rng.choice(k, size=n, p=truth)
    reports = privatize_many(categories, epsilon, k, rng)
    counts = counts_from_reports(reports, k)
    estimate = invert_rr_counts(counts, epsilon, k, clip=False, renormalize=False)

    np.testing.assert_allclose(estimate, truth, atol=0.02)


def test_clipping_and_renormalisation_are_optional_and_explicit() -> None:
    epsilon, k = 0.2, 3
    counts = np.array([100, 0, 0])

    raw = invert_rr_counts(counts, epsilon, k, clip=False, renormalize=False)
    assert raw.min() < 0.0

    cleaned = invert_rr_counts(counts, epsilon, k)
    assert cleaned.min() >= 0.0
    assert cleaned.sum() == pytest.approx(1.0)


def test_large_epsilon_is_numerically_stable() -> None:
    params = rr_params(700.0, 3)
    assert math.isfinite(params.p) and math.isfinite(params.q)
    assert params.p == pytest.approx(1.0)
    assert params.q == pytest.approx(0.0, abs=1e-12)

    matrix = rr_transition_matrix(700.0, 3)
    assert np.all(np.isfinite(matrix))
    np.testing.assert_allclose(matrix.sum(axis=1), np.ones(3), atol=1e-12)


@pytest.mark.parametrize("epsilon", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_epsilon_is_rejected(epsilon: float) -> None:
    with pytest.raises(ValueError):
        rr_params(epsilon, 3)


@pytest.mark.parametrize("k", [-1, 0, 1])
def test_invalid_k_is_rejected(k: int) -> None:
    with pytest.raises(ValueError):
        rr_params(1.0, k)


def test_non_numeric_parameters_are_rejected() -> None:
    with pytest.raises(TypeError):
        rr_params("1.0", 3)
    with pytest.raises(TypeError):
        rr_params(1.0, 3.5)


@pytest.mark.parametrize("category", [-1, 3])
def test_invalid_category_is_rejected(category: int) -> None:
    rng = np.random.default_rng(4)
    with pytest.raises(ValueError):
        privatize_one(category, 1.0, 3, rng)


def test_out_of_range_categories_are_rejected_by_the_vectorised_api() -> None:
    rng = np.random.default_rng(5)
    with pytest.raises(ValueError):
        privatize_many([0, 1, 9], 1.0, 3, rng)
    with pytest.raises(ValueError):
        privatize_many([[0, 1], [1, 0]], 1.0, 3, rng)


def test_counts_from_reports_validates_and_counts() -> None:
    np.testing.assert_array_equal(counts_from_reports([0, 0, 2], 3), np.array([2, 0, 1]))
    np.testing.assert_array_equal(counts_from_reports([], 3), np.zeros(3, dtype=int))
    with pytest.raises(ValueError):
        counts_from_reports([0, 5], 3)


@pytest.mark.parametrize(
    "counts",
    [
        np.array([1.0, 2.0]),
        np.array([-1.0, 1.0, 1.0]),
        np.array([np.nan, 1.0, 1.0]),
        np.zeros(3),
    ],
)
def test_malformed_counts_are_rejected(counts: np.ndarray) -> None:
    with pytest.raises(ValueError):
        invert_rr_counts(counts, 1.0, 3)

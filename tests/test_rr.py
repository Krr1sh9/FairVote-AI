"""Tests for the canonical Python k-ary Randomized Response channel."""

from __future__ import annotations

import numpy as np
import pytest

from fairvote.privacy.mechanisms.kary_rr import (
    debias_distribution,
    invert_rr_counts,
    privatize_many,
    privatize_one,
    rr_params,
    rr_transition_matrix,
)


def test_rr_transition_matrix_rows_sum_to_one() -> None:
    for epsilon in [0.05, 0.2, 1.0, 5.0, 30.0]:
        for k in [2, 3, 7]:
            with pytest.warns(UserWarning) if k == 2 or epsilon > 10 else _does_not_warn():
                matrix = rr_transition_matrix(epsilon, k)
            assert matrix.shape == (k, k)
            assert np.all(np.isfinite(matrix))
            assert np.all(matrix >= 0.0)
            assert np.all(matrix <= 1.0)
            assert np.allclose(matrix.sum(axis=1), 1.0, atol=1e-12)


def test_transition_matrix_uses_canonical_params() -> None:
    epsilon = 1.3
    k = 7
    params = rr_params(epsilon, k)
    matrix = rr_transition_matrix(epsilon, k)

    assert np.allclose(np.diag(matrix), params.p_keep, atol=1e-12)
    off_diag = matrix[~np.eye(k, dtype=bool)]
    assert np.allclose(off_diag, params.p_flip, atol=1e-12)


def test_p_keep_monotonic_in_epsilon() -> None:
    epsilons = [0.02, 0.1, 0.5, 1.0, 2.0, 8.0]
    for k in [2, 5, 10]:
        with pytest.warns(UserWarning) if k == 2 else _does_not_warn():
            p_keeps = [rr_params(eps, k).p_keep for eps in epsilons]
        assert p_keeps == sorted(p_keeps)
        assert len(set(np.round(p_keeps, 12))) == len(p_keeps)


def test_privatize_outputs_valid_categories() -> None:
    rng = np.random.default_rng(123)
    k = 6
    truth = rng.integers(0, k, size=5000, dtype=int)
    reports = privatize_many(truth, epsilon=0.7, k=k, rng=rng)

    assert reports.shape == truth.shape
    assert reports.dtype.kind in {"i", "u"}
    assert int(reports.min()) >= 0
    assert int(reports.max()) < k

    single = privatize_one(3, epsilon=0.7, k=k, rng=np.random.default_rng(5))
    assert 0 <= single < k


def test_debiasing_is_approximately_unbiased_at_large_n() -> None:
    rng = np.random.default_rng(2025)
    k = 5
    epsilon = 1.2
    n = 250_000
    true_dist = np.array([0.07, 0.18, 0.31, 0.16, 0.28], dtype=float)
    truth = rng.choice(k, size=n, p=true_dist)
    reports = privatize_many(truth, epsilon=epsilon, k=k, rng=rng)

    estimate = debias_distribution(reports, epsilon=epsilon, k=k, clip=False, renormalize=False)
    assert np.allclose(estimate, true_dist, atol=0.015)

    counts = np.bincount(reports, minlength=k)
    estimate_from_counts = invert_rr_counts(counts, epsilon=epsilon, k=k, clip=False, renormalize=False)
    assert np.allclose(estimate_from_counts, estimate, atol=1e-12)


def test_large_epsilon_is_numerically_stable() -> None:
    with pytest.warns(UserWarning):
        params = rr_params(1000.0, 4)
    assert np.isfinite(params.p_keep)
    assert np.isfinite(params.p_flip)
    assert params.p_keep == pytest.approx(1.0)
    assert params.p_flip == pytest.approx(0.0)

    with pytest.warns(UserWarning):
        matrix = rr_transition_matrix(1000.0, 4)
    assert np.allclose(matrix, np.eye(4), atol=1e-15)


def test_invalid_epsilon_and_k_are_rejected() -> None:
    for bad_epsilon in [0.0, -0.1, float("inf"), float("nan")]:
        with pytest.raises(ValueError):
            rr_params(bad_epsilon, 3)

    for bad_k in [0, 1, 1.5, "3"]:
        with pytest.raises((TypeError, ValueError)):
            rr_params(1.0, bad_k)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        privatize_one(3, epsilon=1.0, k=3)
    with pytest.raises(ValueError):
        privatize_many([0, 1, 4], epsilon=1.0, k=3)


class _does_not_warn:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False

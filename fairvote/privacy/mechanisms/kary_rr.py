"""Canonical k-ary Randomized Response (RR) channel for FairVote-AI.

This module is the single authoritative Python implementation of the local
privacy channel used across the package, experiments, MRP models, and dashboard.
All Python code that needs RR probabilities, transition matrices, privatisation,
or analytic debiasing should import from here rather than reimplementing the
formula.

Mechanism
---------
For ``k`` categories and privacy parameter ``epsilon``:

* keep/report the selected category with probability ``p``
* otherwise report one of the other ``k - 1`` categories uniformly with
  per-category probability ``q``

The implementation computes ``p`` and ``q`` in a stable form using
``exp(-epsilon)`` so that large epsilon values do not overflow.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

IntArrayLike = Sequence[int] | np.ndarray


@dataclass(frozen=True)
class KaryRRParams:
    """Parameters for the canonical k-ary Randomized Response channel."""

    epsilon: float
    k: int
    p: float  # P(report true category)
    q: float  # P(report any single non-true category)

    @property
    def p_keep(self) -> float:
        """Alias used by docs/tests for the truthful-report probability."""

        return self.p

    @property
    def p_flip(self) -> float:
        """Alias for the per-category flip probability."""

        return self.q


# ---------------------------------------------------------------------------
# Core parameter computation
# ---------------------------------------------------------------------------


def rr_params(epsilon: float, k: int) -> KaryRRParams:
    """Return numerically stable k-ary RR probabilities.

    Parameters
    ----------
    epsilon:
        Positive finite privacy budget. Smaller values give stronger privacy
        and noisier reports.
    k:
        Number of response categories. Must be an integer >= 2.

    Returns
    -------
    KaryRRParams
        ``p`` is the probability of reporting the true category; ``q`` is the
        probability of reporting any one specific alternative category.
    """

    epsilon_f, k_i = _validate_epsilon_k(epsilon, k)

    if epsilon_f > 10.0:
        warnings.warn(
            f"epsilon={epsilon_f} is very high. This provides minimal privacy protection. "
            "Use a smaller epsilon when privacy, rather than near-truthful reporting, is required.",
            UserWarning,
            stacklevel=2,
        )

    if k_i < 3:
        warnings.warn(
            f"k={k_i} is very small. Binary RR is valid but has limited category entropy.",
            UserWarning,
            stacklevel=2,
        )

    # Stable parameterisation.  The algebraically equivalent expression based
    # on exp(epsilon) overflows for large epsilon.  exp(-epsilon) underflows
    # safely to 0, which corresponds to p -> 1 and q -> 0.
    exp_neg = math.exp(-epsilon_f)
    denom = 1.0 + (k_i - 1) * exp_neg
    p = 1.0 / denom
    q = exp_neg / denom

    # In finite precision, p + (k-1)q should be 1 to numerical tolerance.  Do
    # not force-correct it because the stable q expression is more informative
    # for large epsilon than deriving q from 1-p, which can suffer cancellation.
    return KaryRRParams(epsilon=epsilon_f, k=k_i, p=float(p), q=float(q))


def rr_transition_matrix(epsilon: float, k: int) -> np.ndarray:
    """Return the row-stochastic RR transition matrix ``A``.

    ``A[t, r]`` is the probability of reporting category ``r`` when the true or
    stated category is ``t``.
    """

    params = rr_params(epsilon, k)
    matrix = np.full((params.k, params.k), params.q, dtype=float)
    np.fill_diagonal(matrix, params.p)
    return matrix


def privatize_one(
    true_category: int,
    epsilon: float,
    k: int,
    rng: np.random.Generator | None = None,
) -> int:
    """Apply k-ary RR to one category and return an integer in ``[0, k-1]``."""

    _epsilon_f, k_i = _validate_epsilon_k(epsilon, k)
    _validate_category(true_category, k_i)
    generator = rng if rng is not None else np.random.default_rng()
    params = rr_params(_epsilon_f, k_i)

    if generator.random() < params.p:
        return int(true_category)

    raw = int(generator.integers(0, k_i - 1))
    return raw if raw < int(true_category) else raw + 1


def privatize_many(
    true_categories: IntArrayLike,
    epsilon: float,
    k: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Vectorised k-ary RR over a 1D array of categories."""

    _epsilon_f, k_i = _validate_epsilon_k(epsilon, k)
    arr = np.asarray(true_categories, dtype=int)
    if arr.ndim != 1:
        raise ValueError("true_categories must be a 1D array-like of ints.")
    if arr.size == 0:
        return np.array([], dtype=int)
    if np.any((arr < 0) | (arr >= k_i)):
        raise ValueError(f"All true categories must be in [0, {k_i - 1}].")

    generator = rng if rng is not None else np.random.default_rng()
    params = rr_params(_epsilon_f, k_i)
    n = int(arr.size)

    keep_true = generator.random(n) < params.p
    out = np.empty(n, dtype=int)
    out[keep_true] = arr[keep_true]

    flip_idx = np.where(~keep_true)[0]
    if flip_idx.size:
        raw = generator.integers(0, k_i - 1, size=flip_idx.size, dtype=int)
        true_values = arr[flip_idx]
        out[flip_idx] = np.where(raw < true_values, raw, raw + 1)

    return out


# ---------------------------------------------------------------------------
# Aggregation and debiasing utilities
# ---------------------------------------------------------------------------


def counts_from_reports(reported_categories: IntArrayLike, k: int) -> np.ndarray:
    """Convert reported category labels into a length-``k`` count vector."""

    k_i = _validate_k(k)
    reports = np.asarray(reported_categories, dtype=int)
    if reports.ndim != 1:
        raise ValueError("reported_categories must be a 1D array-like of ints.")
    if reports.size == 0:
        return np.zeros(k_i, dtype=int)
    if np.any((reports < 0) | (reports >= k_i)):
        raise ValueError(f"All reported categories must be in [0, {k_i - 1}].")
    return np.bincount(reports, minlength=k_i).astype(int)


def invert_rr_counts(
    counts: IntArrayLike,
    epsilon: float,
    k: int,
    *,
    clip: bool = True,
    renormalize: bool = True,
) -> np.ndarray:
    """Estimate the latent true distribution from RR reported counts.

    This is the canonical analytic inverse of the k-ary RR observation channel.
    Finite-sample estimates may fall outside the probability simplex; by
    default they are clipped and renormalised for downstream reporting.
    """

    params = rr_params(epsilon, k)
    count_vec = np.asarray(counts, dtype=float)
    if count_vec.ndim != 1 or count_vec.size != params.k:
        raise ValueError(f"counts must be a 1D array of length {params.k}.")
    if np.any(~np.isfinite(count_vec)):
        raise ValueError("counts must be finite.")
    if np.any(count_vec < 0):
        raise ValueError("counts must be non-negative.")

    n = float(np.sum(count_vec))
    if n <= 0.0:
        raise ValueError("counts sum must be > 0.")

    reported_dist = count_vec / n
    denom = params.p - params.q
    if denom <= 0.0 or not math.isfinite(denom):
        raise RuntimeError("Invalid RR parameters: p - q must be finite and positive.")

    theta_hat = (reported_dist - params.q) / denom

    if clip:
        theta_hat = np.clip(theta_hat, 0.0, 1.0)

    if renormalize:
        total = float(np.sum(theta_hat))
        if total <= 0.0 or not math.isfinite(total):
            theta_hat = np.full(params.k, 1.0 / params.k, dtype=float)
        else:
            theta_hat = theta_hat / total

    return theta_hat.astype(float, copy=False)


def debias_distribution(
    reported_categories: IntArrayLike,
    epsilon: float,
    k: int,
    *,
    clip: bool = True,
    renormalize: bool = True,
) -> np.ndarray:
    """Estimate the latent true distribution from raw RR reported labels."""

    counts = counts_from_reports(reported_categories, k)
    return invert_rr_counts(counts, epsilon, k, clip=clip, renormalize=renormalize)


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------


def _validate_epsilon_k(epsilon: float, k: int) -> tuple[float, int]:
    k_i = _validate_k(k)
    if not isinstance(epsilon, Real):
        raise TypeError("epsilon must be a real number.")
    epsilon_f = float(epsilon)
    if not math.isfinite(epsilon_f):
        raise ValueError("epsilon must be finite.")
    if epsilon_f <= 0.0:
        raise ValueError("epsilon must be > 0.")
    return epsilon_f, k_i


def _validate_k(k: int) -> int:
    if not isinstance(k, Integral):
        raise TypeError("k must be an int.")
    k_i = int(k)
    if k_i < 2:
        raise ValueError("k must be >= 2.")
    return k_i


def _validate_category(category: int, k: int) -> None:
    if not isinstance(category, Integral):
        raise TypeError("true_category must be an int.")
    if int(category) < 0 or int(category) >= k:
        raise ValueError(f"true_category must be in [0, {k - 1}].")

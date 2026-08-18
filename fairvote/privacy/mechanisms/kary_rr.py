from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

IntArrayLike = Sequence[int] | np.ndarray


@dataclass(frozen=True)
class RRParams:
    # epsilon is the privacy-budget value used to parameterise the k-ary Randomised Response mechanism.
    epsilon: float

    # k is the number of possible response categories.
    k: int

    # p is the probability that the mechanism reports the respondent's true category.
    p: float

    # q is the probability assigned to each specific category different from the respondent's true category.
    q: float


def rr_params(epsilon: float, k: int) -> RRParams:
    epsilon_f, k_i = _validate_epsilon_k(epsilon, k)

    # Using exp(-epsilon) gives the same probabilities as the standard form while avoiding an unnecessary positive exponential.
    exp_neg = math.exp(-epsilon_f)
    denominator = 1.0 + (k_i - 1) * exp_neg

    # These probabilities satisfy p plus (k - 1) times q equals one and p divided by q equals exp(epsilon).
    return RRParams(
        epsilon=epsilon_f,
        k=k_i,
        p=float(1.0 / denominator),
        q=float(exp_neg / denominator),
    )


def rr_transition_matrix(epsilon: float, k: int) -> np.ndarray:
    params = rr_params(epsilon, k)

    # Off-diagonal entries use q and diagonal entries use p, matching the probabilities of reporting each category.
    matrix = np.full((params.k, params.k), params.q, dtype=float)
    np.fill_diagonal(matrix, params.p)
    return matrix


def privatize_one(
    true_category: int,
    epsilon: float,
    k: int,
    rng: np.random.Generator,
) -> int:
    epsilon_f, k_i = _validate_epsilon_k(epsilon, k)
    _validate_category(true_category, k_i)
    params = rr_params(epsilon_f, k_i)

    # The true category is reported directly with probability p.
    if rng.random() < params.p:
        return int(true_category)

    # Otherwise one of the other k - 1 categories is chosen uniformly, giving each alternative probability q overall.
    drawn = int(rng.integers(0, k_i - 1))
    return drawn if drawn < int(true_category) else drawn + 1


def privatize_many(
    true_categories: IntArrayLike,
    epsilon: float,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    epsilon_f, k_i = _validate_epsilon_k(epsilon, k)
    categories = np.asarray(true_categories, dtype=int)
    if categories.ndim != 1:
        raise ValueError("true_categories must be a 1D array of ints.")
    if categories.size == 0:
        return np.array([], dtype=int)
    if np.any((categories < 0) | (categories >= k_i)):
        raise ValueError(f"All true categories must be in [0, {k_i - 1}].")

    params = rr_params(epsilon_f, k_i)
    n = int(categories.size)

    # Independent uniform draws decide which respondents keep their true category.
    keep_true = rng.random(n) < params.p
    reports = np.empty(n, dtype=int)
    reports[keep_true] = categories[keep_true]

    # Respondents not keeping the truth are assigned uniformly among the other k - 1 categories.
    flipped = np.flatnonzero(~keep_true)
    if flipped.size:
        drawn = rng.integers(0, k_i - 1, size=flipped.size, dtype=int)
        true_values = categories[flipped]

        # Values at or above the true category are shifted by one so the true category cannot be selected as an alternative.
        reports[flipped] = np.where(drawn < true_values, drawn, drawn + 1)

    return reports


def counts_from_reports(reported_categories: IntArrayLike, k: int) -> np.ndarray:
    k_i = _validate_k(k)
    reports = np.asarray(reported_categories, dtype=int)
    if reports.ndim != 1:
        raise ValueError("reported_categories must be a 1D array of ints.")
    if reports.size == 0:
        return np.zeros(k_i, dtype=int)
    if np.any((reports < 0) | (reports >= k_i)):
        raise ValueError(f"All reported categories must be in [0, {k_i - 1}].")

    # minlength keeps zero-count categories in their fixed positions within the length-k count vector.
    return np.bincount(reports, minlength=k_i).astype(int)


def invert_rr_counts(
    counts: IntArrayLike,
    epsilon: float,
    k: int,
    *,
    clip: bool = True,
    renormalize: bool = True,
) -> np.ndarray:
    params = rr_params(epsilon, k)
    count_vector = np.asarray(counts, dtype=float)
    if count_vector.ndim != 1 or count_vector.size != params.k:
        raise ValueError(f"counts must be a 1D array of length {params.k}.")
    if not np.all(np.isfinite(count_vector)):
        raise ValueError("counts must be finite.")
    if np.any(count_vector < 0):
        raise ValueError("counts must be non-negative.")

    total = float(count_vector.sum())
    if total <= 0.0:
        raise ValueError("counts must sum to more than 0 because estimation is impossible with no reports.")

    # The expected reported frequency for category j is q plus (p - q) times its true population proportion.
    reported_frequencies = count_vector / total
    theta_hat = (reported_frequencies - params.q) / (params.p - params.q)

    # Finite-sample noise can make the unconstrained analytical inversion fall outside the probability range.
    if clip:
        theta_hat = np.clip(theta_hat, 0.0, 1.0)

    # Renormalisation makes the returned values sum to one when requested.
    if renormalize:
        mass = float(theta_hat.sum())
        theta_hat = np.full(params.k, 1.0 / params.k) if mass <= 0.0 else theta_hat / mass

    return np.asarray(theta_hat, dtype=float)


def _validate_epsilon_k(epsilon: float, k: int) -> tuple[float, int]:
    k_i = _validate_k(k)
    if not isinstance(epsilon, Real) or isinstance(epsilon, bool):
        raise TypeError("epsilon must be a real number.")
    epsilon_f = float(epsilon)
    if not math.isfinite(epsilon_f):
        raise ValueError("epsilon must be finite.")
    if epsilon_f <= 0.0:
        raise ValueError("epsilon must be > 0.")
    return epsilon_f, k_i


def _validate_k(k: int) -> int:
    if not isinstance(k, Integral) or isinstance(k, bool):
        raise TypeError("k must be an int.")
    k_i = int(k)
    if k_i < 2:
        raise ValueError("k must be >= 2.")
    return k_i


def _validate_category(category: int, k: int) -> None:
    if not isinstance(category, Integral) or isinstance(category, bool):
        raise TypeError("true_category must be an int.")
    if not 0 <= int(category) < k:
        raise ValueError(f"true_category must be in [0, {k - 1}].")

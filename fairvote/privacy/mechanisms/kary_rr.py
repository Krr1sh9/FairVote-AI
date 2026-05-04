"""k-ary Randomized Response mechanisms and transition utilities.

This module defines the local privacy channel used by FairVote-AI. The
mechanism maps a true category to a reported category before collection, so
analysts only observe privatized reports.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np


# Type alias used by the vectorised privatisation functions below.
IntArrayLike = Union[Sequence[int], np.ndarray]


@dataclass(frozen=True)
class KaryRRParams:
    """
    Parameters for k-ary Randomized Response (RR).

    Mechanism:
      - true category t in {0..K-1}
      - report t with prob p
      - otherwise report one of the other K-1 categories uniformly

    For epsilon > 0:
      p = exp(eps) / (exp(eps) + K - 1)
      q = (1 - p) / (K - 1)
    """
    epsilon: float
    k: int
    p: float
    q: float


# ---------------------------------------------------------------------------
# Core parameter computation
# ---------------------------------------------------------------------------

def rr_params(epsilon: float, k: int) -> KaryRRParams:
    """
    Compute stable RR parameters p and q for given epsilon and number of categories k.
    Uses a numerically stable form to avoid overflow.

    p = 1 / (1 + (k-1)*exp(-epsilon))
    q = (1 - p) / (k - 1)
    """
    # Validate the privacy budget and category count before doing any numerical
    # work, because invalid RR parameters would make the probability channel
    # meaningless rather than merely imprecise.
    _validate_epsilon_k(epsilon, k)

    # Stable computation for p:
    # p = exp(eps)/(exp(eps)+k-1) = 1/(1+(k-1)*exp(-eps))
    # Guard against overflow: exp(epsilon) can be huge for large epsilon, but
    # exp(-epsilon) is always safe for positive epsilon.  The threshold of 700
    # avoids IEEE 754 inf while yielding p ≈ 1 in the extreme case.
    exp_neg = math.exp(-epsilon) if epsilon < 700 else 0.0
    p = 1.0 / (1.0 + (k - 1) * exp_neg)
    # q is the per-category probability of flipping to any single other option.
    q = (1.0 - p) / (k - 1)
    return KaryRRParams(epsilon=epsilon, k=k, p=p, q=q)


def privatize_one(
    true_category: int,
    epsilon: float,
    k: int,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """
    Apply k-ary RR to a single true category.
    Returns an integer in [0, k-1].
    """
    _validate_epsilon_k(epsilon, k)
    _validate_category(true_category, k)

    if rng is None:
        rng = np.random.default_rng()

    params = rr_params(epsilon, k)

    # With probability p the client reports the true category. This is still
    # part of the local randomisation mechanism; the server cannot tell whether
    # an individual report was truthful or produced by the random branch.
    if rng.random() < params.p:
        return int(true_category)

    # Choose uniformly among the other categories so that every non-true label
    # receives the same q probability prescribed by k-ary RR.
    r = int(rng.integers(0, k - 1))
    # Map r in [0..k-2] to [0..k-1] excluding true_category
    return r if r < true_category else r + 1


def privatize_many(
    true_categories: IntArrayLike,
    epsilon: float,
    k: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Vectorized k-ary RR over many true categories.

    Returns: np.ndarray of shape (n,) of ints in [0, k-1].
    """
    # Validate once for the entire batch rather than per element.
    _validate_epsilon_k(epsilon, k)

    arr = np.asarray(true_categories, dtype=int)
    if arr.ndim != 1:
        raise ValueError("true_categories must be a 1D array-like of ints.")
    if arr.size == 0:
        return np.array([], dtype=int)

    if np.any((arr < 0) | (arr >= k)):
        raise ValueError(f"All true categories must be in [0, {k-1}].")

    if rng is None:
        rng = np.random.default_rng()

    params = rr_params(epsilon, k)
    n = arr.size

    # Draw one Bernoulli decision per respondent. Keeping this vectorised avoids
    # changing the mechanism while making large synthetic runs much faster.
    keep_true = rng.random(n) < params.p
    out = np.empty(n, dtype=int)
    out[keep_true] = arr[keep_true]

    # Only respondents on the random branch need an alternative category.
    idx = np.where(~keep_true)[0]
    if idx.size > 0:
        # Draw r in [0..k-2] and map to exclude the true category for each element
        r = rng.integers(0, k - 1, size=idx.size, dtype=int)
        t = arr[idx]
        out[idx] = np.where(r < t, r, r + 1)

    return out


# ---------------------------------------------------------------------------
# Aggregation utility
# ---------------------------------------------------------------------------

def counts_from_reports(reported_categories: IntArrayLike, k: int) -> np.ndarray:
    """
    Convert reported category labels into a length-k count vector.
    """
    if not isinstance(k, int) or k < 2:
        raise ValueError("k must be an int >= 2.")

    reps = np.asarray(reported_categories, dtype=int)
    if reps.ndim != 1:
        raise ValueError("reported_categories must be a 1D array-like of ints.")
    if reps.size == 0:
        return np.zeros(k, dtype=int)
    if np.any((reps < 0) | (reps >= k)):
        raise ValueError(f"All reported categories must be in [0, {k-1}].")

    # minlength=k ensures the output always has exactly k bins, even if some
    # categories receive zero reports in a small sample.
    return np.bincount(reps, minlength=k).astype(int)


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------

def _validate_epsilon_k(epsilon: float, k: int) -> None:
    if not isinstance(k, int):
        raise TypeError("k must be an int.")
    if k < 2:
        raise ValueError("k must be >= 2.")
    if not isinstance(epsilon, (int, float)):
        raise TypeError("epsilon must be a float.")
    if not math.isfinite(float(epsilon)):
        raise ValueError("epsilon must be finite.")
    if float(epsilon) <= 0.0:
        raise ValueError("epsilon must be > 0.")


def _validate_category(category: int, k: int) -> None:
    if not isinstance(category, (int, np.integer)):
        raise TypeError("true_category must be an int.")
    if category < 0 or category >= k:
        raise ValueError(f"true_category must be in [0, {k-1}].")

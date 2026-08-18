from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from fairvote.privacy.mechanisms.kary_rr import (
    IntArrayLike,
    counts_from_reports,
    invert_rr_counts,
)

FloatArrayLike = Sequence[float] | np.ndarray


def raw_frequencies(reported_categories: IntArrayLike, k: int) -> np.ndarray:
    # Raw frequencies use the observed privatised report counts directly without applying RR inversion.
    counts = counts_from_reports(reported_categories, k)
    total = int(counts.sum())
    if total <= 0:
        raise ValueError("reported_categories must be non-empty.")

    # Dividing each count by the total produces a probability distribution over the k reported categories.
    return counts.astype(float) / float(total)


def debiased_estimate(
    reported_categories: IntArrayLike,
    epsilon: float,
    k: int,
    *,
    clip: bool = True,
    renormalize: bool = True,
) -> np.ndarray:
    # The RR correction is applied to the report counts using the analytical inversion implemented in kary_rr.py.
    counts = counts_from_reports(reported_categories, k)

    # clip and renormalize are passed through so callers can request either a constrained distribution or the raw analytical inversion.
    return invert_rr_counts(counts, epsilon, k, clip=clip, renormalize=renormalize)


def project_distribution(values: FloatArrayLike) -> np.ndarray:
    # This helper converts a finite one-dimensional vector into a valid probability distribution by clipping and renormalising.
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("values must be a non-empty 1D array.")
    if not np.all(np.isfinite(vector)):
        raise ValueError("values must be finite.")

    # Values outside the probability range are first clipped to the interval from zero to one.
    clipped = np.clip(vector, 0.0, 1.0)
    mass = float(clipped.sum())

    # If clipping leaves no positive mass, a uniform distribution provides a valid deterministic fallback.
    if mass <= 0.0:
        return np.full(vector.size, 1.0 / vector.size)

    # Otherwise the clipped values are renormalised to sum to one.
    return np.asarray(clipped / mass, dtype=float)

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
    counts = counts_from_reports(reported_categories, k)
    total = int(counts.sum())
    if total <= 0:
        raise ValueError("reported_categories must be non-empty.")
    return counts.astype(float) / float(total)


def debiased_estimate(
    reported_categories: IntArrayLike,
    epsilon: float,
    k: int,
    *,
    clip: bool = True,
    renormalize: bool = True,
) -> np.ndarray:
    counts = counts_from_reports(reported_categories, k)
    return invert_rr_counts(counts, epsilon, k, clip=clip, renormalize=renormalize)


def project_distribution(values: FloatArrayLike) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("values must be a non-empty 1D array.")
    if not np.all(np.isfinite(vector)):
        raise ValueError("values must be finite.")

    clipped = np.clip(vector, 0.0, 1.0)
    mass = float(clipped.sum())
    if mass <= 0.0:
        return np.full(vector.size, 1.0 / vector.size)
    return np.asarray(clipped / mass, dtype=float)

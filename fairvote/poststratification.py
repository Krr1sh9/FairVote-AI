from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fairvote.privacy.estimators import debiased_estimate, project_distribution
from fairvote.privacy.mechanisms.kary_rr import IntArrayLike


@dataclass(frozen=True)
class PoststratifiedEstimate:
    estimate: np.ndarray
    cell_estimates: np.ndarray
    fallback_cells: int


def poststratified_estimate(
    cell_indices: IntArrayLike,
    reported_categories: IntArrayLike,
    weights: np.ndarray,
    epsilon: float,
    k: int,
) -> PoststratifiedEstimate:
    cells = np.asarray(cell_indices, dtype=int)
    reports = np.asarray(reported_categories, dtype=int)
    weight_array = np.asarray(weights, dtype=float)

    if cells.ndim != 1 or reports.ndim != 1:
        raise ValueError("cell_indices and reported_categories must be 1D arrays.")
    if cells.shape != reports.shape:
        raise ValueError("cell_indices and reported_categories must have the same length.")
    if cells.size == 0:
        raise ValueError("the sample must contain at least one respondent.")
    if weight_array.ndim != 1 or weight_array.size == 0:
        raise ValueError("weights must be a non-empty 1D array.")
    if not np.all(np.isfinite(weight_array)) or np.any(weight_array < 0.0):
        raise ValueError("weights must be finite and non-negative.")

    weight_total = float(weight_array.sum())
    if weight_total <= 0.0:
        raise ValueError("weights must sum to a positive value.")

    n_cells = int(weight_array.size)
    if np.any((cells < 0) | (cells >= n_cells)):
        raise ValueError(f"cell_indices must be in [0, {n_cells - 1}].")

    normalized_weights = weight_array / weight_total
    raw_overall = debiased_estimate(reports, epsilon, k, clip=False, renormalize=False)

    raw_cell_estimates = np.empty((n_cells, int(k)), dtype=float)
    fallback_cells = 0
    for cell in range(n_cells):
        cell_reports = reports[cells == cell]
        if cell_reports.size == 0:
            raw_cell_estimates[cell] = raw_overall
            fallback_cells += 1
        else:
            raw_cell_estimates[cell] = debiased_estimate(
                cell_reports,
                epsilon,
                k,
                clip=False,
                renormalize=False,
            )

    combined = normalized_weights @ raw_cell_estimates
    estimate = project_distribution(combined)

    return PoststratifiedEstimate(
        estimate=estimate,
        cell_estimates=raw_cell_estimates,
        fallback_cells=fallback_cells,
    )

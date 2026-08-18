from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fairvote.privacy.estimators import debiased_estimate, project_distribution
from fairvote.privacy.mechanisms.kary_rr import IntArrayLike


@dataclass(frozen=True)
class PoststratifiedEstimate:
    # estimate is the final population-level distribution after weighting and final clipping and renormalisation.
    estimate: np.ndarray

    # cell_estimates stores the unconstrained RR-corrected value used for each cell, with the whole-sample fallback used for empty cells.
    cell_estimates: np.ndarray

    # fallback_cells counts demographic cells that had no sampled respondents and therefore used the whole-sample fallback.
    fallback_cells: int


def poststratified_estimate(
    cell_indices: IntArrayLike,
    reported_categories: IntArrayLike,
    weights: np.ndarray,
    epsilon: float,
    k: int,
) -> PoststratifiedEstimate:
    # Inputs are converted to NumPy arrays so validation and cell-wise operations use consistent types.
    cells = np.asarray(cell_indices, dtype=int)
    reports = np.asarray(reported_categories, dtype=int)
    weight_array = np.asarray(weights, dtype=float)

    # Cell identifiers and privatised reports must form matching one-dimensional respondent-level arrays.
    if cells.ndim != 1 or reports.ndim != 1:
        raise ValueError("cell_indices and reported_categories must be 1D arrays.")
    if cells.shape != reports.shape:
        raise ValueError("cell_indices and reported_categories must have the same length.")
    if cells.size == 0:
        raise ValueError("the sample must contain at least one respondent.")

    # Population weights must provide one finite non-negative value for each demographic cell.
    if weight_array.ndim != 1 or weight_array.size == 0:
        raise ValueError("weights must be a non-empty 1D array.")
    if not np.all(np.isfinite(weight_array)) or np.any(weight_array < 0.0):
        raise ValueError("weights must be finite and non-negative.")

    # The weights are normalised internally, so they only need a positive total rather than already summing to one.
    weight_total = float(weight_array.sum())
    if weight_total <= 0.0:
        raise ValueError("weights must sum to a positive value.")

    n_cells = int(weight_array.size)

    # Every respondent's cell index must refer to one of the cells represented by the supplied weights.
    if np.any((cells < 0) | (cells >= n_cells)):
        raise ValueError(f"cell_indices must be in [0, {n_cells - 1}].")

    normalized_weights = weight_array / weight_total

    # The whole-sample RR inversion is kept unconstrained because it is used only as the fallback input for empty cells.
    raw_overall = debiased_estimate(reports, epsilon, k, clip=False, renormalize=False)

    # Each row stores one demographic cell's unconstrained RR inversion before population weighting.
    raw_cell_estimates = np.empty((n_cells, int(k)), dtype=float)
    fallback_cells = 0
    for cell in range(n_cells):
        cell_reports = reports[cells == cell]

        # An empty cell uses the unconstrained whole-sample RR inversion rather than attempting an inversion with no reports.
        if cell_reports.size == 0:
            raw_cell_estimates[cell] = raw_overall
            fallback_cells += 1
        else:
            # Non-empty cells are corrected separately without clipping or renormalising at the cell level.
            raw_cell_estimates[cell] = debiased_estimate(
                cell_reports,
                epsilon,
                k,
                clip=False,
                renormalize=False,
            )

    # Known population weights combine the cell estimates before any final constraint is applied.
    combined = normalized_weights @ raw_cell_estimates

    # Final clipping and renormalisation are applied once after the weighted cell estimates have been combined.
    estimate = project_distribution(combined)

    return PoststratifiedEstimate(
        estimate=estimate,
        cell_estimates=raw_cell_estimates,
        fallback_cells=fallback_cells,
    )

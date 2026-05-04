"""Post-stratification helpers for fitted RR-aware MRP models.

Post-stratification applies model-predicted latent true-category probabilities
to known or synthetic population cell counts. The fitted model remains
responsible for the RR-aware observation model; this module only performs the
population weighting step.
"""

# fairvote/inference/mrp/poststratify.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from fairvote.inference.mrp.model import DesignInfo, RRMultinomialModel, build_design_matrix


@dataclass(frozen=True)
class PostStratResult:
    """
    Post-stratification outputs.

    - overall: np.ndarray shape (K,)
    - by_feature: dict feature -> dict level_name -> np.ndarray shape (K,)
    - cell_theta: np.ndarray shape (C, K) predicted theta for each poststrat cell
    - cell_counts: np.ndarray shape (C,) counts for each cell
    """
    overall: np.ndarray
    by_feature: Dict[str, Dict[str, np.ndarray]]
    cell_theta: np.ndarray
    cell_counts: np.ndarray


def build_features_from_cells(
    cells: np.ndarray,
    by: Sequence[str],
) -> Dict[str, np.ndarray]:
    """
    Convert poststrat cell matrix to a features dict.

    Args:
      cells: np.ndarray shape (C, d), integer-coded feature levels
      by: list of feature names in the same order as columns of cells

    Returns:
      dict feature -> np.ndarray shape (C,)
    """
    cells = np.asarray(cells, dtype=int)
    if cells.ndim != 2:
        raise ValueError("cells must be a 2D array of shape (C, d).")
    if cells.shape[1] != len(by):
        raise ValueError("cells.shape[1] must equal len(by).")

    out: Dict[str, np.ndarray] = {}
    for j, f in enumerate(by):
        out[f] = cells[:, j].astype(int)
    return out


def poststratify(
    model: RRMultinomialModel,
    *,
    cells: np.ndarray,
    counts: np.ndarray,
    by: Sequence[str],
    design_info: DesignInfo,
    include_features: Optional[Sequence[str]] = None,
) -> PostStratResult:
    """
    Post-stratify a fitted RRMultinomialModel using a poststratification table.

    Inputs:
      model: fitted RRMultinomialModel
      cells: np.ndarray (C, d) integer-coded strata cells (d=len(by))
      counts: np.ndarray (C,) population counts in each cell
      by: the feature names corresponding to columns in `cells`
      design_info: DesignInfo used for encoding (must match model training)
      include_features: which single features to produce subgroup breakdowns for
                        (default: all features in design_info.feature_names)

    Outputs:
      - overall distribution over true categories: sum_c count_c * theta(cell_c) / total
      - subgroup distribution for each level of each selected feature:
          sum_{c in subgroup} count_c * theta(cell_c) / sum_{c in subgroup} count_c

    This is the classic MRP "P" step (Poststratification) after the "MR" modelling step.
    """
    if model.W is None:
        raise RuntimeError("Model must be fitted before poststratify().")

    cells = np.asarray(cells, dtype=int)
    counts = np.asarray(counts, dtype=float)

    if cells.ndim != 2:
        raise ValueError("cells must be 2D (C, d).")
    if counts.ndim != 1 or counts.size != cells.shape[0]:
        raise ValueError("counts must be 1D with length equal to number of cells.")
    if np.any(counts < 0):
        raise ValueError("counts must be non-negative.")
    total = float(np.sum(counts))
    if total <= 0:
        raise ValueError("counts must sum to > 0.")

    if len(by) != cells.shape[1]:
        raise ValueError("len(by) must match cells.shape[1].")

    # Build cell features from the post-stratification table. These are
    # population cells, not individual respondent records.
    cell_features = build_features_from_cells(cells, by)

    # Encode using the SAME feature ordering/levels as training
    # We do this by creating a features dict containing all features in design_info.feature_names.
    # If a training feature isn't in `by`, we cannot poststratify properly.
    missing = [f for f in design_info.feature_names if f not in cell_features]
    if missing:
        raise ValueError(
            "Poststrat table does not contain all features used in training.\n"
            f"Missing: {missing}\n"
            f"Available columns: {list(cell_features.keys())}\n"
            "Fix: build poststrat_table(pop, by=[...]) including ALL training features."
        )

    # Ensure correct feature order + levels. Reusing the training encoding is
    # essential: a different one-hot column order would silently change model
    # interpretation even though the matrix shape might still look plausible.
    X_cells, _info = build_design_matrix(
        {f: cell_features[f] for f in design_info.feature_names},
        design_info.feature_levels,
        feature_order=design_info.feature_names,
        intercept=design_info.has_intercept,
    )
    if _info.n_cols != design_info.n_cols:
        raise RuntimeError("Design matrix columns mismatch (encoding inconsistency).")

    cell_theta = model.predict_theta(X_cells)  # (C, K)
    K = cell_theta.shape[1]

    # Overall estimate: population-weighted average of cell-level latent
    # probabilities.  This is the standard MRP "P" calculation.
    overall = (counts[:, None] * cell_theta).sum(axis=0) / total

    # Subgroup estimates: marginalise over all other features to obtain per-
    # level distributions for each requested feature dimension.
    if include_features is None:
        include_features = list(design_info.feature_names)
    else:
        include_features = list(include_features)

    by_feature: Dict[str, Dict[str, np.ndarray]] = {}
    for feat in include_features:
        if feat not in design_info.feature_levels:
            continue
        # Find which column in `cells` corresponds to this feature in `by`
        if feat not in by:
            continue
        j = list(by).index(feat)
        levels = design_info.feature_levels[feat]

        sub_out: Dict[str, np.ndarray] = {}
        for lvl_idx, lvl_name in enumerate(levels):
            mask = (cells[:, j] == lvl_idx)
            w = counts[mask]
            tot = float(np.sum(w))
            if tot <= 0:
                continue
            # Marginalise: weight only the cells belonging to this level,
            # then renormalise by the level's total population mass.
            est = (w[:, None] * cell_theta[mask]).sum(axis=0) / tot
            sub_out[lvl_name] = est
        by_feature[feat] = sub_out

    return PostStratResult(
        overall=overall.astype(float),
        by_feature=by_feature,
        cell_theta=cell_theta.astype(float),
        cell_counts=counts.astype(float),
    )


def poststratify_overall_only(
    model: RRMultinomialModel,
    *,
    cells: np.ndarray,
    counts: np.ndarray,
    by: Sequence[str],
    design_info: DesignInfo,
) -> np.ndarray:
    """
    Convenience wrapper: return only the overall post-stratified estimate.
    """
    res = poststratify(
        model,
        cells=cells,
        counts=counts,
        by=by,
        design_info=design_info,
        include_features=[],
    )
    return res.overall

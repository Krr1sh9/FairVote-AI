"""Post-stratification helpers for fitted linear/neural MRP-style models."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from fairvote.inference.mrp.design import DesignInfo, build_design_matrix


class ThetaPredictor(Protocol):
    """Protocol for fitted models that can predict latent probabilities."""

    W: np.ndarray | None

    def predict_theta(self, X: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class PostStratResult:
    """Post-stratification outputs."""

    overall: np.ndarray
    by_feature: dict[str, dict[str, np.ndarray]]
    cell_theta: np.ndarray
    cell_counts: np.ndarray
    weights: np.ndarray


def build_features_from_cells(cells: np.ndarray, by: Sequence[str]) -> dict[str, np.ndarray]:
    """Convert integer-coded poststratification cells to feature arrays."""
    cells = np.asarray(cells)
    if cells.ndim != 2:
        raise ValueError("cells must be a 2D array of shape (C, d)")
    if cells.shape[0] == 0:
        raise ValueError("cells cannot be empty")
    if cells.shape[1] != len(by):
        raise ValueError("cells.shape[1] must equal len(by)")
    if not np.all(np.isfinite(cells.astype(float))):
        raise ValueError("cells contains NaN or infinite values")
    if np.issubdtype(cells.dtype, np.floating) and not np.all(np.equal(cells, np.floor(cells))):
        raise ValueError("cells must contain integer-coded categories")
    cells_int = cells.astype(int, copy=False)
    return {feature: cells_int[:, j].astype(int) for j, feature in enumerate(by)}


def _validate_cells_against_design(cells: np.ndarray, by: Sequence[str], design_info: DesignInfo) -> None:
    by_list = list(by)
    if len(by_list) != cells.shape[1]:
        raise ValueError("len(by) must match cells.shape[1]")
    missing = [feature for feature in design_info.feature_names if feature not in by_list]
    if missing:
        raise ValueError(
            "Poststrat table does not contain all features used in training. "
            f"Missing: {missing}. Available columns: {by_list}"
        )
    for feature in design_info.feature_names:
        column = by_list.index(feature)
        level_count = len(design_info.feature_levels[feature])
        values = cells[:, column]
        if np.any((values < 0) | (values >= level_count)):
            raise ValueError(f"Poststrat cells for feature '{feature}' contain values outside [0, {level_count - 1}]")


def normalise_poststrat_weights(counts: np.ndarray, *, expected_n: int | None = None) -> np.ndarray:
    """Validate counts and return weights that sum to one."""
    weights = np.asarray(counts, dtype=float).reshape(-1)
    if expected_n is not None and weights.size != int(expected_n):
        raise ValueError("counts must have one entry per poststratification cell")
    if weights.size == 0:
        raise ValueError("counts cannot be empty")
    if not np.all(np.isfinite(weights)):
        raise ValueError("counts contains NaN or infinite values")
    if np.any(weights < 0):
        raise ValueError("counts must be non-negative")
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("counts must sum to > 0")
    return weights / total


def _normalise_distribution(p: np.ndarray, *, name: str) -> np.ndarray:
    out = np.asarray(p, dtype=float).reshape(-1)
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} contains NaN or infinite values")
    out = np.clip(out, 0.0, 1.0)
    total = float(np.sum(out))
    if total <= 0.0:
        raise ValueError(f"{name} sums to zero")
    return out / total


def poststratify(
    model: ThetaPredictor,
    *,
    cells: np.ndarray,
    counts: np.ndarray,
    by: Sequence[str],
    design_info: DesignInfo,
    include_features: Sequence[str] | None = None,
) -> PostStratResult:
    """Post-stratify a fitted model using integer-coded population cells."""
    if getattr(model, "W", None) is None:
        raise RuntimeError("Model must be fitted before poststratify()")

    cells_arr = np.asarray(cells)
    cell_features = build_features_from_cells(cells_arr, by)
    cells_int = cells_arr.astype(int, copy=False)
    _validate_cells_against_design(cells_int, by, design_info)
    weights = normalise_poststrat_weights(counts, expected_n=cells_int.shape[0])

    X_cells, info = build_design_matrix(
        {feature: cell_features[feature] for feature in design_info.feature_names},
        design_info.feature_levels,
        feature_order=design_info.feature_names,
        intercept=design_info.has_intercept,
    )
    if info.n_cols != design_info.n_cols:
        raise RuntimeError("Design matrix columns mismatch (encoding inconsistency)")

    cell_theta = np.asarray(model.predict_theta(X_cells), dtype=float)
    if cell_theta.ndim != 2 or cell_theta.shape[0] != cells_int.shape[0]:
        raise ValueError("model.predict_theta returned an invalid matrix for population cells")

    overall = _normalise_distribution((weights[:, None] * cell_theta).sum(axis=0), name="overall")

    selected_features = list(design_info.feature_names) if include_features is None else list(include_features)

    by_feature: dict[str, dict[str, np.ndarray]] = {}
    by_list = list(by)
    for feature in selected_features:
        if feature not in design_info.feature_levels or feature not in by_list:
            continue
        column = by_list.index(feature)
        levels = design_info.feature_levels[feature]
        feature_out: dict[str, np.ndarray] = {}
        for level_idx, level_name in enumerate(levels):
            mask = cells_int[:, column] == level_idx
            if not np.any(mask):
                continue
            level_weights = weights[mask]
            level_total = float(np.sum(level_weights))
            if level_total <= 0.0:
                continue
            estimate = (level_weights[:, None] * cell_theta[mask]).sum(axis=0) / level_total
            feature_out[level_name] = _normalise_distribution(estimate, name=f"{feature}={level_name}")
        by_feature[feature] = feature_out

    return PostStratResult(
        overall=overall.astype(float),
        by_feature=by_feature,
        cell_theta=cell_theta.astype(float),
        cell_counts=np.asarray(counts, dtype=float).reshape(-1),
        weights=weights.astype(float),
    )


def poststratify_overall_only(
    model: ThetaPredictor,
    *,
    cells: np.ndarray,
    counts: np.ndarray,
    by: Sequence[str],
    design_info: DesignInfo,
) -> np.ndarray:
    """Return only the overall post-stratified distribution."""
    return poststratify(
        model,
        cells=cells,
        counts=counts,
        by=by,
        design_info=design_info,
        include_features=[],
    ).overall

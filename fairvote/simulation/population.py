from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# These constants define the labels used by the fixed synthetic population.
REGIONS: tuple[str, ...] = ("North", "South")
AGE_GROUPS: tuple[str, ...] = ("18-34", "35-54", "55+")
CATEGORY_NAMES: tuple[str, ...] = ("Option A", "Option B", "Option C")


@dataclass(frozen=True)
class Population:
    # Each demographic cell has one region label, one age-group label and one population weight.
    regions: tuple[str, ...]
    age_groups: tuple[str, ...]
    weights: np.ndarray

    # Each row of preferences gives the category distribution within the corresponding demographic cell.
    preferences: np.ndarray
    category_names: tuple[str, ...]

    @property
    def n_cells(self) -> int:
        # The number of demographic cells is determined by the number of population weights.
        return int(self.weights.size)

    @property
    def n_categories(self) -> int:
        # The number of response categories is determined by the number of preference columns.
        return int(self.preferences.shape[1])

    @property
    def cell_labels(self) -> tuple[str, ...]:
        # Region and age-group labels are paired positionally to produce one display label per demographic cell.
        return tuple(f"{r} / {a}" for r, a in zip(self.regions, self.age_groups, strict=True))

    def true_distribution(self) -> np.ndarray:
        # Weighting the within-cell preference distributions by population shares gives the overall population distribution.
        return np.asarray(self.weights @ self.preferences, dtype=float)


def make_population(
    regions: Sequence[str],
    age_groups: Sequence[str],
    weights: Sequence[float],
    preferences: Sequence[Sequence[float]],
    category_names: Sequence[str] = CATEGORY_NAMES,
) -> Population:
    # Inputs are converted to floating-point arrays before shape and value validation.
    weight_array = np.asarray(weights, dtype=float)
    preference_array = np.asarray(preferences, dtype=float)

    # The population requires one non-empty weight vector and one preference row for each demographic cell.
    if weight_array.ndim != 1 or weight_array.size == 0:
        raise ValueError("weights must be a non-empty 1D array.")
    if preference_array.ndim != 2 or preference_array.shape[0] != weight_array.size:
        raise ValueError("preferences must have one row per demographic cell.")

    # Region and age-group label sequences must align with the same demographic-cell ordering as the weights.
    if len(regions) != weight_array.size or len(age_groups) != weight_array.size:
        raise ValueError("regions and age_groups must have one label per cell.")

    # Each preference column corresponds to one response-category name.
    if preference_array.shape[1] != len(category_names):
        raise ValueError("preferences must have one column per category name.")

    # Population weights and within-cell preference values must be finite and non-negative.
    _check_finite_non_negative(weight_array, "weights")
    _check_finite_non_negative(preference_array, "preferences")

    # Population weights are normalised internally, so the supplied values only need a positive total.
    weight_total = float(weight_array.sum())
    if weight_total <= 0.0:
        raise ValueError("weights must sum to a positive value.")

    # Each preference row is also normalised internally and therefore must have positive total mass.
    row_totals = preference_array.sum(axis=1)
    if np.any(row_totals <= 0.0):
        raise ValueError("every preference row must sum to a positive value.")

    # The returned Population stores normalised cell weights and normalised within-cell category distributions.
    return Population(
        regions=tuple(regions),
        age_groups=tuple(age_groups),
        weights=weight_array / weight_total,
        preferences=preference_array / row_totals[:, None],
        category_names=tuple(category_names),
    )


def default_population() -> Population:
    # Two regions crossed with three age groups create the six fixed demographic cells used throughout the study.
    return make_population(
        regions=tuple(region for region in REGIONS for _ in AGE_GROUPS),
        age_groups=AGE_GROUPS * len(REGIONS),
        weights=(0.10, 0.15, 0.20, 0.15, 0.20, 0.20),
        preferences=(
            (0.55, 0.30, 0.15),
            (0.45, 0.35, 0.20),
            (0.30, 0.50, 0.20),
            (0.60, 0.25, 0.15),
            (0.40, 0.40, 0.20),
            (0.20, 0.60, 0.20),
        ),
    )


def _check_finite_non_negative(values: np.ndarray, name: str) -> None:
    # This shared validation rejects non-finite values before checking the non-negativity constraint.
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite.")
    if np.any(values < 0.0):
        raise ValueError(f"{name} must be non-negative.")

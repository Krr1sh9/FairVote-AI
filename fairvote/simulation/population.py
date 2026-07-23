from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

REGIONS: tuple[str, ...] = ("North", "South")
AGE_GROUPS: tuple[str, ...] = ("18-34", "35-54", "55+")
CATEGORY_NAMES: tuple[str, ...] = ("Option A", "Option B", "Option C")


@dataclass(frozen=True)
class Population:
    regions: tuple[str, ...]
    age_groups: tuple[str, ...]
    weights: np.ndarray
    preferences: np.ndarray
    category_names: tuple[str, ...]

    @property
    def n_cells(self) -> int:
        return int(self.weights.size)

    @property
    def n_categories(self) -> int:
        return int(self.preferences.shape[1])

    @property
    def cell_labels(self) -> tuple[str, ...]:
        return tuple(f"{r} / {a}" for r, a in zip(self.regions, self.age_groups, strict=True))

    def true_distribution(self) -> np.ndarray:
        return np.asarray(self.weights @ self.preferences, dtype=float)


def make_population(
    regions: Sequence[str],
    age_groups: Sequence[str],
    weights: Sequence[float],
    preferences: Sequence[Sequence[float]],
    category_names: Sequence[str] = CATEGORY_NAMES,
) -> Population:
    weight_array = np.asarray(weights, dtype=float)
    preference_array = np.asarray(preferences, dtype=float)

    if weight_array.ndim != 1 or weight_array.size == 0:
        raise ValueError("weights must be a non-empty 1D array.")
    if preference_array.ndim != 2 or preference_array.shape[0] != weight_array.size:
        raise ValueError("preferences must have one row per demographic cell.")
    if len(regions) != weight_array.size or len(age_groups) != weight_array.size:
        raise ValueError("regions and age_groups must have one label per cell.")
    if preference_array.shape[1] != len(category_names):
        raise ValueError("preferences must have one column per category name.")

    _check_finite_non_negative(weight_array, "weights")
    _check_finite_non_negative(preference_array, "preferences")

    weight_total = float(weight_array.sum())
    if weight_total <= 0.0:
        raise ValueError("weights must sum to a positive value.")
    row_totals = preference_array.sum(axis=1)
    if np.any(row_totals <= 0.0):
        raise ValueError("every preference row must sum to a positive value.")

    return Population(
        regions=tuple(regions),
        age_groups=tuple(age_groups),
        weights=weight_array / weight_total,
        preferences=preference_array / row_totals[:, None],
        category_names=tuple(category_names),
    )


def default_population() -> Population:
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
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite.")
    if np.any(values < 0.0):
        raise ValueError(f"{name} must be non-negative.")

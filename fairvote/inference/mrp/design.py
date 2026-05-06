"""Design-matrix builders for FairVote's MRP-style estimators.

Two encoders are provided because the repository has two legitimate data shapes:

* :func:`build_design_matrix` encodes integer-coded categorical arrays used by
  the synthetic experiment pipeline and post-stratification tables.
* :class:`DesignMatrix` encodes dictionaries of string-valued demographics used
  by the dashboard/respondent CSV workflow.

Neither encoder performs any modelling.  They exist so the canonical linear
RR-aware model in :mod:`fairvote.inference.mrp.linear` has one consistent input
format: a numeric 2-D design matrix.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class DesignInfo:
    """Metadata needed to reproduce an integer-coded one-hot encoding."""

    feature_names: List[str]
    feature_levels: Dict[str, List[str]]
    col_slices: Dict[str, slice]
    n_cols: int
    has_intercept: bool

    def to_jsonable(self) -> dict:
        """Return metadata with slices converted to JSON-friendly tuples."""
        data = asdict(self)
        data["col_slices"] = {k: [v.start, v.stop, v.step] for k, v in self.col_slices.items()}
        return data


@dataclass(frozen=True)
class FeatureSpec:
    """Levels retained for one categorical string feature.

    The first category is the dropped baseline in :class:`DesignMatrix`.
    """

    name: str
    categories: List[str]

    def to_jsonable(self) -> dict:
        return {"name": self.name, "categories": list(self.categories)}


def _validate_feature_order(
    features: Dict[str, np.ndarray],
    feature_levels: Dict[str, List[str]],
    feature_order: Sequence[str] | None,
) -> list[str]:
    if feature_order is None:
        order = sorted(features.keys())
    else:
        order = list(feature_order)
    if not order:
        raise ValueError("feature_order cannot be empty")
    duplicates = sorted({f for f in order if order.count(f) > 1})
    if duplicates:
        raise ValueError(f"feature_order contains duplicates: {duplicates}")
    for feature in order:
        if feature not in features:
            raise KeyError(f"Missing feature '{feature}' in features dict")
        if feature not in feature_levels:
            raise KeyError(f"Missing feature '{feature}' in feature_levels dict")
        if len(feature_levels[feature]) < 1:
            raise ValueError(f"Feature '{feature}' must define at least one level")
    return order


def _validate_integer_feature_values(values: np.ndarray, *, feature: str, n_levels: int) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"Feature '{feature}' must be 1D")
    if raw.size == 0:
        raise ValueError(f"Feature '{feature}' is empty")
    if np.issubdtype(raw.dtype, np.floating):
        if not np.all(np.isfinite(raw)):
            raise ValueError(f"Feature '{feature}' contains NaN or infinite values")
        if not np.all(np.equal(raw, np.floor(raw))):
            raise ValueError(f"Feature '{feature}' must contain integer-coded categories")
    arr = raw.astype(int, copy=False)
    if np.any((arr < 0) | (arr >= n_levels)):
        raise ValueError(f"Feature '{feature}' has values outside [0, {n_levels - 1}]")
    return arr


def build_design_matrix(
    features: Dict[str, np.ndarray],
    feature_levels: Dict[str, List[str]],
    *,
    feature_order: Optional[Sequence[str]] = None,
    intercept: bool = True,
) -> Tuple[np.ndarray, DesignInfo]:
    """Build a full one-hot design matrix from integer-coded features.

    Each feature contributes one indicator column per level.  This encoder is
    deterministic and suitable for synthetic population cells where every level
    must be represented explicitly.
    """
    if not features:
        raise ValueError("features cannot be empty")
    order = _validate_feature_order(features, feature_levels, feature_order)

    n: int | None = None
    validated: dict[str, np.ndarray] = {}
    for feature in order:
        arr = _validate_integer_feature_values(
            features[feature],
            feature=feature,
            n_levels=len(feature_levels[feature]),
        )
        if n is None:
            n = int(arr.size)
        elif int(arr.size) != n:
            raise ValueError("All features must have the same length")
        validated[feature] = arr

    if n is None or n <= 0:
        raise ValueError("No rows found")

    d = (1 if intercept else 0) + sum(len(feature_levels[f]) for f in order)
    X = np.zeros((n, d), dtype=np.float32)
    col = 0
    if intercept:
        X[:, 0] = 1.0
        col = 1

    col_slices: Dict[str, slice] = {}
    row_idx = np.arange(n)
    for feature in order:
        level_count = len(feature_levels[feature])
        sl = slice(col, col + level_count)
        col_slices[feature] = sl
        X[row_idx, col + validated[feature]] = 1.0
        col += level_count

    return X, DesignInfo(
        feature_names=list(order),
        feature_levels={f: list(feature_levels[f]) for f in order},
        col_slices=col_slices,
        n_cols=d,
        has_intercept=bool(intercept),
    )


class DesignMatrix:
    """String-row categorical encoder for dashboard/demo MRP.

    The encoding is intercept plus one-hot indicators with one dropped baseline
    per feature.  This is regularised multinomial regression input, not a full
    hierarchical Bayesian model design.
    """

    def __init__(self, feature_names: Sequence[str], *, require_columns: bool = True):
        if not feature_names:
            raise ValueError("feature_names cannot be empty")
        names = [str(name).strip() for name in feature_names]
        if any(not name for name in names):
            raise ValueError("feature_names cannot contain empty names")
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"feature_names contains duplicates: {duplicates}")
        self.feature_names = names
        self.require_columns = bool(require_columns)
        self.specs: List[FeatureSpec] = []
        self._col_offsets: Dict[str, Tuple[int, int]] = {}

    def _validate_rows(self, rows: Sequence[Dict[str, str]], *, fitted: bool) -> list[Dict[str, str]]:
        checked = list(rows)
        if not checked:
            raise ValueError("rows cannot be empty")
        if self.require_columns:
            missing: dict[str, int] = {name: 0 for name in self.feature_names}
            for row in checked:
                for name in self.feature_names:
                    if name not in row:
                        missing[name] += 1
            missing = {k: v for k, v in missing.items() if v}
            if missing:
                phase = "transform" if fitted else "fit"
                raise ValueError(f"Cannot {phase} design matrix: missing feature columns {missing}")
        return checked

    def fit(self, rows: Sequence[Dict[str, str]]) -> "DesignMatrix":
        checked = self._validate_rows(rows, fitted=False)
        specs: List[FeatureSpec] = []
        for feature in self.feature_names:
            vals: list[str] = []
            for row in checked:
                value = str(row.get(feature, "")).strip()
                if value:
                    vals.append(value)
            categories = sorted(set(vals)) or ["(missing)"]
            specs.append(FeatureSpec(name=feature, categories=categories))
        self.specs = specs
        offset = 1
        self._col_offsets = {}
        for spec in self.specs:
            width = max(0, len(spec.categories) - 1)
            self._col_offsets[spec.name] = (offset, width)
            offset += width
        return self

    @property
    def n_features(self) -> int:
        total = 1
        for spec in self.specs:
            total += max(0, len(spec.categories) - 1)
        return total

    def transform(self, rows: Sequence[Dict[str, str]]) -> np.ndarray:
        if not self.specs:
            raise RuntimeError("DesignMatrix must be fitted before transform()")
        checked = self._validate_rows(rows, fitted=True)
        n = len(checked)
        d = self.n_features
        X = np.zeros((n, d), dtype=float)
        X[:, 0] = 1.0

        category_maps = {spec.name: {category: i for i, category in enumerate(spec.categories)} for spec in self.specs}
        for row_idx, row in enumerate(checked):
            for spec in self.specs:
                value = str(row.get(spec.name, "")).strip()
                level_idx = category_maps[spec.name].get(value, 0)
                if level_idx > 0:
                    start, _width = self._col_offsets[spec.name]
                    X[row_idx, start + level_idx - 1] = 1.0
        return X

    def feature_columns(self) -> List[str]:
        if not self.specs:
            return ["intercept"]
        cols = ["intercept"]
        for spec in self.specs:
            baseline = spec.categories[0]
            for category in spec.categories[1:]:
                cols.append(f"{spec.name}={category} (baseline={baseline})")
        return cols

    def to_jsonable(self) -> dict:
        return {
            "feature_names": list(self.feature_names),
            "require_columns": self.require_columns,
            "feature_columns": self.feature_columns(),
            "specs": [spec.to_jsonable() for spec in self.specs],
        }

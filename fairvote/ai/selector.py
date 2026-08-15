from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text

from fairvote.ai.features import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    feature_matrix,
    target_column,
)
from fairvote.study import METHODS

MODEL_PARAMETERS: dict[str, int] = {
    "max_depth": 4,
    "min_samples_leaf": 20,
    "random_state": 42,
}

GROUPING_COLUMNS: tuple[str, ...] = ("epsilon", "n_respondents", "bias")


@dataclass(frozen=True)
class FittedSelector:
    models: dict[str, DecisionTreeRegressor]
    tie_threshold: float


@dataclass(frozen=True)
class Recommendation:
    predicted_errors: dict[str, float]
    best_method: str
    approximate_tie: bool
    tied_methods: tuple[str, ...]
    tie_threshold: float


def tie_threshold(dataset: pd.DataFrame) -> float:
    deviations: list[float] = []
    for _, group in dataset.groupby(list(GROUPING_COLUMNS), sort=True):
        for column in TARGET_COLUMNS:
            value = float(group[column].std(ddof=1))
            if np.isfinite(value):
                deviations.append(value)
    if not deviations:
        return 0.0
    return float(np.median(np.asarray(deviations, dtype=float)))


def train_selector(dataset: pd.DataFrame) -> FittedSelector:
    features = feature_matrix(dataset)
    models: dict[str, DecisionTreeRegressor] = {}
    for method in METHODS:
        model = DecisionTreeRegressor(**MODEL_PARAMETERS)
        model.fit(features, dataset[target_column(method)].to_numpy(dtype=float))
        models[method] = model
    return FittedSelector(models=models, tie_threshold=tie_threshold(dataset))


def predict_errors(selector: FittedSelector, features: np.ndarray) -> np.ndarray:
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError(f"features must have {len(FEATURE_COLUMNS)} columns.")
    return np.column_stack([selector.models[method].predict(matrix) for method in METHODS])


def recommend_from_predictions(predicted: np.ndarray, threshold: float) -> Recommendation:
    values = np.asarray(predicted, dtype=float).ravel()
    if values.size != len(METHODS):
        raise ValueError(f"predicted must contain {len(METHODS)} values.")
    order = np.argsort(values, kind="stable")
    best_index = int(order[0])
    second_index = int(order[1])
    gap = float(values[second_index] - values[best_index])
    approximate_tie = gap <= float(threshold)
    tied = tuple(METHODS[index] for index in sorted((best_index, second_index))) if approximate_tie else ()
    return Recommendation(
        predicted_errors={method: float(values[index]) for index, method in enumerate(METHODS)},
        best_method=METHODS[best_index],
        approximate_tie=approximate_tie,
        tied_methods=tied,
        tie_threshold=float(threshold),
    )


def recommend(selector: FittedSelector, features: np.ndarray) -> Recommendation:
    predicted = predict_errors(selector, features)
    if predicted.shape[0] != 1:
        raise ValueError("recommend expects a single feature row.")
    return recommend_from_predictions(predicted[0], selector.tie_threshold)


def best_method_indices(predicted: np.ndarray) -> np.ndarray:
    values = np.asarray(predicted, dtype=float)
    return np.argsort(values, axis=1, kind="stable")[:, 0]


def export_tree_rules(selector: FittedSelector, method: str) -> str:
    if method not in selector.models:
        raise ValueError(f"method must be one of {tuple(selector.models)}, got {method!r}.")
    return export_text(selector.models[method], feature_names=list(FEATURE_COLUMNS), decimals=4)

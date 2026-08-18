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

# All three estimator-error models use the same fixed decision-tree parameters.
MODEL_PARAMETERS: dict[str, int] = {
    "max_depth": 4,
    "min_samples_leaf": 20,
    "random_state": 42,
}

# These columns define the experiment settings within which variation across repetition seeds is measured for the approximate-tie threshold.
GROUPING_COLUMNS: tuple[str, ...] = ("epsilon", "n_respondents", "bias")


@dataclass(frozen=True)
class FittedSelector:
    # One fitted regression tree is stored for each estimator.
    models: dict[str, DecisionTreeRegressor]

    # The same empirical tie threshold is used when interpreting the predicted errors from the fitted selector.
    tie_threshold: float


@dataclass(frozen=True)
class Recommendation:
    # predicted_errors maps each estimator to its model-predicted L1 error.
    predicted_errors: dict[str, float]

    # best_method is always the estimator with the lowest predicted L1 error after deterministic ordering.
    best_method: str

    # approximate_tie records whether the two lowest predicted errors are separated by no more than the threshold.
    approximate_tie: bool

    # tied_methods contains the two lowest-predicted estimators when an approximate tie is reported.
    tied_methods: tuple[str, ...]

    # tie_threshold records the threshold used for this recommendation.
    tie_threshold: float


def tie_threshold(dataset: pd.DataFrame) -> float:
    deviations: list[float] = []

    # Each group keeps epsilon, sample size and sampling-bias condition fixed while allowing repetition seeds to vary.
    for _, group in dataset.groupby(list(GROUPING_COLUMNS), sort=True):
        # The sample standard deviation is calculated separately for each estimator's observed L1-error target.
        for column in TARGET_COLUMNS:
            value = float(group[column].std(ddof=1))

            # Non-finite values can occur when a group does not contain enough rows to define a sample standard deviation.
            if np.isfinite(value):
                deviations.append(value)

    # A zero threshold is used if no finite within-group standard deviations are available.
    if not deviations:
        return 0.0

    # The global threshold is the median of all finite target standard deviations collected across the groups.
    return float(np.median(np.asarray(deviations, dtype=float)))


def train_selector(dataset: pd.DataFrame) -> FittedSelector:
    # feature_matrix selects only the approved model features and keeps them in the shared feature order.
    features = feature_matrix(dataset)
    models: dict[str, DecisionTreeRegressor] = {}

    # A separate regression tree predicts the L1 error of each estimator from the same feature matrix.
    for method in METHODS:
        model = DecisionTreeRegressor(**MODEL_PARAMETERS)
        model.fit(features, dataset[target_column(method)].to_numpy(dtype=float))
        models[method] = model

    # The tie threshold is estimated from the same dataset supplied for fitting.
    return FittedSelector(models=models, tie_threshold=tie_threshold(dataset))


def predict_errors(selector: FittedSelector, features: np.ndarray) -> np.ndarray:
    matrix = np.asarray(features, dtype=float)

    # A one-dimensional feature row is converted to the two-dimensional shape expected by scikit-learn.
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)

    # Every prediction row must contain exactly the six features defined in FEATURE_COLUMNS.
    if matrix.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError(f"features must have {len(FEATURE_COLUMNS)} columns.")

    # The output columns follow METHODS order so each predicted-error column has a fixed estimator meaning.
    return np.column_stack([selector.models[method].predict(matrix) for method in METHODS])


def recommend_from_predictions(predicted: np.ndarray, threshold: float) -> Recommendation:
    # This function interprets one predicted L1-error value for each estimator.
    values = np.asarray(predicted, dtype=float).ravel()
    if values.size != len(METHODS):
        raise ValueError(f"predicted must contain {len(METHODS)} values.")

    # Stable sorting makes equal predicted errors resolve according to their existing METHODS order.
    order = np.argsort(values, kind="stable")
    best_index = int(order[0])
    second_index = int(order[1])

    # Approximate-tie handling compares only the two lowest predicted estimator errors.
    gap = float(values[second_index] - values[best_index])
    approximate_tie = gap <= float(threshold)

    # tied_methods is populated only when the two lowest predictions meet the approximate-tie rule.
    tied = tuple(METHODS[index] for index in sorted((best_index, second_index))) if approximate_tie else ()

    # A deterministic best method is retained even when the interface reports an approximate tie.
    return Recommendation(
        predicted_errors={method: float(values[index]) for index, method in enumerate(METHODS)},
        best_method=METHODS[best_index],
        approximate_tie=approximate_tie,
        tied_methods=tied,
        tie_threshold=float(threshold),
    )


def recommend(selector: FittedSelector, features: np.ndarray) -> Recommendation:
    # The recommendation interface is intentionally limited to one poll at a time.
    predicted = predict_errors(selector, features)
    if predicted.shape[0] != 1:
        raise ValueError("recommend expects a single feature row.")

    # The fitted selector supplies the empirical threshold used to interpret the three predicted errors.
    return recommend_from_predictions(predicted[0], selector.tie_threshold)


def best_method_indices(predicted: np.ndarray) -> np.ndarray:
    # Stable sorting returns the first estimator in METHODS order when predicted errors are exactly equal.
    values = np.asarray(predicted, dtype=float)
    return np.argsort(values, axis=1, kind="stable")[:, 0]


def export_tree_rules(selector: FittedSelector, method: str) -> str:
    # Tree rules can only be exported for an estimator model contained in this fitted selector.
    if method not in selector.models:
        raise ValueError(f"method must be one of {tuple(selector.models)}, got {method!r}.")

    # The exported text uses the same feature names and ordering used when the decision trees were trained.
    return export_text(selector.models[method], feature_names=list(FEATURE_COLUMNS), decimals=4)

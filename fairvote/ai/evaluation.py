from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from fairvote.ai.features import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    feature_matrix,
)
from fairvote.ai.selector import (
    MODEL_PARAMETERS,
    best_method_indices,
    predict_errors,
    tie_threshold,
    train_selector,
)
from fairvote.study import METHODS

REQUESTED_SPLITS = 5

PREDICTION_COLUMNS: tuple[str, ...] = (
    "evaluation",
    "fold",
    "held_out_epsilon",
    "seed",
    "epsilon",
    "n_respondents",
    "bias",
    *[f"actual_{column}" for column in TARGET_COLUMNS],
    *[f"predicted_{column}" for column in TARGET_COLUMNS],
    "actual_best_method",
    "predicted_best_method",
    "regret",
    "tie_threshold",
    "approximate_tie",
)

PREDICTION_SORT_COLUMNS: tuple[str, ...] = (
    "evaluation",
    "seed",
    "epsilon",
    "n_respondents",
    "bias",
)


def _sort_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    return predictions.sort_values(
        list(PREDICTION_SORT_COLUMNS),
        kind="stable",
    ).reset_index(drop=True)


def _fold_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    evaluation: str,
    fold: int,
    held_out_epsilon: float | None,
) -> pd.DataFrame:
    selector = train_selector(train)
    predicted = predict_errors(selector, feature_matrix(test))
    actual = test.loc[:, list(TARGET_COLUMNS)].to_numpy(dtype=float)

    predicted_index = best_method_indices(predicted)
    actual_index = best_method_indices(actual)
    rows = np.arange(len(test))
    regret = actual[rows, predicted_index] - actual[rows, actual_index]

    ordered = np.sort(predicted, axis=1, kind="stable")
    approximate_tie = (ordered[:, 1] - ordered[:, 0]) <= selector.tie_threshold

    frame = pd.DataFrame(
        {
            "evaluation": evaluation,
            "fold": fold,
            "held_out_epsilon": np.nan if held_out_epsilon is None else float(held_out_epsilon),
            "seed": test["seed"].to_numpy(),
            "epsilon": test["epsilon"].to_numpy(),
            "n_respondents": test["n_respondents"].to_numpy(),
            "bias": test["bias"].to_numpy(),
            "actual_best_method": [METHODS[index] for index in actual_index],
            "predicted_best_method": [METHODS[index] for index in predicted_index],
            "regret": regret,
            "tie_threshold": selector.tie_threshold,
            "approximate_tie": approximate_tie,
        }
    )
    for position, column in enumerate(TARGET_COLUMNS):
        frame[f"actual_{column}"] = actual[:, position]
        frame[f"predicted_{column}"] = predicted[:, position]
    return frame.loc[:, list(PREDICTION_COLUMNS)]


def evaluate_grouped(dataset: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    groups = dataset["seed"].to_numpy()
    n_splits = min(REQUESTED_SPLITS, int(pd.unique(groups).size))
    if n_splits < 2:
        raise ValueError("grouped validation needs at least two distinct seeds.")

    splitter = GroupKFold(n_splits=n_splits)
    frames: list[pd.DataFrame] = []
    for fold, (train_index, test_index) in enumerate(splitter.split(dataset, groups=groups)):
        train = dataset.iloc[train_index]
        test = dataset.iloc[test_index]
        frames.append(_fold_predictions(train, test, "grouped_cv", fold, None))

    predictions = _sort_predictions(pd.concat(frames, ignore_index=True))
    metrics = summarise_predictions(predictions)
    metrics["n_splits"] = int(n_splits)
    metrics["grouping_column"] = "seed"
    return predictions, metrics


def evaluate_leave_one_epsilon_out(dataset: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    epsilons = sorted(float(value) for value in pd.unique(dataset["epsilon"]))
    if len(epsilons) < 2:
        raise ValueError("leave-one-epsilon-out validation needs at least two epsilon values.")

    frames: list[pd.DataFrame] = []
    per_epsilon: dict[str, object] = {}
    for fold, epsilon in enumerate(epsilons):
        held_out = np.isclose(dataset["epsilon"].to_numpy(dtype=float), epsilon)
        train = dataset.loc[~held_out]
        test = dataset.loc[held_out]
        frame = _sort_predictions(
            _fold_predictions(
                train,
                test,
                "leave_one_epsilon_out",
                fold,
                epsilon,
            )
        )
        frames.append(frame)
        per_epsilon[f"{epsilon:g}"] = summarise_predictions(frame)

    predictions = _sort_predictions(pd.concat(frames, ignore_index=True))
    metrics = summarise_predictions(predictions)
    metrics["held_out_epsilons"] = [float(value) for value in epsilons]
    metrics["per_epsilon"] = per_epsilon
    return predictions, metrics


def summarise_predictions(predictions: pd.DataFrame) -> dict[str, object]:
    regression: dict[str, object] = {}
    for column in TARGET_COLUMNS:
        errors = predictions[f"predicted_{column}"].to_numpy(dtype=float) - predictions[f"actual_{column}"].to_numpy(
            dtype=float
        )
        regression[column] = {
            "mae": float(np.mean(np.abs(errors))),
            "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        }

    regret = predictions["regret"].to_numpy(dtype=float)
    actual = predictions.loc[:, [f"actual_{column}" for column in TARGET_COLUMNS]].to_numpy(dtype=float)
    selected = actual.min(axis=1) + regret
    correct = predictions["predicted_best_method"].to_numpy() == predictions["actual_best_method"].to_numpy()
    tie = predictions["approximate_tie"].to_numpy(dtype=bool)
    non_tie = ~tie

    return {
        "n_predictions": int(len(predictions)),
        "regression": regression,
        "recommendation": {
            "argmin_accuracy": float(np.mean(correct)),
            "mean_selected_l1": float(np.mean(selected)),
            "mean_regret": float(np.mean(regret)),
            "median_regret": float(np.median(regret)),
            "approximate_tie_rate": float(np.mean(tie)),
            "non_tie_coverage": float(np.mean(non_tie)),
            "non_tie_argmin_accuracy": (float(np.mean(correct[non_tie])) if non_tie.any() else None),
        },
    }


def baseline_metrics(dataset: pd.DataFrame) -> dict[str, object]:
    actual = dataset.loc[:, list(TARGET_COLUMNS)].to_numpy(dtype=float)
    best = actual.min(axis=1)
    baselines: dict[str, object] = {}
    for position, method in enumerate(METHODS):
        selected = actual[:, position]
        regret = selected - best
        baselines[method] = {
            "mean_selected_l1": float(np.mean(selected)),
            "mean_regret": float(np.mean(regret)),
            "median_regret": float(np.median(regret)),
        }
    baselines["oracle"] = {
        "mean_selected_l1": float(np.mean(best)),
        "mean_regret": 0.0,
        "median_regret": 0.0,
    }
    return baselines


def build_metrics(
    dataset: pd.DataFrame,
    grouped: dict[str, object],
    leave_one_epsilon: dict[str, object],
) -> dict[str, object]:
    return {
        "dataset": {
            "n_rows": int(len(dataset)),
            "epsilons": [float(value) for value in sorted(pd.unique(dataset["epsilon"]))],
            "sample_sizes": [int(value) for value in sorted(pd.unique(dataset["n_respondents"]))],
            "bias_levels": [str(value) for value in sorted(pd.unique(dataset["bias"]))],
            "n_seeds": int(pd.unique(dataset["seed"]).size),
        },
        "grouped_cross_validation": grouped,
        "leave_one_epsilon_out": leave_one_epsilon,
        "fixed_baselines": baseline_metrics(dataset),
        "final_model_tie_threshold": float(tie_threshold(dataset)),
        "model_parameters": dict(MODEL_PARAMETERS),
        "feature_columns": list(FEATURE_COLUMNS),
        "target_columns": list(TARGET_COLUMNS),
    }

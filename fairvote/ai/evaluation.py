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

# Grouped cross-validation requests five folds but is capped by the number of distinct repetition seeds available.
REQUESTED_SPLITS = 5

# These columns define the common per-poll prediction output used by both evaluation procedures.
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

# Prediction rows are sorted deterministically by evaluation type and poll-identifying fields before being returned.
PREDICTION_SORT_COLUMNS: tuple[str, ...] = (
    "evaluation",
    "seed",
    "epsilon",
    "n_respondents",
    "bias",
)


def _sort_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    # Stable sorting keeps the ordering deterministic when rows share the same values in the sort columns.
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
    # The selector and its approximate-tie threshold are fitted only from the training partition for this fold.
    selector = train_selector(train)
    predicted = predict_errors(selector, feature_matrix(test))
    actual = test.loc[:, list(TARGET_COLUMNS)].to_numpy(dtype=float)

    # The predicted and actual best methods are the estimators with the smallest predicted and observed L1 errors respectively.
    predicted_index = best_method_indices(predicted)
    actual_index = best_method_indices(actual)
    rows = np.arange(len(test))

    # Regret is the actual L1 error of the selected estimator minus the lowest actual L1 error available for that poll.
    regret = actual[rows, predicted_index] - actual[rows, actual_index]

    # Approximate ties use the gap between the two lowest predicted errors and the threshold learned from this fold's training data.
    ordered = np.sort(predicted, axis=1, kind="stable")
    approximate_tie = (ordered[:, 1] - ordered[:, 0]) <= selector.tie_threshold

    # Grouped cross-validation has no held-out epsilon, so that field is recorded as missing for those rows.
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

    # Actual and predicted L1 errors are stored for all three estimator targets in the shared target order.
    for position, column in enumerate(TARGET_COLUMNS):
        frame[f"actual_{column}"] = actual[:, position]
        frame[f"predicted_{column}"] = predicted[:, position]
    return frame.loc[:, list(PREDICTION_COLUMNS)]


def evaluate_grouped(dataset: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    # Grouping by repetition seed prevents the same seed from appearing in both training and test data within a fold.
    groups = dataset["seed"].to_numpy()
    n_splits = min(REQUESTED_SPLITS, int(pd.unique(groups).size))
    if n_splits < 2:
        raise ValueError("grouped validation needs at least two distinct seeds.")

    splitter = GroupKFold(n_splits=n_splits)
    frames: list[pd.DataFrame] = []

    # Each dataset row is evaluated in the test partition of one grouped fold.
    for fold, (train_index, test_index) in enumerate(splitter.split(dataset, groups=groups)):
        train = dataset.iloc[train_index]
        test = dataset.iloc[test_index]
        frames.append(_fold_predictions(train, test, "grouped_cv", fold, None))

    # Fold outputs are combined before the overall grouped metrics are calculated.
    predictions = _sort_predictions(pd.concat(frames, ignore_index=True))
    metrics = summarise_predictions(predictions)
    metrics["n_splits"] = int(n_splits)
    metrics["grouping_column"] = "seed"
    return predictions, metrics


def evaluate_leave_one_epsilon_out(dataset: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    # This evaluation treats each epsilon value present in the dataset as the held-out value in turn.
    epsilons = sorted(float(value) for value in pd.unique(dataset["epsilon"]))
    if len(epsilons) < 2:
        raise ValueError("leave-one-epsilon-out validation needs at least two epsilon values.")

    frames: list[pd.DataFrame] = []
    per_epsilon: dict[str, object] = {}

    # Each fold trains on all other evaluated epsilon values and tests only on the held-out epsilon.
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

    # The combined metrics summarise performance across the evaluated held-out epsilon folds.
    predictions = _sort_predictions(pd.concat(frames, ignore_index=True))
    metrics = summarise_predictions(predictions)
    metrics["held_out_epsilons"] = [float(value) for value in epsilons]
    metrics["per_epsilon"] = per_epsilon
    return predictions, metrics


def summarise_predictions(predictions: pd.DataFrame) -> dict[str, object]:
    regression: dict[str, object] = {}

    # Regression quality is measured separately for each estimator's predicted L1-error target.
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

    # Adding regret to the row-wise oracle error recovers the actual L1 error of the estimator selected by the recommender.
    selected = actual.min(axis=1) + regret

    # Argmin accuracy compares the deterministic predicted best method with the estimator that actually has the lowest L1 error.
    correct = predictions["predicted_best_method"].to_numpy() == predictions["actual_best_method"].to_numpy()
    tie = predictions["approximate_tie"].to_numpy(dtype=bool)
    non_tie = ~tie

    # Non-tie accuracy is reported only on rows where the approximate-tie rule does not apply.
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

    # The row-wise minimum is the oracle error because it uses the observed errors of all three estimators for each poll.
    best = actual.min(axis=1)
    baselines: dict[str, object] = {}

    # Each fixed baseline represents always selecting the same estimator for every poll in the dataset.
    for position, method in enumerate(METHODS):
        selected = actual[:, position]
        regret = selected - best
        baselines[method] = {
            "mean_selected_l1": float(np.mean(selected)),
            "mean_regret": float(np.mean(regret)),
            "median_regret": float(np.median(regret)),
        }

    # The oracle selects the lowest observed estimator error per poll and therefore has zero regret by definition.
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
    # This final structure combines dataset metadata, both evaluation procedures, fixed baselines and model configuration.
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
        # This threshold is calculated from the full dataset for the final fitted model rather than for an evaluation fold.
        "final_model_tie_threshold": float(tie_threshold(dataset)),
        "model_parameters": dict(MODEL_PARAMETERS),
        "feature_columns": list(FEATURE_COLUMNS),
        "target_columns": list(TARGET_COLUMNS),
    }

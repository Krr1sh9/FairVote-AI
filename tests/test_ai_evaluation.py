from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.train_selector import main
from fairvote.ai import evaluation
from fairvote.ai.evaluation import (
    PREDICTION_COLUMNS,
    baseline_metrics,
    build_metrics,
    evaluate_grouped,
    evaluate_leave_one_epsilon_out,
)
from fairvote.ai.features import FEATURE_COLUMNS, TARGET_COLUMNS, build_selector_dataset
from fairvote.study import METHODS

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FULL_AI_METRICS_PATH = REPOSITORY_ROOT / "results" / "ai" / "selector_metrics.json"
FULL_AI_PREDICTIONS_PATH = REPOSITORY_ROOT / "results" / "ai" / "selector_predictions.csv"
FULL_AI_RESULTS_AVAILABLE = FULL_AI_METRICS_PATH.is_file() and FULL_AI_PREDICTIONS_PATH.is_file()


@pytest.fixture(scope="module")
def dataset():
    return build_selector_dataset(quick=True)


@pytest.fixture(scope="module")
def grouped(dataset):
    return evaluate_grouped(dataset)


def test_grouped_cross_validation_keeps_seed_groups_out_of_training(
    dataset,
    monkeypatch,
) -> None:
    observed: list[tuple[set[int], set[int]]] = []
    original = evaluation._fold_predictions

    def capture(train, test, evaluation_name, fold, held_out_epsilon):
        observed.append((set(train["seed"]), set(test["seed"])))
        return original(
            train,
            test,
            evaluation_name,
            fold,
            held_out_epsilon,
        )

    monkeypatch.setattr(evaluation, "_fold_predictions", capture)
    predictions, metrics = evaluation.evaluate_grouped(dataset)

    assert metrics["grouping_column"] == "seed"
    assert metrics["n_splits"] >= 2
    assert len(predictions) == len(dataset)
    assert list(predictions.columns) == list(PREDICTION_COLUMNS)
    assert set(predictions["evaluation"]) == {"grouped_cv"}
    assert all(train.isdisjoint(test) for train, test in observed)


def test_leave_one_epsilon_out_excludes_the_held_out_budget(
    dataset,
    monkeypatch,
) -> None:
    observed: list[tuple[set[float], set[float], float]] = []
    original = evaluation._fold_predictions

    def capture(train, test, evaluation_name, fold, held_out_epsilon):
        observed.append(
            (
                set(train["epsilon"].astype(float)),
                set(test["epsilon"].astype(float)),
                float(held_out_epsilon),
            )
        )
        return original(
            train,
            test,
            evaluation_name,
            fold,
            held_out_epsilon,
        )

    monkeypatch.setattr(evaluation, "_fold_predictions", capture)
    predictions, metrics = evaluation.evaluate_leave_one_epsilon_out(dataset)

    assert len(predictions) == len(dataset)
    assert set(predictions["evaluation"]) == {"leave_one_epsilon_out"}
    assert metrics["held_out_epsilons"] == sorted(metrics["held_out_epsilons"])

    for train_epsilons, test_epsilons, held_out in observed:
        assert held_out not in train_epsilons
        assert test_epsilons == {held_out}


def test_recommendation_regret_and_actual_best_method_are_correct(grouped) -> None:
    predictions, _ = grouped
    actual = predictions.loc[:, [f"actual_{column}" for column in TARGET_COLUMNS]].to_numpy(dtype=float)
    actual_best_index = np.argsort(actual, axis=1, kind="stable")[:, 0]
    expected_best = [METHODS[index] for index in actual_best_index]
    method_index = {method: index for index, method in enumerate(METHODS)}
    selected_index = np.array([method_index[method] for method in predictions["predicted_best_method"]])
    rows = np.arange(len(predictions))
    expected_regret = actual[rows, selected_index] - actual[rows, actual_best_index]

    assert predictions["actual_best_method"].tolist() == expected_best
    np.testing.assert_allclose(
        predictions["regret"],
        expected_regret,
        atol=1e-12,
    )
    assert predictions["regret"].min() >= -1e-12


def test_fixed_baselines_include_all_estimators_and_an_oracle(dataset) -> None:
    baselines = baseline_metrics(dataset)

    assert set(baselines) == {*METHODS, "oracle"}
    assert baselines["oracle"]["mean_regret"] == 0.0
    assert baselines["oracle"]["median_regret"] == 0.0

    for method in METHODS:
        assert baselines[method]["mean_regret"] >= 0.0
        assert baselines[method]["mean_selected_l1"] >= baselines["oracle"]["mean_selected_l1"]


def test_metrics_record_the_evaluation_design_and_model_configuration(
    dataset,
) -> None:
    _, grouped_metrics = evaluate_grouped(dataset)
    _, held_out_metrics = evaluate_leave_one_epsilon_out(dataset)
    metrics = build_metrics(dataset, grouped_metrics, held_out_metrics)

    assert metrics["dataset"]["n_rows"] == len(dataset)
    assert metrics["feature_columns"] == list(FEATURE_COLUMNS)
    assert metrics["target_columns"] == list(TARGET_COLUMNS)
    assert metrics["model_parameters"] == {
        "max_depth": 4,
        "min_samples_leaf": 20,
        "random_state": 42,
    }
    assert "grouped_cross_validation" in metrics
    assert "leave_one_epsilon_out" in metrics
    assert "fixed_baselines" in metrics
    json.dumps(metrics)


def test_quick_selector_command_writes_dataset_predictions_metrics_and_plots(
    tmp_path: Path,
) -> None:
    assert main(["--quick", "--output-dir", str(tmp_path)]) == 0

    dataset_path = tmp_path / "ai" / "selector_dataset.csv"
    predictions_path = tmp_path / "ai" / "selector_predictions.csv"
    metrics_path = tmp_path / "ai" / "selector_metrics.json"
    expected_plots = {
        "ai_selector_baselines.png",
        "ai_selector_tree_raw_frequencies.png",
        "ai_selector_tree_rr_debiased.png",
        "ai_selector_tree_poststratified.png",
    }

    assert len(pd.read_csv(dataset_path)) == 36

    predictions = pd.read_csv(predictions_path)
    assert len(predictions) == 72
    assert set(predictions["evaluation"]) == {
        "grouped_cv",
        "leave_one_epsilon_out",
    }

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["dataset"]["n_rows"] == 36
    assert metrics["feature_columns"] == list(FEATURE_COLUMNS)
    assert expected_plots <= {path.name for path in (tmp_path / "plots").glob("*.png")}


@pytest.mark.skipif(
    not FULL_AI_RESULTS_AVAILABLE,
    reason="Full generated AI evaluation results are not present in this checkout.",
)
def test_generated_full_ai_evaluation_matches_the_full_study_shape() -> None:
    metrics = json.loads(FULL_AI_METRICS_PATH.read_text(encoding="utf-8"))
    predictions = pd.read_csv(FULL_AI_PREDICTIONS_PATH)

    assert metrics["dataset"]["n_rows"] == 1440
    assert metrics["feature_columns"] == list(FEATURE_COLUMNS)
    assert metrics["model_parameters"] == {
        "max_depth": 4,
        "min_samples_leaf": 20,
        "random_state": 42,
    }
    assert len(predictions) == 2880
    assert set(predictions["evaluation"]) == {
        "grouped_cv",
        "leave_one_epsilon_out",
    }
    assert np.all(np.isfinite(predictions["regret"].to_numpy(dtype=float)))
    assert predictions["regret"].min() >= -1e-12

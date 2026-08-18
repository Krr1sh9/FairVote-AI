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

# The repository root is used to locate the generated full-run AI metrics and prediction files when they are available.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FULL_AI_METRICS_PATH = REPOSITORY_ROOT / "results" / "ai" / "selector_metrics.json"
FULL_AI_PREDICTIONS_PATH = REPOSITORY_ROOT / "results" / "ai" / "selector_predictions.csv"

# The full-output integration test is enabled only when both generated AI evaluation files are present.
FULL_AI_RESULTS_AVAILABLE = FULL_AI_METRICS_PATH.is_file() and FULL_AI_PREDICTIONS_PATH.is_file()


@pytest.fixture(scope="module")
def dataset():
    # The deterministic quick selector dataset is built once and reused across the evaluation tests in this module.
    return build_selector_dataset(quick=True)


@pytest.fixture(scope="module")
def grouped(dataset):
    # Grouped cross-validation predictions and metrics are computed once for tests that inspect their derived recommendation quantities.
    return evaluate_grouped(dataset)


def test_grouped_cross_validation_keeps_seed_groups_out_of_training(
    dataset,
    monkeypatch,
) -> None:
    # The fold helper is wrapped to record the training and test seed sets while delegating prediction generation to the original implementation.
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

    # The grouped procedure must identify seed as its grouping column, use at least two folds and produce one prediction row for every dataset row.
    assert metrics["grouping_column"] == "seed"
    assert metrics["n_splits"] >= 2
    assert len(predictions) == len(dataset)

    # Grouped predictions must use the declared schema and evaluation label.
    assert list(predictions.columns) == list(PREDICTION_COLUMNS)
    assert set(predictions["evaluation"]) == {"grouped_cv"}

    # No repetition seed passed to a fold's test subset may also appear in that fold's training subset.
    assert all(train.isdisjoint(test) for train, test in observed)


def test_leave_one_epsilon_out_excludes_the_held_out_budget(
    dataset,
    monkeypatch,
) -> None:
    # The fold helper is wrapped to record the epsilon values supplied to each training and test split.
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

    # Every dataset row must be evaluated once under the leave-one-epsilon-out procedure and carry the corresponding evaluation label.
    assert len(predictions) == len(dataset)
    assert set(predictions["evaluation"]) == {"leave_one_epsilon_out"}

    # The recorded held-out epsilon values in the metrics must be sorted.
    assert metrics["held_out_epsilons"] == sorted(metrics["held_out_epsilons"])

    # For every fold, the held-out epsilon must be absent from training and must be the only epsilon present in the test subset.
    for train_epsilons, test_epsilons, held_out in observed:
        assert held_out not in train_epsilons
        assert test_epsilons == {held_out}


def test_recommendation_regret_and_actual_best_method_are_correct(grouped) -> None:
    # The actual per-estimator L1 errors are reconstructed from the grouped prediction table in the same configured method order.
    predictions, _ = grouped
    actual = predictions.loc[:, [f"actual_{column}" for column in TARGET_COLUMNS]].to_numpy(dtype=float)

    # Stable sorting selects the first method in METHODS when multiple estimators share the same smallest actual L1 error.
    actual_best_index = np.argsort(actual, axis=1, kind="stable")[:, 0]
    expected_best = [METHODS[index] for index in actual_best_index]

    # The recommended method names are converted back to column indices so their observed L1 errors can be compared with the poll-level oracle minimum.
    method_index = {method: index for index, method in enumerate(METHODS)}
    selected_index = np.array([method_index[method] for method in predictions["predicted_best_method"]])
    rows = np.arange(len(predictions))
    expected_regret = actual[rows, selected_index] - actual[rows, actual_best_index]

    # Stored actual-best labels and regret values must match these independently reconstructed quantities.
    assert predictions["actual_best_method"].tolist() == expected_best
    np.testing.assert_allclose(
        predictions["regret"],
        expected_regret,
        atol=1e-12,
    )

    # Regret should not be materially negative, with the small tolerance allowing only floating-point round-off.
    assert predictions["regret"].min() >= -1e-12


def test_fixed_baselines_include_all_estimators_and_an_oracle(dataset) -> None:
    # Baseline metrics compare always choosing each fixed estimator with the poll-level oracle that chooses the lowest observed L1 error.
    baselines = baseline_metrics(dataset)

    # The baseline result must contain all configured estimators plus the oracle reference.
    assert set(baselines) == {*METHODS, "oracle"}

    # By construction, selecting the observed best estimator for each poll gives the oracle zero recommendation regret.
    assert baselines["oracle"]["mean_regret"] == 0.0
    assert baselines["oracle"]["median_regret"] == 0.0

    # A fixed estimator cannot have negative mean regret or a lower mean selected L1 error than the poll-level oracle on the same dataset.
    for method in METHODS:
        assert baselines[method]["mean_regret"] >= 0.0
        assert baselines[method]["mean_selected_l1"] >= baselines["oracle"]["mean_selected_l1"]


def test_metrics_record_the_evaluation_design_and_model_configuration(
    dataset,
) -> None:
    # Both evaluation procedures are run so their summaries can be assembled into the final metrics structure.
    _, grouped_metrics = evaluate_grouped(dataset)
    _, held_out_metrics = evaluate_leave_one_epsilon_out(dataset)
    metrics = build_metrics(dataset, grouped_metrics, held_out_metrics)

    # The metrics must record dataset size, model inputs, targets and the fixed decision-tree configuration.
    assert metrics["dataset"]["n_rows"] == len(dataset)
    assert metrics["feature_columns"] == list(FEATURE_COLUMNS)
    assert metrics["target_columns"] == list(TARGET_COLUMNS)
    assert metrics["model_parameters"] == {
        "max_depth": 4,
        "min_samples_leaf": 20,
        "random_state": 42,
    }

    # The final structure must include both evaluation summaries and the fixed-baseline comparison.
    assert "grouped_cross_validation" in metrics
    assert "leave_one_epsilon_out" in metrics
    assert "fixed_baselines" in metrics

    # Serialising the structure verifies that the assembled metrics contain JSON-compatible values.
    json.dumps(metrics)


def test_quick_selector_command_writes_dataset_predictions_metrics_and_plots(
    tmp_path: Path,
) -> None:
    # The complete quick selector pipeline is run into an isolated temporary output directory.
    assert main(["--quick", "--output-dir", str(tmp_path)]) == 0

    dataset_path = tmp_path / "ai" / "selector_dataset.csv"
    predictions_path = tmp_path / "ai" / "selector_predictions.csv"
    metrics_path = tmp_path / "ai" / "selector_metrics.json"

    # The selector pipeline is expected to generate one baseline plot and one tree visualisation for each estimator.
    expected_plots = {
        "ai_selector_baselines.png",
        "ai_selector_tree_raw_frequencies.png",
        "ai_selector_tree_rr_debiased.png",
        "ai_selector_tree_poststratified.png",
    }

    # Quick mode contains 36 selector-dataset rows.
    assert len(pd.read_csv(dataset_path)) == 36

    # The prediction file combines 36 grouped-CV rows with 36 leave-one-epsilon-out rows.
    predictions = pd.read_csv(predictions_path)
    assert len(predictions) == 72
    assert set(predictions["evaluation"]) == {
        "grouped_cv",
        "leave_one_epsilon_out",
    }

    # The generated metrics must describe the quick dataset and six model features, and all four expected AI plots must exist.
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["dataset"]["n_rows"] == 36
    assert metrics["feature_columns"] == list(FEATURE_COLUMNS)
    assert expected_plots <= {path.name for path in (tmp_path / "plots").glob("*.png")}


@pytest.mark.skipif(
    not FULL_AI_RESULTS_AVAILABLE,
    reason="Full generated AI evaluation results are not present in this checkout.",
)
def test_generated_full_ai_evaluation_matches_the_full_study_shape() -> None:
    # This generated-output check runs only when both full AI evaluation artefacts are available.
    metrics = json.loads(FULL_AI_METRICS_PATH.read_text(encoding="utf-8"))
    predictions = pd.read_csv(FULL_AI_PREDICTIONS_PATH)

    # The full AI metrics document records 1,440 dataset rows together with the expected feature schema and tree configuration.
    assert metrics["dataset"]["n_rows"] == 1440
    assert metrics["feature_columns"] == list(FEATURE_COLUMNS)
    assert metrics["model_parameters"] == {
        "max_depth": 4,
        "min_samples_leaf": 20,
        "random_state": 42,
    }

    # The full prediction table contains one 1,440-row block from each of the two evaluation procedures.
    assert len(predictions) == 2880
    assert set(predictions["evaluation"]) == {
        "grouped_cv",
        "leave_one_epsilon_out",
    }

    # Every stored regret value must be finite and non-negative up to the same floating-point tolerance used above.
    assert np.all(np.isfinite(predictions["regret"].to_numpy(dtype=float)))
    assert predictions["regret"].min() >= -1e-12

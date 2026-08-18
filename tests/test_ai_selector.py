from __future__ import annotations

import numpy as np
import pytest
from sklearn.tree import DecisionTreeRegressor

from fairvote.ai.features import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    build_selector_dataset,
    feature_matrix,
)
from fairvote.ai.selector import (
    GROUPING_COLUMNS,
    MODEL_PARAMETERS,
    best_method_indices,
    export_tree_rules,
    predict_errors,
    recommend,
    recommend_from_predictions,
    tie_threshold,
    train_selector,
)
from fairvote.study import METHODS


@pytest.fixture(scope="module")
def dataset():
    # The quick selector dataset is built once and reused across this module's model tests.
    return build_selector_dataset(quick=True)


@pytest.fixture(scope="module")
def selector(dataset):
    # A fitted selector based on the shared quick dataset is reused by tests that do not need separate training runs.
    return train_selector(dataset)


def test_selector_training_is_configured_and_deterministic(dataset) -> None:
    # Training the selector twice on the same dataset checks deterministic fitting under the fixed model parameters.
    first = train_selector(dataset)
    second = train_selector(dataset)
    features = feature_matrix(dataset)

    # The selector is configured with the stated shallow decision-tree parameters.
    assert MODEL_PARAMETERS == {
        "max_depth": 4,
        "min_samples_leaf": 20,
        "random_state": 42,
    }

    # One fitted regression model must be present for each configured estimator.
    assert set(first.models) == set(METHODS)

    # Every fitted estimator model must be a decision-tree regressor fitted with all six feature columns and the declared parameters.
    for model in first.models.values():
        assert isinstance(model, DecisionTreeRegressor)
        assert model.n_features_in_ == len(FEATURE_COLUMNS)
        params = model.get_params()
        assert params["max_depth"] == 4
        assert params["min_samples_leaf"] == 20
        assert params["random_state"] == 42

    # Predictions from independently fitted selectors are compared on the same feature matrix.
    first_predictions = predict_errors(first, features)
    second_predictions = predict_errors(second, features)

    # The prediction matrix must contain one finite non-negative predicted error per poll and estimator, and both fits must agree exactly here.
    assert first_predictions.shape == (len(dataset), len(METHODS))
    assert np.all(np.isfinite(first_predictions))
    assert first_predictions.min() >= 0.0
    np.testing.assert_array_equal(first_predictions, second_predictions)


def test_recommendation_returns_predicted_errors_for_one_poll(selector, dataset) -> None:
    # The first poll's six-feature vector is passed through the fitted selector's single-poll recommendation path.
    features = feature_matrix(dataset)
    prediction = recommend(selector, features[0])

    # The recommendation must name one configured estimator and expose one finite predicted error for every estimator.
    assert prediction.best_method in METHODS
    assert set(prediction.predicted_errors) == set(METHODS)
    assert all(np.isfinite(value) for value in prediction.predicted_errors.values())


def test_tie_handling_is_deterministic() -> None:
    # An exact tie at the minimum predicted error is resolved by the fixed estimator ordering and is also reported as an approximate tie.
    exact = recommend_from_predictions(np.array([0.5, 0.5, 0.9]), 0.0)
    assert exact.best_method == "raw_frequencies"
    assert exact.approximate_tie
    assert exact.tied_methods == ("raw_frequencies", "rr_debiased")

    # Two lowest predicted errors within the supplied threshold are reported as an approximate tie while retaining the lower-error method as best.
    near = recommend_from_predictions(np.array([0.30, 0.32, 0.90]), 0.05)
    assert near.best_method == "raw_frequencies"
    assert near.approximate_tie
    assert near.tied_methods == ("raw_frequencies", "rr_debiased")

    # A gap larger than the supplied threshold leaves the recommendation as a clear single best method.
    clear = recommend_from_predictions(np.array([0.30, 0.50, 0.90]), 0.05)
    assert clear.best_method == "raw_frequencies"
    assert not clear.approximate_tie
    assert clear.tied_methods == ()

    # Row-wise best-method indices use the same stable method order when predicted errors are equal.
    predicted = np.array(
        [
            [0.3, 0.3, 0.3],
            [0.7, 0.2, 0.2],
            [0.4, 0.9, 0.1],
        ]
    )
    np.testing.assert_array_equal(
        best_method_indices(predicted),
        np.array([0, 1, 2]),
    )


def test_tie_threshold_matches_the_stated_empirical_formula(dataset) -> None:
    # The expected threshold is rebuilt independently from within-group sample standard deviations of all three L1 target columns.
    deviations: list[float] = []

    # Groups use the selector's declared grouping columns, and non-finite standard deviations are excluded from the pooled values.
    for _, group in dataset.groupby(list(GROUPING_COLUMNS), sort=True):
        for column in TARGET_COLUMNS:
            value = float(group[column].std(ddof=1))
            if np.isfinite(value):
                deviations.append(value)

    # The selector tie threshold must equal the median of the collected finite within-group standard deviations.
    expected = float(np.median(deviations))
    assert tie_threshold(dataset) == pytest.approx(expected)


def test_exported_tree_rules_are_readable_and_use_only_model_features(selector) -> None:
    # The poststratified model is exported as a textual decision-tree representation for inspection.
    rules = export_tree_rules(selector, "poststratified")

    # The exported representation must be a non-empty string.
    assert isinstance(rules, str)
    assert rules.strip()

    # Audit fields and target names must not appear in the exported rules because they are not model features.
    for forbidden in ("fallback_cells", "bias", "seed", *TARGET_COLUMNS):
        assert forbidden not in rules

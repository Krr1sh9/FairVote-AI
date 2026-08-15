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
    return build_selector_dataset(quick=True)


@pytest.fixture(scope="module")
def selector(dataset):
    return train_selector(dataset)


def test_selector_training_is_configured_and_deterministic(dataset) -> None:
    first = train_selector(dataset)
    second = train_selector(dataset)
    features = feature_matrix(dataset)

    assert MODEL_PARAMETERS == {
        "max_depth": 4,
        "min_samples_leaf": 20,
        "random_state": 42,
    }
    assert set(first.models) == set(METHODS)

    for model in first.models.values():
        assert isinstance(model, DecisionTreeRegressor)
        assert model.n_features_in_ == len(FEATURE_COLUMNS)
        params = model.get_params()
        assert params["max_depth"] == 4
        assert params["min_samples_leaf"] == 20
        assert params["random_state"] == 42

    first_predictions = predict_errors(first, features)
    second_predictions = predict_errors(second, features)

    assert first_predictions.shape == (len(dataset), len(METHODS))
    assert np.all(np.isfinite(first_predictions))
    assert first_predictions.min() >= 0.0
    np.testing.assert_array_equal(first_predictions, second_predictions)


def test_recommendation_returns_predicted_errors_for_one_poll(selector, dataset) -> None:
    features = feature_matrix(dataset)
    prediction = recommend(selector, features[0])

    assert prediction.best_method in METHODS
    assert set(prediction.predicted_errors) == set(METHODS)
    assert all(np.isfinite(value) for value in prediction.predicted_errors.values())


def test_tie_handling_is_deterministic() -> None:
    exact = recommend_from_predictions(np.array([0.5, 0.5, 0.9]), 0.0)
    assert exact.best_method == "raw_frequencies"
    assert exact.approximate_tie
    assert exact.tied_methods == ("raw_frequencies", "rr_debiased")

    near = recommend_from_predictions(np.array([0.30, 0.32, 0.90]), 0.05)
    assert near.best_method == "raw_frequencies"
    assert near.approximate_tie
    assert near.tied_methods == ("raw_frequencies", "rr_debiased")

    clear = recommend_from_predictions(np.array([0.30, 0.50, 0.90]), 0.05)
    assert clear.best_method == "raw_frequencies"
    assert not clear.approximate_tie
    assert clear.tied_methods == ()

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
    deviations: list[float] = []

    for _, group in dataset.groupby(list(GROUPING_COLUMNS), sort=True):
        for column in TARGET_COLUMNS:
            value = float(group[column].std(ddof=1))
            if np.isfinite(value):
                deviations.append(value)

    expected = float(np.median(deviations))
    assert tie_threshold(dataset) == pytest.approx(expected)


def test_exported_tree_rules_are_readable_and_use_only_model_features(selector) -> None:
    rules = export_tree_rules(selector, "poststratified")

    assert isinstance(rules, str)
    assert rules.strip()

    for forbidden in ("fallback_cells", "bias", "seed", *TARGET_COLUMNS):
        assert forbidden not in rules

from __future__ import annotations

import json

import numpy as np
import pytest

# These canonical MRP tests deliberately exercise binary (k=2) Randomized
# Response fixtures. The RR implementation correctly warns that k=2 has
# limited category entropy, but that warning is expected here and should not
# make a clean test run look noisy.
pytestmark = pytest.mark.filterwarnings(
    r"ignore:k=2 is very small\. Binary RR is valid but has limited category entropy\.:UserWarning"
)

from fairvote.inference.mrp import (
    DesignMatrix,
    LinearRRMRPModel,
    RRMultinomialModel,
    build_design_matrix,
    poststratify,
)
from fairvote.inference.mrp.poststratify import normalise_poststrat_weights
from fairvote.privacy.mechanisms.kary_rr import privatize_many


def test_design_matrix_consistency_integer_coded_features():
    features = {
        "region": np.array([0, 1, 0, 1]),
        "age": np.array([0, 0, 1, 1]),
    }
    levels = {"region": ["North", "South"], "age": ["Young", "Old"]}

    X1, info1 = build_design_matrix(features, levels, feature_order=["region", "age"])
    X2, info2 = build_design_matrix(features, levels, feature_order=["region", "age"])

    assert np.array_equal(X1, X2)
    assert info1.n_cols == info2.n_cols == 5
    assert info1.feature_names == ["region", "age"]
    assert X1.shape == (4, 5)
    assert np.all(X1[:, 0] == 1.0)


def test_string_design_matrix_rejects_missing_category_columns():
    rows = [{"region": "North", "age": "18-29"}, {"region": "South"}]
    with pytest.raises(ValueError, match="missing feature columns"):
        DesignMatrix(["region", "age"]).fit(rows)


def test_poststratification_rejects_unseen_population_category_codes():
    features = {"region": np.array([0, 1]), "age": np.array([0, 1])}
    levels = {"region": ["North", "South"], "age": ["Young", "Old"]}
    X, info = build_design_matrix(features, levels, feature_order=["region", "age"])
    y = np.array([0, 1])
    model = LinearRRMRPModel(2, epsilon=1.0, seed=4)
    model.fit(X, y, steps=5, batch_size=2)

    cells = np.array([[0, 0], [2, 1]])
    counts = np.array([10, 20])
    with pytest.raises(ValueError, match="outside"):
        poststratify(model, cells=cells, counts=counts, by=["region", "age"], design_info=info)


def test_poststratification_weights_sum_to_one():
    weights = normalise_poststrat_weights(np.array([10, 30, 60]), expected_n=3)
    assert np.isclose(float(weights.sum()), 1.0)
    assert np.all(weights >= 0.0)


def test_rr_aware_likelihood_uses_reported_labels_not_latent_labels():
    X = np.column_stack([np.ones(80), np.r_[np.zeros(40), np.ones(40)]])
    true_labels = np.r_[np.zeros(40, dtype=int), np.ones(40, dtype=int)]
    reported_labels = 1 - true_labels

    model_true = LinearRRMRPModel(2, epsilon=3.0, seed=123, l2=0.0)
    model_reported = LinearRRMRPModel(2, epsilon=3.0, seed=123, l2=0.0)
    model_true.fit(X, true_labels, steps=120, batch_size=32, lr=0.05)
    model_reported.fit(X, reported_labels, steps=120, batch_size=32, lr=0.05)

    pred_true = model_true.predict_theta(X)
    pred_reported = model_reported.predict_theta(X)
    assert not np.allclose(pred_true, pred_reported, atol=1e-3)
    assert pred_reported[:40, 1].mean() > pred_reported[:40, 0].mean()


def test_canonical_mrp_path_stable_outputs_on_fixture_data():
    rng = np.random.default_rng(9)
    features = {"region": np.repeat([0, 1], 100)}
    levels = {"region": ["North", "South"]}
    X, _info = build_design_matrix(features, levels, feature_order=["region"])
    true = np.repeat([0, 1], 100)
    reported = privatize_many(true, epsilon=4.0, k=2, rng=rng)

    model_a = RRMultinomialModel(2, epsilon=4.0, seed=77, l2=0.01)
    model_b = LinearRRMRPModel(2, epsilon=4.0, seed=77, l2=0.01)
    info_a = model_a.fit(X, reported, steps=80, batch_size=64, lr=0.05, keep_history=True, history_every=10)
    info_b = model_b.fit(X, reported, steps=80, batch_size=64, lr=0.05, keep_history=True, history_every=10)

    assert info_a.steps == info_b.steps == 80
    assert info_a.final_loss == pytest.approx(info_b.final_loss)
    assert info_a.runtime_sec >= 0.0
    assert info_a.history is not None and len(info_a.history) >= 2
    assert np.allclose(model_a.predict_theta(X), model_b.predict_theta(X))
    p = model_a.poststratify(X, np.ones(X.shape[0]))
    assert p.shape == (2,)
    assert np.isclose(float(p.sum()), 1.0)


def test_fitted_model_metadata_export_is_json_serialisable(tmp_path):
    X = np.column_stack([np.ones(6), [0, 0, 0, 1, 1, 1]])
    y = np.array([0, 0, 1, 1, 1, 0])
    model = LinearRRMRPModel(2, epsilon=1.0, seed=5)
    model.fit(X, y, steps=5, batch_size=3, keep_history=True)

    metadata = model.export_metadata()
    assert metadata["model_name"] == "linear_rr_mrp_regularized_multinomial"
    assert "not full Bayesian" in metadata["honest_description"]
    assert metadata["fit_diagnostics"]["steps"] == 5

    path = tmp_path / "metadata.json"
    model.save_metadata(path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["coefficient_shape"] == [2, 2]

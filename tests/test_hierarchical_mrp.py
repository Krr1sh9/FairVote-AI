from __future__ import annotations

import numpy as np

from fairvote.inference.mrp import HierarchicalRRMRPModel
from fairvote.privacy.mechanisms.kary_rr import privatize_many


def test_hierarchical_rr_mrp_fits_and_poststratifies_sparse_levels():
    rng = np.random.default_rng(2026)
    n = 240
    features = {
        "region": np.r_[np.zeros(110, dtype=int), np.ones(110, dtype=int), np.full(20, 2, dtype=int)],
        "age_group": np.tile(np.array([0, 1, 2], dtype=int), 80),
    }
    levels = {"region": ["A", "B", "Sparse"], "age_group": ["Young", "Mid", "Old"]}
    true = np.where(features["region"] == 0, 0, np.where(features["region"] == 1, 1, 2)).astype(int)
    reported = privatize_many(true, epsilon=4.0, k=3, rng=rng)

    model = HierarchicalRRMRPModel(3, epsilon=4.0, seed=7, effect_l2=0.2)
    info = model.fit(
        features, reported, levels, feature_order=["region", "age_group"], steps=80, batch_size=64, lr=0.05
    )

    assert info.steps == 80
    theta = model.predict_theta_from_features(features)
    assert theta.shape == (n, 3)
    assert np.allclose(theta.sum(axis=1), 1.0)
    assert model.feature_info["region"].observed_counts == [110, 110, 20]

    post = model.poststratify(features, np.ones(n))
    assert post.shape == (3,)
    assert np.isclose(post.sum(), 1.0)
    meta = model.export_metadata()
    assert meta["model_name"] == "hierarchical_rr_mrp_partial_pooling_map"
    assert meta["feature_info"]["region"]["n_levels"] == 3

from __future__ import annotations

import numpy as np
import pytest


def test_group_metrics_known_values():
    gm = pytest.importorskip("fairvote.metrics.group_metrics")
    worst_group_l1 = gm.worst_group_l1
    weighted_group_l1 = gm.weighted_group_l1
    p90_group_l1 = gm.p90_group_l1

    # Two groups with simple distributions
    est = {
        "A": np.array([0.6, 0.4]),
        "B": np.array([0.2, 0.8]),
    }
    truth = {
        "A": np.array([0.5, 0.5]),   # L1 = 0.2
        "B": np.array([0.0, 1.0]),   # L1 = 0.4
    }
    masses = {"A": 0.7, "B": 0.3}

    w = worst_group_l1(est, truth, group_masses=masses, min_mass=0.0)
    assert abs(w - 0.4) < 1e-9

    wg = weighted_group_l1(est, truth, group_masses=masses)
    # weighted = 0.7*0.2 + 0.3*0.4 = 0.26
    assert abs(wg - 0.26) < 1e-9

    p90 = p90_group_l1(est, truth, group_masses=masses, min_mass=0.0)
    # with only 2 groups, p90 should be the larger error (or very close)
    assert abs(p90 - 0.4) < 1e-9

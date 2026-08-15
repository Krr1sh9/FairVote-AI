from __future__ import annotations

import numpy as np
import pytest

from fairvote.metrics import l1_error, max_absolute_error


def test_error_metrics_match_hand_calculated_values() -> None:
    estimate = np.array([0.5, 0.3, 0.2])
    truth = np.array([0.4, 0.4, 0.2])

    assert l1_error(estimate, truth) == pytest.approx(0.2)
    assert max_absolute_error(estimate, truth) == pytest.approx(0.1)
    assert l1_error(truth, truth) == 0.0
    assert max_absolute_error(truth, truth) == 0.0

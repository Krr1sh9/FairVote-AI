from __future__ import annotations

import numpy as np
import pytest

from fairvote.metrics import l1_error, max_absolute_error


def test_error_metrics_match_hand_calculated_values() -> None:
    # This deterministic example compares one estimated distribution with a known three-category truth.
    estimate = np.array([0.5, 0.3, 0.2])
    truth = np.array([0.4, 0.4, 0.2])

    # The absolute category-wise differences are 0.1, 0.1 and 0.0, so their L1 sum is 0.2.
    assert l1_error(estimate, truth) == pytest.approx(0.2)

    # The largest absolute category-wise difference in the same example is 0.1.
    assert max_absolute_error(estimate, truth) == pytest.approx(0.1)

    # Both error measures must be zero when the estimate is exactly equal to the truth.
    assert l1_error(truth, truth) == 0.0
    assert max_absolute_error(truth, truth) == 0.0

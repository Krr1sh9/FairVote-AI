from __future__ import annotations

import numpy as np
import pytest

from fairvote.metrics import l1_error, max_absolute_error


def test_known_example() -> None:
    estimate = np.array([0.5, 0.3, 0.2])
    truth = np.array([0.4, 0.4, 0.2])
    assert l1_error(estimate, truth) == pytest.approx(0.2)
    assert max_absolute_error(estimate, truth) == pytest.approx(0.1)


def test_identical_vectors_have_zero_error() -> None:
    vector = [0.25, 0.25, 0.5]
    assert l1_error(vector, vector) == 0.0
    assert max_absolute_error(vector, vector) == 0.0


def test_metrics_accept_plain_sequences() -> None:
    assert l1_error([1.0, 0.0], [0.0, 1.0]) == pytest.approx(2.0)


@pytest.mark.parametrize("metric", [l1_error, max_absolute_error])
def test_mismatched_shapes_are_rejected(metric) -> None:
    with pytest.raises(ValueError):
        metric([0.5, 0.5], [0.3, 0.3, 0.4])


@pytest.mark.parametrize("metric", [l1_error, max_absolute_error])
def test_non_finite_values_are_rejected(metric) -> None:
    with pytest.raises(ValueError):
        metric([0.5, np.nan], [0.5, 0.5])


@pytest.mark.parametrize("metric", [l1_error, max_absolute_error])
def test_empty_and_multidimensional_inputs_are_rejected(metric) -> None:
    with pytest.raises(ValueError):
        metric([], [])
    with pytest.raises(ValueError):
        metric([[0.5, 0.5]], [[0.5, 0.5]])

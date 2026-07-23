from __future__ import annotations

import numpy as np
import pytest

from fairvote.privacy.mechanisms.kary_rr import rr_transition_matrix
from fairvote.simulation.population import default_population, make_population
from fairvote.simulation.sampling import (
    BIAS_LEVELS,
    run_synthetic_poll,
    sampling_probabilities,
)


def test_population_weights_and_preferences_sum_to_one() -> None:
    population = default_population()
    assert population.n_cells == 6
    assert population.n_categories == 3
    assert population.weights.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(population.preferences.sum(axis=1), np.ones(population.n_cells))
    assert population.true_distribution().sum() == pytest.approx(1.0)


def test_population_cells_are_two_regions_by_three_age_groups() -> None:
    population = default_population()
    assert set(population.regions) == {"North", "South"}
    assert set(population.age_groups) == {"18-34", "35-54", "55+"}
    assert len(set(population.cell_labels)) == 6


def test_true_distribution_is_hand_calculable() -> None:
    population = make_population(
        regions=("North", "South"),
        age_groups=("18-34", "55+"),
        weights=(0.25, 0.75),
        preferences=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    )
    np.testing.assert_allclose(population.true_distribution(), [0.25, 0.75, 0.0])


def test_make_population_normalises_input() -> None:
    population = make_population(
        regions=("North", "South"),
        age_groups=("18-34", "55+"),
        weights=(1.0, 3.0),
        preferences=((2.0, 1.0, 1.0), (1.0, 1.0, 2.0)),
    )
    np.testing.assert_allclose(population.weights, [0.25, 0.75])
    np.testing.assert_allclose(population.preferences[0], [0.5, 0.25, 0.25])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"weights": (-0.5, 1.5)},
        {"weights": (0.0, 0.0)},
        {"weights": (0.5, 0.5, 0.0)},
        {"preferences": ((0.5, 0.5, 0.0), (0.0, 0.0, 0.0))},
        {"preferences": ((0.5, 0.5), (0.5, 0.5))},
        {"preferences": ((np.nan, 0.5, 0.5), (0.5, 0.25, 0.25))},
    ],
)
def test_invalid_population_specifications_are_rejected(kwargs: dict) -> None:
    base = {
        "regions": ("North", "South"),
        "age_groups": ("18-34", "55+"),
        "weights": (0.5, 0.5),
        "preferences": ((0.5, 0.25, 0.25), (0.25, 0.5, 0.25)),
    }
    with pytest.raises(ValueError):
        make_population(**{**base, **kwargs})


def test_representative_sampling_uses_population_weights() -> None:
    population = default_population()
    np.testing.assert_allclose(sampling_probabilities(population, "none"), population.weights)


@pytest.mark.parametrize("bias", BIAS_LEVELS)
def test_sampling_probabilities_are_valid(bias: str) -> None:
    probabilities = sampling_probabilities(default_population(), bias)
    assert probabilities.sum() == pytest.approx(1.0)
    assert probabilities.min() > 0.0


def test_bias_shifts_sampling_towards_younger_cells() -> None:
    population = default_population()
    young = np.array([age == "18-34" for age in population.age_groups])
    oldest = np.array([age == "55+" for age in population.age_groups])
    shares = {bias: sampling_probabilities(population, bias) for bias in BIAS_LEVELS}

    assert float(shares["none"][young].sum()) < float(shares["moderate"][young].sum())
    assert float(shares["moderate"][young].sum()) < float(shares["strong"][young].sum())
    assert float(shares["none"][oldest].sum()) > float(shares["moderate"][oldest].sum())
    assert float(shares["moderate"][oldest].sum()) > float(shares["strong"][oldest].sum())


def test_representative_sampling_approximates_weights_at_large_n() -> None:
    population = default_population()
    sample = run_synthetic_poll(population, 100_000, 1.0, "none", np.random.default_rng(7))
    observed = np.bincount(sample.cell_indices, minlength=population.n_cells) / len(sample)
    np.testing.assert_allclose(observed, population.weights, atol=0.01)


def test_biased_sampling_approximates_its_probabilities_at_large_n() -> None:
    population = default_population()
    sample = run_synthetic_poll(population, 100_000, 1.0, "strong", np.random.default_rng(8))
    observed = np.bincount(sample.cell_indices, minlength=population.n_cells) / len(sample)
    np.testing.assert_allclose(observed, sampling_probabilities(population, "strong"), atol=0.01)


def test_poll_exposes_only_cells_and_privatised_categories() -> None:
    population = default_population()
    sample = run_synthetic_poll(population, 500, 1.0, "moderate", np.random.default_rng(9))

    assert len(sample) == 500
    assert sample.reported_categories.shape == sample.cell_indices.shape
    assert not hasattr(sample, "true_categories")
    assert sample.cell_indices.min() >= 0
    assert sample.cell_indices.max() < population.n_cells
    assert sample.reported_categories.min() >= 0
    assert sample.reported_categories.max() < population.n_categories


def test_large_sample_reports_match_the_population_rr_channel() -> None:
    population = default_population()
    epsilon = 1.0
    sample = run_synthetic_poll(population, 250_000, epsilon, "none", np.random.default_rng(10))
    observed = np.bincount(sample.reported_categories, minlength=3) / len(sample)
    expected = population.true_distribution() @ rr_transition_matrix(epsilon, 3)
    np.testing.assert_allclose(observed, expected, atol=0.01)


def test_same_seed_gives_identical_public_results() -> None:
    population = default_population()
    first = run_synthetic_poll(population, 300, 0.8, "moderate", np.random.default_rng(42))
    second = run_synthetic_poll(population, 300, 0.8, "moderate", np.random.default_rng(42))
    np.testing.assert_array_equal(first.cell_indices, second.cell_indices)
    np.testing.assert_array_equal(first.reported_categories, second.reported_categories)


def test_different_seeds_give_different_results() -> None:
    population = default_population()
    first = run_synthetic_poll(population, 300, 0.8, "moderate", np.random.default_rng(1))
    second = run_synthetic_poll(population, 300, 0.8, "moderate", np.random.default_rng(2))
    assert not np.array_equal(first.reported_categories, second.reported_categories)


def test_invalid_poll_arguments_are_rejected() -> None:
    population = default_population()
    rng = np.random.default_rng(11)
    with pytest.raises(ValueError):
        sampling_probabilities(population, "extreme")
    with pytest.raises(ValueError):
        run_synthetic_poll(population, 0, 1.0, "none", rng)
    with pytest.raises(ValueError):
        run_synthetic_poll(population, 100, -1.0, "none", rng)

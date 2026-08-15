from __future__ import annotations

import numpy as np
import pytest

from fairvote.simulation.population import default_population, make_population
from fairvote.simulation.sampling import run_synthetic_poll, sampling_probabilities


def test_default_population_has_the_expected_structure_and_truth() -> None:
    population = default_population()

    assert population.n_cells == 6
    assert population.n_categories == 3
    assert population.regions == (
        "North",
        "North",
        "North",
        "South",
        "South",
        "South",
    )
    assert population.age_groups == (
        "18-34",
        "35-54",
        "55+",
        "18-34",
        "35-54",
        "55+",
    )
    np.testing.assert_allclose(
        population.weights,
        [0.10, 0.15, 0.20, 0.15, 0.20, 0.20],
    )
    np.testing.assert_allclose(
        population.true_distribution(),
        [0.3925, 0.42, 0.1875],
    )
    np.testing.assert_allclose(
        population.preferences.sum(axis=1),
        np.ones(population.n_cells),
    )


def test_population_truth_is_the_weighted_cell_preference_distribution() -> None:
    population = make_population(
        regions=("North", "South"),
        age_groups=("18-34", "55+"),
        weights=(0.25, 0.75),
        preferences=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    )

    np.testing.assert_allclose(
        population.true_distribution(),
        [0.25, 0.75, 0.0],
    )


def test_bias_conditions_shift_sampling_towards_younger_cells() -> None:
    population = default_population()
    none = sampling_probabilities(population, "none")
    moderate = sampling_probabilities(population, "moderate")
    strong = sampling_probabilities(population, "strong")

    np.testing.assert_allclose(none, population.weights)

    for probabilities in (none, moderate, strong):
        assert probabilities.sum() == pytest.approx(1.0)
        assert probabilities.min() > 0.0

    young = np.array([age == "18-34" for age in population.age_groups])
    oldest = np.array([age == "55+" for age in population.age_groups])

    assert none[young].sum() < moderate[young].sum() < strong[young].sum()
    assert none[oldest].sum() > moderate[oldest].sum() > strong[oldest].sum()


def test_large_samples_follow_the_configured_sampling_probabilities() -> None:
    population = default_population()

    for seed, bias in ((7, "none"), (8, "strong")):
        sample = run_synthetic_poll(
            population,
            100_000,
            1.0,
            bias,
            np.random.default_rng(seed),
        )
        observed = np.bincount(
            sample.cell_indices,
            minlength=population.n_cells,
        ) / len(sample)
        expected = sampling_probabilities(population, bias)

        np.testing.assert_allclose(observed, expected, atol=0.01)


def test_poll_exposes_only_cells_and_privatised_categories() -> None:
    population = default_population()
    sample = run_synthetic_poll(
        population,
        500,
        1.0,
        "moderate",
        np.random.default_rng(9),
    )

    assert len(sample) == 500
    assert sample.reported_categories.shape == sample.cell_indices.shape
    assert not hasattr(sample, "true_categories")
    assert sample.cell_indices.min() >= 0
    assert sample.cell_indices.max() < population.n_cells
    assert sample.reported_categories.min() >= 0
    assert sample.reported_categories.max() < population.n_categories


def test_same_seed_produces_identical_public_poll_data() -> None:
    population = default_population()

    first = run_synthetic_poll(
        population,
        300,
        0.8,
        "moderate",
        np.random.default_rng(42),
    )
    second = run_synthetic_poll(
        population,
        300,
        0.8,
        "moderate",
        np.random.default_rng(42),
    )

    np.testing.assert_array_equal(first.cell_indices, second.cell_indices)
    np.testing.assert_array_equal(
        first.reported_categories,
        second.reported_categories,
    )

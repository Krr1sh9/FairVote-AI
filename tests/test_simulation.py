from __future__ import annotations

import numpy as np
import pytest

from fairvote.simulation.population import default_population, make_population
from fairvote.simulation.sampling import run_synthetic_poll, sampling_probabilities


def test_default_population_has_the_expected_structure_and_truth() -> None:
    # This test checks the fixed demographic structure, population weights and resulting truth used by the default synthetic study population.
    population = default_population()

    # The default population contains six demographic cells and three response categories.
    assert population.n_cells == 6
    assert population.n_categories == 3

    # Region labels follow the fixed ordering of three northern cells followed by three southern cells.
    assert population.regions == (
        "North",
        "North",
        "North",
        "South",
        "South",
        "South",
    )

    # Within each region, the three age groups appear in the same fixed order.
    assert population.age_groups == (
        "18-34",
        "35-54",
        "55+",
        "18-34",
        "35-54",
        "55+",
    )

    # The stored population weights must match the six shares defined by the default population specification.
    np.testing.assert_allclose(
        population.weights,
        [0.10, 0.15, 0.20, 0.15, 0.20, 0.20],
    )

    # Weighting the six within-cell preference distributions must produce the known synthetic population truth.
    np.testing.assert_allclose(
        population.true_distribution(),
        [0.3925, 0.42, 0.1875],
    )

    # Every demographic cell's preference row must form a probability distribution over the three categories.
    np.testing.assert_allclose(
        population.preferences.sum(axis=1),
        np.ones(population.n_cells),
    )


def test_population_truth_is_the_weighted_cell_preference_distribution() -> None:
    # This two-cell example makes the population-level weighted combination directly hand-checkable.
    population = make_population(
        regions=("North", "South"),
        age_groups=("18-34", "55+"),
        weights=(0.25, 0.75),
        preferences=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    )

    # A quarter of the population is entirely in category zero and three quarters are entirely in category one.
    np.testing.assert_allclose(
        population.true_distribution(),
        [0.25, 0.75, 0.0],
    )


def test_bias_conditions_shift_sampling_towards_younger_cells() -> None:
    # Sampling probabilities are compared across the three bias conditions defined by the simulation.
    population = default_population()
    none = sampling_probabilities(population, "none")
    moderate = sampling_probabilities(population, "moderate")
    strong = sampling_probabilities(population, "strong")

    # With no sampling bias, the sampling probabilities must equal the population cell weights.
    np.testing.assert_allclose(none, population.weights)

    # Every bias condition must still produce strictly positive sampling probabilities that sum to one.
    for probabilities in (none, moderate, strong):
        assert probabilities.sum() == pytest.approx(1.0)
        assert probabilities.min() > 0.0

    # Boolean masks collect the two cells belonging to the youngest and oldest age groups.
    young = np.array([age == "18-34" for age in population.age_groups])
    oldest = np.array([age == "55+" for age in population.age_groups])

    # The configured bias levels progressively increase the sampling share of younger cells and decrease the share of the oldest cells.
    assert none[young].sum() < moderate[young].sum() < strong[young].sum()
    assert none[oldest].sum() > moderate[oldest].sum() > strong[oldest].sum()


def test_large_samples_follow_the_configured_sampling_probabilities() -> None:
    # Large fixed-seed samples provide an empirical check of the realised demographic sampling proportions for two bias conditions.
    population = default_population()

    for seed, bias in ((7, "none"), (8, "strong")):
        sample = run_synthetic_poll(
            population,
            100_000,
            1.0,
            bias,
            np.random.default_rng(seed),
        )

        # Realised cell proportions are calculated from the sampled demographic cell indices.
        observed = np.bincount(
            sample.cell_indices,
            minlength=population.n_cells,
        ) / len(sample)
        expected = sampling_probabilities(population, bias)

        # Each observed cell proportion must be within 0.01 of the corresponding configured sampling probability for these fixed simulations.
        np.testing.assert_allclose(observed, expected, atol=0.01)


def test_poll_exposes_only_cells_and_privatised_categories() -> None:
    # This test checks the respondent-level data exposed by a returned synthetic poll sample.
    population = default_population()
    sample = run_synthetic_poll(
        population,
        500,
        1.0,
        "moderate",
        np.random.default_rng(9),
    )

    # The returned sample contains one cell index and one privatised category for each of the 500 respondents.
    assert len(sample) == 500
    assert sample.reported_categories.shape == sample.cell_indices.shape

    # Original sampled categories are not retained as an attribute of the returned PollSample.
    assert not hasattr(sample, "true_categories")

    # Returned cell indices and privatised category indices must remain within their respective valid ranges.
    assert sample.cell_indices.min() >= 0
    assert sample.cell_indices.max() < population.n_cells
    assert sample.reported_categories.min() >= 0
    assert sample.reported_categories.max() < population.n_categories


def test_same_seed_produces_identical_public_poll_data() -> None:
    # Two independently created generators with the same seed are used with identical poll settings.
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

    # The same seed and settings must reproduce both returned respondent-level arrays exactly.
    np.testing.assert_array_equal(first.cell_indices, second.cell_indices)
    np.testing.assert_array_equal(
        first.reported_categories,
        second.reported_categories,
    )

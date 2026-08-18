from __future__ import annotations

from collections.abc import Iterator

# The full grid contains the four privacy-budget values used for the main experiment.
FULL_EPSILONS: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)

# Quick mode uses a smaller subset of the full epsilon grid for fast pipeline checks.
QUICK_EPSILONS: tuple[float, ...] = (0.5, 1.0)

# The full grid contains the four respondent counts used for the main experiment.
FULL_SAMPLE_SIZES: tuple[int, ...] = (250, 500, 1000, 2000)

# Quick mode uses a smaller subset of the full sample-size grid.
QUICK_SAMPLE_SIZES: tuple[int, ...] = (500, 1000)

# These three labels define the sampling-bias conditions and their fixed iteration order.
BIAS_CONDITIONS: tuple[str, ...] = ("none", "moderate", "strong")

# The full study repeats each base configuration thirty times, while quick mode uses three repetitions.
FULL_REPETITIONS = 30
QUICK_REPETITIONS = 3

# The repetition number is added to this base value to produce the seed field returned for each grid row.
BASE_SEED = 1000

# Each grid configuration contains epsilon, respondent count, bias label, seed, epsilon index, sample-size index and bias index in that order.
GridConfiguration = tuple[float, int, str, int, int, int, int]


def iter_experiment_grid(quick: bool = False) -> Iterator[GridConfiguration]:
    # The quick flag selects the reduced grid and repetition count without changing the full-grid definitions.
    epsilons = QUICK_EPSILONS if quick else FULL_EPSILONS
    sample_sizes = QUICK_SAMPLE_SIZES if quick else FULL_SAMPLE_SIZES
    repetitions = QUICK_REPETITIONS if quick else FULL_REPETITIONS

    for epsilon in epsilons:
        # The epsilon index is always taken from the full grid so a shared epsilon keeps the same index in full and quick runs.
        epsilon_index = FULL_EPSILONS.index(epsilon)
        for n_respondents in sample_sizes:
            # The sample-size index is also taken from the full grid for consistent indexing across run modes.
            size_index = FULL_SAMPLE_SIZES.index(n_respondents)
            for bias_index, bias in enumerate(BIAS_CONDITIONS):
                # For a given repetition index, the same seed is reused across base configurations, while the returned grid indices identify the configuration position separately.
                for repetition in range(repetitions):
                    yield (
                        epsilon,
                        n_respondents,
                        bias,
                        BASE_SEED + repetition,
                        epsilon_index,
                        size_index,
                        bias_index,
                    )


def poll_row_count(quick: bool = False) -> int:
    # The count uses the same active grid dimensions and repetition count as the iterator.
    epsilons = QUICK_EPSILONS if quick else FULL_EPSILONS
    sample_sizes = QUICK_SAMPLE_SIZES if quick else FULL_SAMPLE_SIZES
    repetitions = QUICK_REPETITIONS if quick else FULL_REPETITIONS

    # This returns the number of synthetic polls rather than the number of estimator-level result rows.
    return len(epsilons) * len(sample_sizes) * len(BIAS_CONDITIONS) * repetitions

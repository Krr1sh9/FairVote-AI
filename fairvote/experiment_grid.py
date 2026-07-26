from __future__ import annotations

from collections.abc import Iterator

FULL_EPSILONS: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
QUICK_EPSILONS: tuple[float, ...] = (0.5, 1.0)
FULL_SAMPLE_SIZES: tuple[int, ...] = (250, 500, 1000, 2000)
QUICK_SAMPLE_SIZES: tuple[int, ...] = (500, 1000)
BIAS_CONDITIONS: tuple[str, ...] = ("none", "moderate", "strong")
FULL_REPETITIONS = 30
QUICK_REPETITIONS = 3
BASE_SEED = 1000

GridConfiguration = tuple[float, int, str, int, int, int, int]


def iter_experiment_grid(quick: bool = False) -> Iterator[GridConfiguration]:
    epsilons = QUICK_EPSILONS if quick else FULL_EPSILONS
    sample_sizes = QUICK_SAMPLE_SIZES if quick else FULL_SAMPLE_SIZES
    repetitions = QUICK_REPETITIONS if quick else FULL_REPETITIONS

    for epsilon in epsilons:
        epsilon_index = FULL_EPSILONS.index(epsilon)
        for n_respondents in sample_sizes:
            size_index = FULL_SAMPLE_SIZES.index(n_respondents)
            for bias_index, bias in enumerate(BIAS_CONDITIONS):
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
    epsilons = QUICK_EPSILONS if quick else FULL_EPSILONS
    sample_sizes = QUICK_SAMPLE_SIZES if quick else FULL_SAMPLE_SIZES
    repetitions = QUICK_REPETITIONS if quick else FULL_REPETITIONS
    return len(epsilons) * len(sample_sizes) * len(BIAS_CONDITIONS) * repetitions

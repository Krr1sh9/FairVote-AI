from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.plots import make_plots
from fairvote.simulation.population import default_population
from fairvote.study import (
    METHODS,
    SUPPORTED_BIAS_LEVELS,
    SUPPORTED_EPSILONS,
    SUPPORTED_SAMPLE_SIZES,
    evaluate_poll,
)

FULL_REPETITIONS = 30
QUICK_REPETITIONS = 3
QUICK_EPSILONS: tuple[float, ...] = (0.5, 1.0)
QUICK_SAMPLE_SIZES: tuple[int, ...] = (500, 1000)

RESULT_COLUMNS: tuple[str, ...] = (
    "epsilon",
    "n_respondents",
    "bias",
    "seed",
    "method",
    "l1_error",
    "max_abs_error",
)

SUMMARY_COLUMNS: tuple[str, ...] = (
    "method",
    "bias",
    "epsilon",
    "n_respondents",
    "n_repetitions",
    "l1_mean",
    "l1_std",
    "max_abs_error_mean",
    "max_abs_error_std",
)


def run_grid(quick: bool = False) -> pd.DataFrame:
    population = default_population()
    epsilons = QUICK_EPSILONS if quick else SUPPORTED_EPSILONS
    sample_sizes = QUICK_SAMPLE_SIZES if quick else SUPPORTED_SAMPLE_SIZES
    repetitions = QUICK_REPETITIONS if quick else FULL_REPETITIONS

    rows: list[dict[str, object]] = []
    for epsilon in epsilons:
        epsilon_index = SUPPORTED_EPSILONS.index(epsilon)
        for n_respondents in sample_sizes:
            size_index = SUPPORTED_SAMPLE_SIZES.index(n_respondents)
            for bias_index, bias in enumerate(SUPPORTED_BIAS_LEVELS):
                for repetition in range(repetitions):
                    seed = 1000 + repetition
                    rng = np.random.default_rng([seed, epsilon_index, size_index, bias_index])
                    study = evaluate_poll(
                        population,
                        n_respondents,
                        epsilon,
                        bias,
                        rng,
                    )
                    for method in METHODS:
                        rows.append(
                            {
                                "epsilon": epsilon,
                                "n_respondents": n_respondents,
                                "bias": bias,
                                "seed": seed,
                                "method": method,
                                "l1_error": study.l1_errors[method],
                                "max_abs_error": study.max_abs_errors[method],
                            }
                        )

    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    summary = results.groupby(
        ["method", "bias", "epsilon", "n_respondents"],
        as_index=False,
    ).agg(
        n_repetitions=("l1_error", "size"),
        l1_mean=("l1_error", "mean"),
        l1_std=("l1_error", "std"),
        max_abs_error_mean=("max_abs_error", "mean"),
        max_abs_error_std=("max_abs_error", "std"),
    )
    return (
        summary.loc[:, SUMMARY_COLUMNS]
        .sort_values(["method", "bias", "epsilon", "n_respondents"])
        .reset_index(drop=True)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the FairVote-AI estimator comparison.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a small reference grid with 3 repetitions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for CSVs and plots (default: results).",
    )
    args = parser.parse_args(argv)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_csv in (output_dir / "experiment_results.csv", output_dir / "summary.csv"):
        stale_csv.unlink(missing_ok=True)

    results = run_grid(quick=args.quick)
    summary = summarise(results)

    results_path = output_dir / "experiment_results.csv"
    summary_path = output_dir / "summary.csv"
    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_paths = make_plots(results, output_dir / "plots")

    mode = "quick" if args.quick else "full"
    print(f"FairVote-AI experiment finished ({mode} mode).")
    print(f"  rows written        : {len(results)}")
    print(f"  configurations      : {len(summary)}")
    print(f"  results CSV         : {results_path}")
    print(f"  summary CSV         : {summary_path}")
    print(f"  plots               : {', '.join(path.name for path in plot_paths)}")
    print("  mean L1 error by method:")
    for method, mean_l1 in results.groupby("method")["l1_error"].mean().items():
        print(f"    {method:<18} {mean_l1:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

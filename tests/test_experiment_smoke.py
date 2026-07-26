from __future__ import annotations

import py_compile
from pathlib import Path

import pandas as pd
import pytest

from experiments.run_experiments import (
    RESULT_COLUMNS,
    SUMMARY_COLUMNS,
    main,
    run_grid,
    summarise,
)
from fairvote.experiment_grid import BIAS_CONDITIONS, FULL_EPSILONS, FULL_SAMPLE_SIZES, poll_row_count
from fairvote.study import METHODS, SUPPORTED_BIAS_LEVELS, SUPPORTED_EPSILONS, SUPPORTED_SAMPLE_SIZES

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FULL_EXPERIMENT_RESULTS_PATH = REPOSITORY_ROOT / "results" / "experiment_results.csv"
FULL_SUMMARY_RESULTS_PATH = REPOSITORY_ROOT / "results" / "summary.csv"
FULL_STATISTICAL_RESULTS_AVAILABLE = FULL_EXPERIMENT_RESULTS_PATH.is_file() and FULL_SUMMARY_RESULTS_PATH.is_file()


@pytest.fixture(scope="module")
def quick_results() -> pd.DataFrame:
    return run_grid(quick=True)


def test_supported_grid_values_are_shared() -> None:
    assert SUPPORTED_EPSILONS is FULL_EPSILONS
    assert SUPPORTED_SAMPLE_SIZES is FULL_SAMPLE_SIZES
    assert SUPPORTED_BIAS_LEVELS is BIAS_CONDITIONS


def test_run_grid_produces_tidy_rows(quick_results: pd.DataFrame) -> None:
    assert tuple(quick_results.columns) == RESULT_COLUMNS
    assert len(quick_results) == 108
    assert len(quick_results) == len(
        quick_results.drop_duplicates(["epsilon", "n_respondents", "bias", "seed", "method"])
    )
    assert set(quick_results["bias"]) == set(SUPPORTED_BIAS_LEVELS)
    assert quick_results["l1_error"].min() >= 0.0
    assert quick_results["max_abs_error"].min() >= 0.0


@pytest.mark.skipif(
    not FULL_STATISTICAL_RESULTS_AVAILABLE,
    reason="Full generated statistical results are not present in this checkout.",
)
def test_full_generated_experiment_has_expected_shapes() -> None:
    results = pd.read_csv(FULL_EXPERIMENT_RESULTS_PATH)
    summary = pd.read_csv(FULL_SUMMARY_RESULTS_PATH)
    assert poll_row_count(quick=False) * len(METHODS) == 4320
    assert len(results) == 4320
    assert len(summary) == 144
    assert tuple(results.columns) == RESULT_COLUMNS
    assert tuple(summary.columns) == SUMMARY_COLUMNS


def test_summarise_reports_expected_schema(quick_results: pd.DataFrame) -> None:
    summary = summarise(quick_results)
    assert tuple(summary.columns) == SUMMARY_COLUMNS
    assert len(summary) == 36
    assert (summary["n_repetitions"] == 3).all()


def test_quick_mode_writes_csvs_and_three_plots(tmp_path: Path) -> None:
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()
    existing_ai_plot = plots_dir / "ai_selector_baselines.png"
    existing_ai_plot.write_bytes(b"existing AI plot")

    assert main(["--quick", "--output-dir", str(tmp_path)]) == 0

    results = pd.read_csv(tmp_path / "experiment_results.csv")
    summary = pd.read_csv(tmp_path / "summary.csv")
    assert tuple(results.columns) == RESULT_COLUMNS
    assert tuple(summary.columns) == SUMMARY_COLUMNS
    assert len(results) == 108
    assert len(summary) == 36

    plot_names = {path.name for path in plots_dir.glob("*.png")}
    assert {
        "l1_vs_epsilon.png",
        "l1_vs_sample_size.png",
        "l1_by_bias.png",
    } <= plot_names
    assert existing_ai_plot.read_bytes() == b"existing AI plot"


def test_running_twice_overwrites_reproducibly(tmp_path: Path) -> None:
    assert main(["--quick", "--output-dir", str(tmp_path)]) == 0
    first = pd.read_csv(tmp_path / "experiment_results.csv")
    assert main(["--quick", "--output-dir", str(tmp_path)]) == 0
    second = pd.read_csv(tmp_path / "experiment_results.csv")
    pd.testing.assert_frame_equal(first, second)


def test_app_compiles_without_starting_a_server() -> None:
    py_compile.compile(str(REPOSITORY_ROOT / "app.py"), doraise=True)

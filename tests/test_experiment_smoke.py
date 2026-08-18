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

# The repository root is used to locate committed full-run outputs and the Streamlit entry point.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

# These paths point to the committed full statistical result tables when they are present in the checkout.
FULL_EXPERIMENT_RESULTS_PATH = REPOSITORY_ROOT / "results" / "experiment_results.csv"
FULL_SUMMARY_RESULTS_PATH = REPOSITORY_ROOT / "results" / "summary.csv"

# The full-result shape test is enabled only when both generated statistical CSV files are available.
FULL_STATISTICAL_RESULTS_AVAILABLE = FULL_EXPERIMENT_RESULTS_PATH.is_file() and FULL_SUMMARY_RESULTS_PATH.is_file()


@pytest.fixture(scope="module")
def quick_results() -> pd.DataFrame:
    # Quick-mode estimator rows are generated once for this module and reused by the tests that inspect them.
    return run_grid(quick=True)


def test_supported_grid_values_are_shared() -> None:
    # These identity checks confirm that study-level supported settings reference the same shared grid objects.
    assert SUPPORTED_EPSILONS is FULL_EPSILONS
    assert SUPPORTED_SAMPLE_SIZES is FULL_SAMPLE_SIZES
    assert SUPPORTED_BIAS_LEVELS is BIAS_CONDITIONS


def test_run_grid_produces_tidy_rows(quick_results: pd.DataFrame) -> None:
    # Quick mode must return the declared result schema and the expected 108 estimator-level rows.
    assert tuple(quick_results.columns) == RESULT_COLUMNS
    assert len(quick_results) == 108

    # Each combination of grid settings, repetition seed and estimator must identify one unique result row.
    assert len(quick_results) == len(
        quick_results.drop_duplicates(["epsilon", "n_respondents", "bias", "seed", "method"])
    )

    # All configured bias levels must appear, and both stored error metrics must remain non-negative.
    assert set(quick_results["bias"]) == set(SUPPORTED_BIAS_LEVELS)
    assert quick_results["l1_error"].min() >= 0.0
    assert quick_results["max_abs_error"].min() >= 0.0


@pytest.mark.skipif(
    not FULL_STATISTICAL_RESULTS_AVAILABLE,
    reason="Full generated statistical results are not present in this checkout.",
)
def test_full_generated_experiment_has_expected_shapes() -> None:
    # This integration check reads the generated full-run tables only when both files exist.
    results = pd.read_csv(FULL_EXPERIMENT_RESULTS_PATH)
    summary = pd.read_csv(FULL_SUMMARY_RESULTS_PATH)

    # The full grid contains 1,440 simulated polls, with one estimator-level result row for each configured method.
    assert poll_row_count(quick=False) * len(METHODS) == 4320
    assert len(results) == 4320

    # Aggregating by method, bias, epsilon and sample size produces the expected 144 summary rows.
    assert len(summary) == 144

    # Both full-run tables must retain the schemas declared by the experiment runner.
    assert tuple(results.columns) == RESULT_COLUMNS
    assert tuple(summary.columns) == SUMMARY_COLUMNS


def test_summarise_reports_expected_schema(quick_results: pd.DataFrame) -> None:
    # Summarising the quick estimator rows must produce the declared summary schema.
    summary = summarise(quick_results)
    assert tuple(summary.columns) == SUMMARY_COLUMNS

    # Quick mode produces 36 method-by-configuration summary rows, each aggregating three repetitions.
    assert len(summary) == 36
    assert (summary["n_repetitions"] == 3).all()


def test_quick_mode_writes_csvs_and_three_plots(tmp_path: Path) -> None:
    # A temporary output directory isolates this end-to-end quick-mode run from the repository's generated results.
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()

    # An existing recommender plot is placed beside the statistical plots to verify that this pipeline leaves it unchanged.
    existing_ai_plot = plots_dir / "ai_selector_baselines.png"
    existing_ai_plot.write_bytes(b"existing AI plot")

    # A zero return code indicates that the quick experiment command completed successfully.
    assert main(["--quick", "--output-dir", str(tmp_path)]) == 0

    # The generated CSV files must use the declared schemas and quick-mode row counts.
    results = pd.read_csv(tmp_path / "experiment_results.csv")
    summary = pd.read_csv(tmp_path / "summary.csv")
    assert tuple(results.columns) == RESULT_COLUMNS
    assert tuple(summary.columns) == SUMMARY_COLUMNS
    assert len(results) == 108
    assert len(summary) == 36

    # The three statistical plot files must be present after the run.
    plot_names = {path.name for path in plots_dir.glob("*.png")}
    assert {
        "l1_vs_epsilon.png",
        "l1_vs_sample_size.png",
        "l1_by_bias.png",
    } <= plot_names

    # The pre-existing recommender plot must retain its original contents.
    assert existing_ai_plot.read_bytes() == b"existing AI plot"


def test_running_twice_overwrites_reproducibly(tmp_path: Path) -> None:
    # The same quick experiment is executed twice into the same isolated output directory.
    assert main(["--quick", "--output-dir", str(tmp_path)]) == 0
    first = pd.read_csv(tmp_path / "experiment_results.csv")
    assert main(["--quick", "--output-dir", str(tmp_path)]) == 0
    second = pd.read_csv(tmp_path / "experiment_results.csv")

    # The parsed estimator-level result tables must be exactly equal across the two runs in this test environment.
    pd.testing.assert_frame_equal(first, second)


def test_app_compiles_without_starting_a_server() -> None:
    # Byte-compiling the Streamlit entry point checks that it compiles successfully without launching the app server.
    py_compile.compile(str(REPOSITORY_ROOT / "app.py"), doraise=True)

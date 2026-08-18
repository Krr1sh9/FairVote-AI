from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest

from experiments import run_experiments
from fairvote import experiment_grid
from fairvote.ai import features
from fairvote.study import METHODS

# The repository root is used to locate generated results and to scan project source files in the structural tests below.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

# These paths point to the full-run statistical results and selector dataset when they are present in the checkout.
FULL_EXPERIMENT_RESULTS_PATH = REPOSITORY_ROOT / "results" / "experiment_results.csv"
FULL_SELECTOR_DATASET_PATH = REPOSITORY_ROOT / "results" / "ai" / "selector_dataset.csv"

# The cross-pipeline target comparison is enabled only when both full generated data files are available.
FULL_AI_TARGET_RESULTS_AVAILABLE = FULL_EXPERIMENT_RESULTS_PATH.is_file() and FULL_SELECTOR_DATASET_PATH.is_file()

# These names are the experiment-grid constants that this test suite expects to have a single definition in the shared grid module.
GRID_CONSTANTS = {
    "FULL_EPSILONS",
    "QUICK_EPSILONS",
    "FULL_SAMPLE_SIZES",
    "QUICK_SAMPLE_SIZES",
    "BIAS_CONDITIONS",
    "FULL_REPETITIONS",
    "QUICK_REPETITIONS",
    "BASE_SEED",
}

# Repository scans skip version-control, environment, cache and build directories that are not part of the project source under test.
EXCLUDED_SCAN_DIRECTORIES = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "build",
    "dist",
}


def test_shared_grid_values_are_exact() -> None:
    # This test fixes the full and quick experiment settings, repetition counts and base seed used by the shared grid module.
    assert experiment_grid.FULL_EPSILONS == (0.25, 0.5, 1.0, 2.0)
    assert experiment_grid.QUICK_EPSILONS == (0.5, 1.0)
    assert experiment_grid.FULL_SAMPLE_SIZES == (250, 500, 1000, 2000)
    assert experiment_grid.QUICK_SAMPLE_SIZES == (500, 1000)
    assert experiment_grid.BIAS_CONDITIONS == ("none", "moderate", "strong")
    assert experiment_grid.FULL_REPETITIONS == 30
    assert experiment_grid.QUICK_REPETITIONS == 3
    assert experiment_grid.BASE_SEED == 1000


def test_shared_grid_preserves_all_required_shapes() -> None:
    # Base configurations are the Cartesian products of epsilon, sample size and bias before repetitions or estimator rows are counted.
    full_configurations = (
        len(experiment_grid.FULL_EPSILONS)
        * len(experiment_grid.FULL_SAMPLE_SIZES)
        * len(experiment_grid.BIAS_CONDITIONS)
    )
    quick_configurations = (
        len(experiment_grid.QUICK_EPSILONS)
        * len(experiment_grid.QUICK_SAMPLE_SIZES)
        * len(experiment_grid.BIAS_CONDITIONS)
    )

    # These assertions check base configurations, simulated-poll rows, estimator-level result rows and method-by-configuration summary rows.
    assert full_configurations == 48
    assert quick_configurations == 12
    assert experiment_grid.poll_row_count(quick=False) == 1440
    assert experiment_grid.poll_row_count(quick=True) == 36
    assert experiment_grid.poll_row_count(quick=False) * len(METHODS) == 4320
    assert experiment_grid.poll_row_count(quick=True) * len(METHODS) == 108
    assert full_configurations * len(METHODS) == 144
    assert quick_configurations * len(METHODS) == 36


def test_seed_derivation_and_full_indices_are_unchanged() -> None:
    # Quick mode still reports epsilon and sample-size indices from the full grids, while seeds start from the shared base seed.
    quick = list(experiment_grid.iter_experiment_grid(quick=True))

    # These boundary entries pin the iterator ordering, repetition seeds and preserved full-grid indices for representative quick-grid points.
    assert quick[0] == (0.5, 500, "none", 1000, 1, 1, 0)
    assert quick[2] == (0.5, 500, "none", 1002, 1, 1, 0)
    assert quick[-1] == (1.0, 1000, "strong", 1002, 2, 2, 2)


def test_grid_constants_are_defined_only_in_the_shared_module() -> None:
    # This source scan records top-level assignments of shared grid constant names outside fairvote/experiment_grid.py.
    duplicate_assignments: list[tuple[str, str]] = []
    shared_path = REPOSITORY_ROOT / "fairvote" / "experiment_grid.py"
    for path in REPOSITORY_ROOT.rglob("*.py"):
        if path == shared_path or any(part in EXCLUDED_SCAN_DIRECTORIES for part in path.parts):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            names: list[str] = []
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        names.append(target.id)
            for name in names:
                if name in GRID_CONSTANTS:
                    duplicate_assignments.append((str(path.relative_to(REPOSITORY_ROOT)), name))

    # An empty result means none of the scanned Python files redefines these constants as a top-level simple name.
    assert duplicate_assignments == []


def test_both_pipelines_use_the_shared_grid(monkeypatch) -> None:
    # Replacing the shared iterator with one known configuration reveals whether each pipeline calls the shared grid entry point.
    calls: list[bool] = []
    configuration = (0.5, 500, "none", 1000, 1, 1, 0)

    def shared_iterator(quick: bool = False):
        # Recording the flag checks that both callers preserve the requested quick-mode setting.
        calls.append(quick)
        yield configuration

    monkeypatch.setattr(experiment_grid, "iter_experiment_grid", shared_iterator)
    statistical = run_experiments.run_grid(quick=True)
    selector = features.build_selector_dataset(quick=True)

    # One shared configuration produces one statistical row per estimator and one selector row for the simulated poll.
    assert calls == [True, True]
    assert len(statistical) == len(METHODS)
    assert len(selector) == 1


@pytest.mark.skipif(
    not FULL_AI_TARGET_RESULTS_AVAILABLE,
    reason="Full generated statistical and AI results are not present in this checkout.",
)
def test_full_ai_targets_match_full_experiment_l1_errors() -> None:
    # This integration check runs only when both full-run generated data files are available.
    experiment = pd.read_csv(FULL_EXPERIMENT_RESULTS_PATH)
    selector = pd.read_csv(FULL_SELECTOR_DATASET_PATH)

    # These four fields identify the same simulated poll across the estimator-level experiment table and selector dataset.
    keys = ["epsilon", "n_respondents", "bias", "seed"]

    # Pivoting the experiment table creates one L1 target column per estimator so it can be compared directly with the selector dataset.
    targets = (
        experiment.pivot(index=keys, columns="method", values="l1_error")
        .rename(columns={method: f"{method}_l1" for method in METHODS})
        .reset_index()
    )

    # A one-to-one merge requires each poll key to identify exactly one row in each comparison table.
    merged = selector.merge(targets, on=keys, suffixes=("_selector", "_experiment"), validate="one_to_one")

    # Each selector target must exactly match the corresponding stored experiment L1 error at parsed numeric value level.
    for method in METHODS:
        column = f"{method}_l1"
        left = merged[f"{column}_selector"].to_numpy()
        right = merged[f"{column}_experiment"].to_numpy()
        assert (left == right).all()


def test_fairvote_never_imports_experiments() -> None:
    # This source scan records import statements inside the fairvote package that reference the experiments package by module name.
    imports: list[tuple[str, str]] = []
    for path in (REPOSITORY_ROOT / "fairvote").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "experiments" or alias.name.startswith("experiments."):
                        imports.append((str(path.relative_to(REPOSITORY_ROOT)), alias.name))
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and (node.module == "experiments" or node.module.startswith("experiments."))
            ):
                imports.append((str(path.relative_to(REPOSITORY_ROOT)), node.module))

    # No recorded imports keeps the core fairvote package independent of the experiments package at the Python import level checked here.
    assert imports == []

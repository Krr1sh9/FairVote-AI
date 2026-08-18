from __future__ import annotations

import numpy as np
import pandas as pd

from fairvote import experiment_grid
from fairvote.simulation.population import default_population
from fairvote.study import METHODS, StudyResult, evaluate_poll

# These are the six inputs available to the selector when it predicts estimator error for a poll.
FEATURE_COLUMNS: tuple[str, ...] = (
    "epsilon",
    "n_respondents",
    "demographic_imbalance",
    "cell_proportion_min",
    "cell_proportion_max",
    "cell_proportion_std",
)

# Each target is the measured L1 error of one of the three estimators for the same synthetic poll.
TARGET_COLUMNS: tuple[str, ...] = (
    "raw_frequencies_l1",
    "rr_debiased_l1",
    "poststratified_l1",
)

# These values are retained in the selector dataset but are not supplied to the selector as model features.
AUDIT_COLUMNS: tuple[str, ...] = ("seed", "bias", "fallback_cells")

# The selector dataset stores one row per synthetic poll with model inputs, audit fields and the three error targets.
DATASET_COLUMNS: tuple[str, ...] = (
    "epsilon",
    "n_respondents",
    "seed",
    "bias",
    "demographic_imbalance",
    "fallback_cells",
    "cell_proportion_min",
    "cell_proportion_max",
    "cell_proportion_std",
    *TARGET_COLUMNS,
)

# These columns document values that must remain separate from the selector's approved feature set.
# This includes hidden simulation controls, audit information, measured outcomes and estimator identifiers.
FORBIDDEN_FEATURE_COLUMNS: tuple[str, ...] = (
    "bias",
    "seed",
    "fallback_cells",
    "truth",
    "method",
    "method_label",
    "configuration_index",
    "epsilon_index",
    "sample_size_index",
    "bias_index",
    "repetition",
    "l1_error",
    "max_abs_error",
    *TARGET_COLUMNS,
    *METHODS,
)


def target_column(method: str) -> str:
    # Each estimator is associated with one L1-error target column in the selector dataset.
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}.")
    return f"{method}_l1"


def extract_features(result: StudyResult, epsilon: float, n_respondents: int) -> dict[str, float]:
    # The realised cell proportions describe the demographic composition of this particular synthetic poll.
    proportions = np.asarray(result.sample_cell_proportions, dtype=float)
    if proportions.ndim != 1 or proportions.size == 0:
        raise ValueError("sample_cell_proportions must be a non-empty 1D array.")
    if not np.all(np.isfinite(proportions)):
        raise ValueError("sample_cell_proportions must be finite.")

    # The minimum, maximum and population standard deviation provide compact summaries of the realised cell proportions.
    # demographic_imbalance is calculated by the shared study workflow against the known population cell weights.
    return {
        "epsilon": float(epsilon),
        "n_respondents": float(n_respondents),
        "demographic_imbalance": float(result.demographic_imbalance),
        "cell_proportion_min": float(proportions.min()),
        "cell_proportion_max": float(proportions.max()),
        "cell_proportion_std": float(proportions.std(ddof=0)),
    }


def feature_matrix(dataset: pd.DataFrame) -> np.ndarray:
    # Selecting columns in FEATURE_COLUMNS order ensures that training and prediction use the same feature positions.
    missing = [column for column in FEATURE_COLUMNS if column not in dataset.columns]
    if missing:
        raise ValueError(f"dataset is missing feature columns: {missing}.")
    return dataset.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=float)


def build_selector_dataset(quick: bool = False) -> pd.DataFrame:
    # The selector dataset is generated from the same experiment grid used by the statistical study.
    population = default_population()
    rows: list[dict[str, object]] = []

    # Each grid entry represents one synthetic poll rather than one estimator-level result row.
    for (
        epsilon,
        n_respondents,
        bias,
        seed,
        epsilon_index,
        size_index,
        bias_index,
    ) in experiment_grid.iter_experiment_grid(quick=quick):
        # The generator is initialised from the repetition seed together with the three grid-position indices.
        rng = np.random.default_rng([seed, epsilon_index, size_index, bias_index])
        result = evaluate_poll(population, n_respondents, epsilon, bias, rng)

        # Audit fields are stored alongside the six approved model features without becoming model inputs.
        row: dict[str, object] = {
            "seed": seed,
            "bias": bias,
            **extract_features(result, epsilon, n_respondents),
            "fallback_cells": int(result.fallback_cells),
        }

        # The dataset records respondent count as an integer even though the model feature matrix later converts features to floating-point values.
        row["n_respondents"] = int(n_respondents)

        # The observed L1 error of each estimator becomes that estimator's supervised regression target.
        for method in METHODS:
            row[target_column(method)] = float(result.l1_errors[method])

        rows.append(row)

    # Supplying DATASET_COLUMNS fixes the output column order used by training, evaluation and exported results.
    return pd.DataFrame(rows, columns=list(DATASET_COLUMNS))


def expected_row_count(quick: bool = False) -> int:
    # There is one selector-dataset row for each synthetic poll in the chosen experiment grid.
    return experiment_grid.poll_row_count(quick=quick)


def validate_selector_dataset(dataset: pd.DataFrame, quick: bool | None = None) -> None:
    # Validation checks the structure and value constraints expected by selector training.
    missing = [column for column in DATASET_COLUMNS if column not in dataset.columns]
    if missing:
        raise ValueError(f"dataset is missing required columns: {missing}.")
    if dataset.empty:
        raise ValueError("dataset must contain at least one row.")

    # Every required column except the categorical bias label is expected to contain finite numeric values.
    numeric_columns = [column for column in DATASET_COLUMNS if column != "bias"]
    numeric = dataset.loc[:, numeric_columns].to_numpy(dtype=float)
    if not np.all(np.isfinite(numeric)):
        raise ValueError("all numeric dataset values must be finite.")

    # Respondent counts represent poll sizes and therefore must be positive whole numbers.
    sizes = dataset["n_respondents"].to_numpy()
    if not np.all(sizes > 0) or not np.all(sizes == sizes.astype(int)):
        raise ValueError("n_respondents must contain positive integers.")

    # fallback_cells counts how many demographic cells used the poststratification fallback for a poll.
    fallback = dataset["fallback_cells"].to_numpy(dtype=float)
    if not np.all(fallback == fallback.astype(int)):
        raise ValueError("fallback_cells must contain integer values.")
    if not np.all((fallback >= 0) & (fallback <= default_population().n_cells)):
        raise ValueError("fallback_cells must lie between 0 and the number of demographic cells.")

    # These three features are summaries of cell proportions and must remain within the probability range.
    for column in ("cell_proportion_min", "cell_proportion_max", "cell_proportion_std"):
        values = dataset[column].to_numpy(dtype=float)
        if not np.all((values >= 0.0) & (values <= 1.0)):
            raise ValueError(f"{column} must lie between 0 and 1.")

    # L1 error cannot be negative for any estimator.
    for column in TARGET_COLUMNS:
        if np.any(dataset[column].to_numpy(dtype=float) < 0.0):
            raise ValueError(f"{column} must be non-negative.")

    # The four identifying fields distinguish the individual synthetic polls in the experiment grid.
    keys = ["epsilon", "n_respondents", "bias", "seed"]
    if dataset.duplicated(subset=keys).any():
        raise ValueError("dataset must contain one row per epsilon, sample size, bias and seed.")

    # When the caller specifies full or quick mode, the row count is also checked against the corresponding experiment grid.
    if quick is not None:
        expected = expected_row_count(quick=quick)
        if len(dataset) != expected:
            raise ValueError(f"dataset must contain {expected} rows, got {len(dataset)}.")

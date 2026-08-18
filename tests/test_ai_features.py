from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fairvote.ai.features import (
    AUDIT_COLUMNS,
    DATASET_COLUMNS,
    FEATURE_COLUMNS,
    FORBIDDEN_FEATURE_COLUMNS,
    TARGET_COLUMNS,
    build_selector_dataset,
    expected_row_count,
    extract_features,
    feature_matrix,
    target_column,
    validate_selector_dataset,
)
from fairvote.simulation.population import default_population
from fairvote.study import METHODS, evaluate_poll

# The repository root is used to locate the generated full selector dataset when it is present in the checkout.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FULL_SELECTOR_DATASET_PATH = REPOSITORY_ROOT / "results" / "ai" / "selector_dataset.csv"


def test_selector_schema_uses_only_observable_features() -> None:
    # The recommender input schema is fixed to these six numeric features.
    assert FEATURE_COLUMNS == (
        "epsilon",
        "n_respondents",
        "demographic_imbalance",
        "cell_proportion_min",
        "cell_proportion_max",
        "cell_proportion_std",
    )

    # Each estimator contributes one L1-error target column to the selector dataset.
    assert TARGET_COLUMNS == (
        "raw_frequencies_l1",
        "rr_debiased_l1",
        "poststratified_l1",
    )

    # Seed, sampling-bias label and empty-cell fallback count are retained for audit purposes rather than used as model inputs.
    assert AUDIT_COLUMNS == ("seed", "bias", "fallback_cells")

    # The target-column helper must map the configured estimators to the declared target columns in the same order.
    assert tuple(target_column(method) for method in METHODS) == TARGET_COLUMNS

    # Audit fields, targets, explicitly forbidden fields and estimator identifiers must all remain outside the model feature set.
    assert set(AUDIT_COLUMNS).isdisjoint(FEATURE_COLUMNS)
    assert set(TARGET_COLUMNS).isdisjoint(FEATURE_COLUMNS)
    assert set(FORBIDDEN_FEATURE_COLUMNS).isdisjoint(FEATURE_COLUMNS)
    assert set(METHODS).isdisjoint(FEATURE_COLUMNS)


def test_feature_extraction_matches_a_hand_checked_poll() -> None:
    # This fixed synthetic poll provides a reproducible StudyResult from which the six recommender features can be checked directly.
    population = default_population()
    rng = np.random.default_rng([1000, 2, 2, 2])
    result = evaluate_poll(population, 1000, 1.0, "strong", rng)

    extracted = extract_features(result, 1.0, 1000)
    proportions = result.sample_cell_proportions

    # Configured epsilon and sample size are copied into the feature dictionary as numeric values.
    assert extracted["epsilon"] == 1.0
    assert extracted["n_respondents"] == 1000.0

    # The demographic feature must equal the imbalance diagnostic already calculated for the poll.
    assert extracted["demographic_imbalance"] == pytest.approx(result.demographic_imbalance)

    # The remaining three features summarise the observed demographic cell proportions using their minimum, maximum and population-standard-deviation values.
    assert extracted["cell_proportion_min"] == pytest.approx(float(proportions.min()))
    assert extracted["cell_proportion_max"] == pytest.approx(float(proportions.max()))
    assert extracted["cell_proportion_std"] == pytest.approx(float(np.std(proportions, ddof=0)))

    # The empty-cell fallback diagnostic is not returned as a model feature.
    assert "fallback_cells" not in extracted


def test_feature_matrix_contains_only_the_six_model_features() -> None:
    # Quick mode supplies a complete selector dataset while keeping this feature-isolation test inexpensive.
    dataset = build_selector_dataset(quick=True)
    matrix = feature_matrix(dataset)

    # The matrix must contain one row per poll and one column for each declared model feature.
    assert matrix.shape == (len(dataset), len(FEATURE_COLUMNS))

    # Matrix columns must preserve the exact order declared by FEATURE_COLUMNS.
    for position, column in enumerate(FEATURE_COLUMNS):
        np.testing.assert_allclose(
            matrix[:, position],
            dataset[column].to_numpy(dtype=float),
        )

    # Audit fields and target values are deliberately replaced to test that feature_matrix does not depend on them.
    changed = dataset.copy()
    changed["seed"] = 999999
    changed["bias"] = "strong"
    changed["fallback_cells"] = 6

    for column in TARGET_COLUMNS:
        changed[column] = 999.0

    # Changing every audit field and target must leave the extracted model matrix unchanged.
    np.testing.assert_array_equal(feature_matrix(changed), matrix)


def test_quick_selector_dataset_is_deterministic_and_well_formed() -> None:
    # Building the quick selector dataset twice checks deterministic generation under the fixed quick-grid settings and seed derivation.
    first = build_selector_dataset(quick=True)
    second = build_selector_dataset(quick=True)

    # Quick mode contains 36 simulated polls and must return the complete declared dataset schema.
    assert expected_row_count(quick=True) == 36
    assert len(first) == 36
    assert list(first.columns) == list(DATASET_COLUMNS)

    # Epsilon, sample size, bias and repetition seed together must uniquely identify each quick-grid poll row.
    keys = ["epsilon", "n_respondents", "bias", "seed"]
    assert not first.duplicated(subset=keys).any()

    # All three estimator L1 targets must be finite and non-negative.
    targets = first.loc[:, list(TARGET_COLUMNS)].to_numpy(dtype=float)
    assert np.all(np.isfinite(targets))
    assert targets.min() >= 0.0

    # The generated dataset must pass the project's quick-mode validation and match the independently rebuilt copy exactly.
    validate_selector_dataset(first, quick=True)
    pd.testing.assert_frame_equal(first, second)


@pytest.mark.skipif(
    not FULL_SELECTOR_DATASET_PATH.is_file(),
    reason="The full generated AI selector dataset is not present in this checkout.",
)
def test_full_generated_selector_dataset_matches_the_full_grid() -> None:
    # This generated-output check runs only when the full selector dataset is available in the repository checkout.
    dataset = pd.read_csv(FULL_SELECTOR_DATASET_PATH)

    # The full experiment grid contains 1,440 simulated polls, with one selector row for each poll.
    assert expected_row_count(quick=False) == 1440
    assert len(dataset) == 1440
    assert list(dataset.columns) == list(DATASET_COLUMNS)

    # The committed full dataset must satisfy the same schema and content validation used by the selector pipeline.
    validate_selector_dataset(dataset, quick=False)

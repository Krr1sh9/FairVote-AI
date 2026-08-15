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

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FULL_SELECTOR_DATASET_PATH = REPOSITORY_ROOT / "results" / "ai" / "selector_dataset.csv"


def test_selector_schema_uses_only_observable_features() -> None:
    assert FEATURE_COLUMNS == (
        "epsilon",
        "n_respondents",
        "demographic_imbalance",
        "cell_proportion_min",
        "cell_proportion_max",
        "cell_proportion_std",
    )
    assert TARGET_COLUMNS == (
        "raw_frequencies_l1",
        "rr_debiased_l1",
        "poststratified_l1",
    )
    assert AUDIT_COLUMNS == ("seed", "bias", "fallback_cells")
    assert tuple(target_column(method) for method in METHODS) == TARGET_COLUMNS
    assert set(AUDIT_COLUMNS).isdisjoint(FEATURE_COLUMNS)
    assert set(TARGET_COLUMNS).isdisjoint(FEATURE_COLUMNS)
    assert set(FORBIDDEN_FEATURE_COLUMNS).isdisjoint(FEATURE_COLUMNS)
    assert set(METHODS).isdisjoint(FEATURE_COLUMNS)


def test_feature_extraction_matches_a_hand_checked_poll() -> None:
    population = default_population()
    rng = np.random.default_rng([1000, 2, 2, 2])
    result = evaluate_poll(population, 1000, 1.0, "strong", rng)

    extracted = extract_features(result, 1.0, 1000)
    proportions = result.sample_cell_proportions

    assert extracted["epsilon"] == 1.0
    assert extracted["n_respondents"] == 1000.0
    assert extracted["demographic_imbalance"] == pytest.approx(result.demographic_imbalance)
    assert extracted["cell_proportion_min"] == pytest.approx(float(proportions.min()))
    assert extracted["cell_proportion_max"] == pytest.approx(float(proportions.max()))
    assert extracted["cell_proportion_std"] == pytest.approx(float(np.std(proportions, ddof=0)))
    assert "fallback_cells" not in extracted


def test_feature_matrix_contains_only_the_six_model_features() -> None:
    dataset = build_selector_dataset(quick=True)
    matrix = feature_matrix(dataset)

    assert matrix.shape == (len(dataset), len(FEATURE_COLUMNS))

    for position, column in enumerate(FEATURE_COLUMNS):
        np.testing.assert_allclose(
            matrix[:, position],
            dataset[column].to_numpy(dtype=float),
        )

    changed = dataset.copy()
    changed["seed"] = 999999
    changed["bias"] = "strong"
    changed["fallback_cells"] = 6

    for column in TARGET_COLUMNS:
        changed[column] = 999.0

    np.testing.assert_array_equal(feature_matrix(changed), matrix)


def test_quick_selector_dataset_is_deterministic_and_well_formed() -> None:
    first = build_selector_dataset(quick=True)
    second = build_selector_dataset(quick=True)

    assert expected_row_count(quick=True) == 36
    assert len(first) == 36
    assert list(first.columns) == list(DATASET_COLUMNS)

    keys = ["epsilon", "n_respondents", "bias", "seed"]
    assert not first.duplicated(subset=keys).any()

    targets = first.loc[:, list(TARGET_COLUMNS)].to_numpy(dtype=float)
    assert np.all(np.isfinite(targets))
    assert targets.min() >= 0.0

    validate_selector_dataset(first, quick=True)
    pd.testing.assert_frame_equal(first, second)


@pytest.mark.skipif(
    not FULL_SELECTOR_DATASET_PATH.is_file(),
    reason="The full generated AI selector dataset is not present in this checkout.",
)
def test_full_generated_selector_dataset_matches_the_full_grid() -> None:
    dataset = pd.read_csv(FULL_SELECTOR_DATASET_PATH)

    assert expected_row_count(quick=False) == 1440
    assert len(dataset) == 1440
    assert list(dataset.columns) == list(DATASET_COLUMNS)

    validate_selector_dataset(dataset, quick=False)

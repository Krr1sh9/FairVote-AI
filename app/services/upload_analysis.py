"""Pure service-layer helpers for upload analysis orchestration.

This module intentionally keeps real respondent data separate from synthetic
experiment data.  A ``true_choice``/truth column is useful for simulations, but
it must not be silently used on real respondent exports.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

SYNTHETIC_TRUTH_COLUMN_NAMES = frozenset(
    {
        "true_choice",
        "truth",
        "true_answer",
        "actual_choice",
        "actual_answer",
        "unrandomized_answer",
    }
)


@dataclass(frozen=True)
class UploadRunLabels:
    """Column choices made for one uploaded analysis run."""

    response_col: str
    truth_col: str | None
    group_cols: tuple[str, ...]
    poststrat_cols: tuple[str, ...]
    synthetic_evaluation_mode: bool = False

    def has_synthetic_truth(self) -> bool:
        """Return whether the run includes synthetic truth for evaluation only."""
        return self.truth_col is not None and self.synthetic_evaluation_mode


def candidate_truth_columns(columns: Sequence[str]) -> list[str]:
    """Return columns whose names look like synthetic truth labels."""
    out: list[str] = []
    for col in columns:
        if str(col).strip().lower() in SYNTHETIC_TRUTH_COLUMN_NAMES:
            out.append(str(col))
    return out


def validate_truth_column_policy(
    *,
    truth_col: str | None,
    synthetic_evaluation_mode: bool,
) -> str | None:
    """Return the truth column if explicitly permitted; otherwise reject it.

    Real respondent exports should contain only perturbed answers and
    demographics.  Truth columns are therefore accepted only when the analyst has
    deliberately enabled synthetic-evaluation mode.
    """
    if truth_col in {None, "", "(none)"}:
        return None
    if not synthetic_evaluation_mode:
        raise ValueError(
            "Truth columns are allowed only in synthetic evaluation mode. "
            "Disable the truth column or explicitly mark the upload as synthetic."
        )
    return str(truth_col)

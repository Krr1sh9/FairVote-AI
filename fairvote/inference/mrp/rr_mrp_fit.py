"""Backward-compatible imports for the canonical linear RR-aware MRP path.

The implementation now lives in :mod:`fairvote.inference.mrp.linear`.  This file
is intentionally thin so older dashboard/import paths continue to work without
maintaining a second linear MRP implementation.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

from fairvote.inference.mrp.design import DesignMatrix
from fairvote.inference.mrp.linear import LinearRRMRPModel, MRPRRMultinomialModel


def fit_rr_mrp_from_rows(
    *,
    poll_rows: Sequence[Dict[str, str]],
    response_col: str,
    feature_cols: Sequence[str],
    epsilon: float,
    k: int,
    l2: float = 1.0,
    seed: int = 0,
    lr: float = 0.05,
    steps: int = 2000,
    batch_size: int = 512,
) -> Tuple[LinearRRMRPModel, DesignMatrix, np.ndarray, np.ndarray]:
    """Build a dashboard-style design matrix and fit canonical linear RR-MRP."""
    rows = list(poll_rows)
    design = DesignMatrix(feature_cols).fit(rows)
    X = design.transform(rows)
    y = np.array([int(float(str(row.get(response_col, "0")).strip())) for row in rows], dtype=int)
    model = LinearRRMRPModel(k=int(k), epsilon=float(epsilon), l2=float(l2), seed=int(seed))
    model.design_info = design
    model.fit(X, y, lr=float(lr), steps=int(steps), batch_size=int(batch_size))
    return model, design, X, y


__all__ = ["DesignMatrix", "LinearRRMRPModel", "MRPRRMultinomialModel", "fit_rr_mrp_from_rows"]

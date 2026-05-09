"""Metric and summary helpers for dashboard analyses."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


def fmt(x: float, ndp: int = 4) -> str:
    if x != x:
        return "nan"
    return f"{x:.{ndp}f}"


def group_metric_summary(
    group_rows: list[dict[str, Any]],
    metric_key: str,
    major_only: bool,
    major_mass: float,
) -> dict[str, float]:
    """Compute worst, p90, mass-weighted mean and max/min error ratio."""

    if not group_rows:
        return {"worst": float("nan"), "p90": float("nan"), "weighted": float("nan"), "error_ratio": float("nan")}

    rows: list[tuple[float, float]] = []
    for r in group_rows:
        mass = float(r.get("mass", 0.0))
        if major_only and mass < float(major_mass):
            continue
        v = float(r.get(metric_key, float("nan")))
        if not np.isfinite(v):
            continue
        rows.append((mass, v))

    if not rows:
        return {"worst": float("nan"), "p90": float("nan"), "weighted": float("nan"), "error_ratio": float("nan")}

    masses = np.asarray([m for m, _ in rows], dtype=float)
    vals = np.asarray([v for _, v in rows], dtype=float)

    worst = float(np.max(vals))
    p90 = float(np.quantile(vals, 0.90))
    wsum = float(np.sum(masses))
    weighted = float(np.sum(masses * vals) / wsum) if wsum > 0 else float("nan")

    min_err = float(np.min(vals))
    if min_err <= 1e-12:
        error_ratio = float("inf") if worst > 1e-12 else 1.0
    else:
        error_ratio = float(worst / min_err)
    if len(vals) < 2:
        error_ratio = float("nan")

    return {"worst": worst, "p90": p90, "weighted": weighted, "error_ratio": error_ratio}


def overall_metrics(p_hat: np.ndarray, p_true: Optional[np.ndarray]) -> dict[str, float]:
    if p_true is None:
        return {"overall_l1": float("nan"), "overall_mae": float("nan"), "correct_winner": 0.0}
    p_hat = np.asarray(p_hat, dtype=float)
    p_true = np.asarray(p_true, dtype=float)
    return {
        "overall_l1": float(np.sum(np.abs(p_hat - p_true))),
        "overall_mae": float(np.mean(np.abs(p_hat - p_true))),
        "correct_winner": float(np.argmax(p_hat) == np.argmax(p_true)),
    }


def overall_estimate_rows(
    labels: list[str],
    p_baseline: np.ndarray,
    *,
    p_lo: Optional[np.ndarray] = None,
    p_hi: Optional[np.ndarray] = None,
    p_true: Optional[np.ndarray] = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, lab in enumerate(labels):
        row: dict[str, Any] = {"category_id": i, "label": lab, "rr_debias_p": float(p_baseline[i])}
        if p_lo is not None and p_hi is not None:
            row["ci_low"] = float(p_lo[i])
            row["ci_high"] = float(p_hi[i])
        if p_true is not None:
            row["true_p"] = float(p_true[i])
        rows.append(row)
    return rows

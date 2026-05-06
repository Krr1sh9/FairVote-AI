"""Category encoding, grouping and post-stratification helpers for the dashboard."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.parsing.upload import category_index_from_value


@dataclass(frozen=True)
class CategoryMap:
    labels: list[str]
    to_int: dict[str, int]


def display_labels_for_categories(
    cmap: CategoryMap,
    *,
    option_labels: Sequence[str],
    use_poll_config: bool = True,
) -> tuple[list[str], bool]:
    """Return display labels without changing category encoding or calculations."""

    fallback = [str(label) for label in cmap.labels]
    if not use_poll_config or not option_labels:
        return fallback, False

    mapped: list[str] = []
    for pos, raw_label in enumerate(cmap.labels):
        raw = str(raw_label).strip()
        idx = pos if raw == f"(missing_{pos})" else category_index_from_value(raw_label)
        if idx is None or idx >= len(option_labels):
            return fallback, False
        mapped.append(str(option_labels[idx]))

    return mapped, True


def category_display_map(
    cmap: CategoryMap,
    display_labels: Sequence[str],
    option_labels: Sequence[str],
) -> dict[str, Any]:
    """Build a display-only lookup for category values and internal ids."""

    raw_to_display: dict[str, str] = {}
    id_to_display: dict[int, str] = {}

    for internal_id, raw_label in enumerate(cmap.labels):
        display = str(display_labels[internal_id]) if internal_id < len(display_labels) else str(raw_label)
        raw_to_display[str(raw_label).strip()] = display
        id_to_display[int(internal_id)] = display

        raw_idx = category_index_from_value(raw_label)
        if raw_idx is not None and raw_idx < len(option_labels):
            raw_to_display[str(raw_idx)] = str(option_labels[raw_idx])

    return {
        "raw": raw_to_display,
        "id": id_to_display,
        "option_labels": [str(label) for label in option_labels],
    }


def format_category_value(value: Any, category_display_map: dict[str, Any]) -> str:
    """Format one answer/category value for display without altering calculations."""

    raw = str(value).strip()
    raw_lookup = category_display_map.get("raw", {})
    if raw in raw_lookup:
        return str(raw_lookup[raw])

    idx = category_index_from_value(value)
    if idx is not None:
        option_labels = category_display_map.get("option_labels", [])
        if 0 <= idx < len(option_labels):
            return str(option_labels[idx])
        id_lookup = category_display_map.get("id", {})
        if idx in id_lookup:
            return str(id_lookup[idx])

    return raw


def format_group_key(
    group_key: Sequence[Any],
    group_cols: Sequence[str],
    category_display_map: dict[str, Any],
    answer_like_cols: set[str],
) -> str:
    """Format a group tuple, mapping answer-like components to party labels."""

    if not group_key:
        return "(all)"

    parts: list[str] = []
    for i, value in enumerate(group_key):
        col = str(group_cols[i]).strip().lower() if i < len(group_cols) else ""
        if col in answer_like_cols:
            parts.append(format_category_value(value, category_display_map))
        else:
            parts.append(str(value).strip())
    return " | ".join(parts)


def answer_like_columns(response_col: str, truth_col: str | None) -> set[str]:
    """Column names that should not be offered as learned-MRP features."""

    names = {
        "reported_choice",
        "true_choice",
        "stated_choice",
        "perturbed_answer",
        str(response_col).strip().lower(),
    }
    if truth_col is not None:
        names.add(str(truth_col).strip().lower())
    return {name for name in names if name}


def build_category_map(values: Sequence[str], *, k_override: int | None = None) -> CategoryMap:
    uniq = sorted({str(v).strip() for v in values if str(v).strip() != ""})
    labels = uniq.copy()

    if k_override is not None and int(k_override) > len(labels):
        for i in range(len(labels), int(k_override)):
            labels.append(f"(missing_{i})")

    to_int = {lab: i for i, lab in enumerate(labels)}
    return CategoryMap(labels=labels, to_int=to_int)


def encode_categories(values: Sequence[str], cmap: CategoryMap) -> np.ndarray:
    out = []
    for v in values:
        lab = str(v).strip()
        out.append(cmap.to_int.get(lab, -1))
    return np.asarray(out, dtype=int)


def group_keys(row: dict[str, str], cols: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(row.get(c, "")).strip() for c in cols)


def filter_valid(reported: np.ndarray, truth: np.ndarray | None) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    mask = reported >= 0
    rep = reported[mask]
    tru = truth[mask] if truth is not None else None
    return rep, tru, mask


def read_population_weights(
    rows: list[dict[str, str]], key_cols: Sequence[str], count_col: str
) -> dict[tuple[str, ...], float]:
    weights: dict[tuple[str, ...], float] = {}
    total = 0.0
    for r in rows:
        key = group_keys(r, key_cols)
        try:
            c = float(r.get(count_col, "nan"))
        except Exception:
            c = float("nan")
        if not np.isfinite(c) or c <= 0.0:
            continue
        weights[key] = weights.get(key, 0.0) + float(c)
        total += float(c)

    if total <= 0.0:
        return {}

    for k in list(weights.keys()):
        weights[k] = weights[k] / total
    return weights


def poststratify_from_groups(
    group_estimates: dict[tuple[str, ...], np.ndarray],
    pop_weights: dict[tuple[str, ...], float],
    *,
    fallback: np.ndarray,
) -> np.ndarray:
    k = int(fallback.size)
    out = np.zeros(k, dtype=float)
    total_w = 0.0

    for g, w in pop_weights.items():
        if w <= 0.0:
            continue
        p_g = group_estimates.get(g, fallback)
        out += float(w) * np.asarray(p_g, dtype=float)
        total_w += float(w)

    if total_w <= 0.0:
        return fallback.copy()

    out = out / total_w
    out = np.clip(out, 0.0, 1.0)
    s = float(out.sum())
    if s > 0.0:
        out /= s
    else:
        out = fallback.copy()
    return out


def normalised_mean_probability(P: np.ndarray) -> np.ndarray:
    p = np.mean(np.asarray(P, dtype=float), axis=0)
    p = np.clip(p, 0.0, 1.0)
    s = float(p.sum())
    if s > 0.0 and np.isfinite(s):
        p /= s
    return p


def poststratify_probabilities(P_pop: np.ndarray, weights: Sequence[float]) -> np.ndarray:
    P_pop = np.asarray(P_pop, dtype=float)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if P_pop.ndim != 2:
        raise ValueError("P_pop must be 2D")
    if w.size != P_pop.shape[0]:
        raise ValueError("weights must match population rows")
    if not np.all(np.isfinite(w)) or np.any(w < 0.0):
        raise ValueError("weights must be finite and non-negative")
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError("weights must have positive sum")
    w = w / total
    p = np.sum(P_pop * w[:, None], axis=0)
    p = np.clip(p, 0.0, 1.0)
    s = float(p.sum())
    if s > 0.0 and np.isfinite(s):
        p /= s
    return p


def parse_hidden_layers(text: str) -> tuple[int, ...]:
    """Parse a compact hidden-layer string such as '32,16'."""

    raw = str(text).strip()
    if not raw:
        raise ValueError("hidden layer sizes must not be empty")
    parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
    widths: list[int] = []
    for p in parts:
        try:
            width = int(p)
        except Exception as exc:
            raise ValueError("hidden layer sizes must be comma-separated integers, e.g. 32,16") from exc
        if width <= 0:
            raise ValueError("hidden layer sizes must be positive")
        widths.append(width)
    if not widths:
        raise ValueError("at least one hidden layer size is required")
    return tuple(widths)

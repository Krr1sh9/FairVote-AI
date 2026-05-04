# app/streamlit_app.py
"""
FairVote-AI Streamlit UI.

This file is intentionally the analyst-facing layer: it loads respondent or
synthetic data, calls the library estimators, and renders tables/plots. The
dashboard may map numeric category ids to party labels for readability, but
that mapping is display-only and must not change stored data or calculations.

Includes:
- Upload-a-poll workflow (real-world demo)
  - Baseline RR-debias estimate (+ optional bootstrap CI)
  - Group audit (by region/age/etc.)
  - Optional direct post-stratification (population CSV)
  - Optional RR-aware MRP (regularized multinomial regression with RR observation model)
  - Report-ready plots (PNG) + ZIP download
  - Fairness / worst-group dashboard (major-groups toggle)
  - One-click Results Bundle export (ZIP: plots + CSVs + markdown + metadata)

- Simulation runner (experiment pipeline)
  - Run experiments.mrp_vs_baselines from the UI
  - Browse existing run folders + view summary.csv

Run:
  pip install -e ".[dev,ai,streamlit,respondent]"
  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import csv
import importlib.util
import io
import json
import math
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import streamlit as st

# Optional plotting
_HAS_MPL = True
try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator

    # Clamp plot rendering to avoid huge PNGs (PIL decompression bomb)
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 150
    plt.rcParams["figure.figsize"] = (8, 4)
except Exception:
    _HAS_MPL = False
    plt = None  # type: ignore
    AutoMinorLocator = None  # type: ignore



# ---------------------------
# Optional imports from your project
# ---------------------------

_HAS_FAIRVOTE_PRIVACY = True
try:
    # Preferred: your project’s implementation
    from fairvote.privacy import estimate_distribution as fv_estimate_distribution
except Exception:
    _HAS_FAIRVOTE_PRIVACY = False
    fv_estimate_distribution = None  # type: ignore[assignment]

_HAS_RR_MRP = True
try:
    from fairvote.inference.mrp.rr_mrp_fit import MRPRRMultinomialModel, DesignMatrix
except Exception:
    _HAS_RR_MRP = False
    MRPRRMultinomialModel = None  # type: ignore[assignment]
    DesignMatrix = None  # type: ignore[assignment]

_HAS_MISREPORT_RR_MRP = True
try:
    from fairvote.inference.mrp.misreport_rr import MisreportRRMultinomialModel, shy_misreport_matrix
except Exception:
    _HAS_MISREPORT_RR_MRP = False
    MisreportRRMultinomialModel = None  # type: ignore[assignment]
    shy_misreport_matrix = None  # type: ignore[assignment]


def _torch_available() -> bool:
    """Return whether PyTorch can be found without importing it eagerly."""

    return importlib.util.find_spec("torch") is not None


def _load_neural_mrp_model():
    """Lazily import the PyTorch neural MRP model only when the user selects it."""

    from fairvote.inference.mrp.rr_neural_mrp import RRNeuralMRPModel

    return RRNeuralMRPModel


# ---------------------------
# Helpers: file IO + parsing
# ---------------------------

def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_uploaded_csv(uploaded_file) -> List[Dict[str, str]]:
    """Read an uploaded CSV into string-valued row dictionaries.

    Uploads are analysis inputs for the dashboard. Evaluation-only columns such
    as true_choice may be present in synthetic CSVs but are not part of the
    respondent-server storage contract.
    """
    raw = uploaded_file.getvalue()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")

    f = io.StringIO(text)
    reader = csv.DictReader(f)
    rows: List[Dict[str, str]] = []
    for r in reader:
        rows.append({k: (v if v is not None else "") for k, v in r.items()})
    return rows


def _read_uploaded_jsonl(uploaded_file) -> List[Dict[str, str]]:
    """Read respondent JSONL and flatten demographics for dashboard use.

    The respondent export stores numeric randomized reports. Any party-name
    mapping is applied later for display only.
    """
    raw = uploaded_file.getvalue()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    
    rows: List[Dict[str, str]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            flat = {}
            for k, v in record.items():
                if k == "demographics" and isinstance(v, dict):
                    for dk, dv in v.items():
                        flat[dk] = str(dv)
                else:
                    flat[k] = str(v)
            rows.append(flat)
        except json.JSONDecodeError:
            pass
    return rows


def _columns(rows: List[Dict[str, str]]) -> List[str]:
    if not rows:
        return []
    return list(rows[0].keys())


def _fmt(x: float, ndp: int = 4) -> str:
    if x != x:
        return "nan"
    return f"{x:.{ndp}f}"


def _find_best_col(cols: List[str], candidates: Sequence[str]) -> int:
    lower = [c.lower() for c in cols]
    for cand in candidates:
        cand_l = cand.lower()
        for i, c in enumerate(lower):
            if c == cand_l:
                return i
    for cand in candidates:
        cand_l = cand.lower()
        for i, c in enumerate(lower):
            if cand_l in c:
                return i
    return 0


def _valid_multiselect_defaults(defaults: Sequence[str], options: Sequence[str]) -> List[str]:
    """Return only defaults that Streamlit can accept for the current options."""

    option_set = set(options)
    return [value for value in defaults if value in option_set]


def _load_poll_option_labels(root: Path) -> List[str]:
    """Load respondent poll option labels for display-only dashboard labelling.

    This reads the public poll configuration, not respondent records. Failure
    is treated as absence of labels so uploads still work with numeric ids.
    """

    config_path = root / "respondent" / "poll_config.json"
    if not config_path.exists():
        return []

    try:
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception:
        return []

    options = config.get("options")
    if not isinstance(options, list):
        return []

    labels = [str(opt).strip() for opt in options]
    return [lab for lab in labels if lab]


def _category_index_from_value(value: Any) -> Optional[int]:
    """Parse a dashboard category value as a non-negative integer index, if safe."""

    if value is None or isinstance(value, (bool, np.bool_)):
        return None

    if isinstance(value, (int, np.integer)):
        idx = int(value)
        return idx if idx >= 0 else None

    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value) or not float(value).is_integer():
            return None
        idx = int(value)
        return idx if idx >= 0 else None

    raw = str(value).strip()
    if raw == "":
        return None

    try:
        as_float = float(raw)
    except Exception:
        return None

    if not np.isfinite(as_float) or not as_float.is_integer():
        return None

    idx = int(as_float)
    return idx if idx >= 0 else None


def _is_numeric_category_value(value: Any) -> bool:
    """Return whether a value can safely be treated as a category index."""

    return _category_index_from_value(value) is not None


def _display_labels_for_categories(
    cmap: "CategoryMap",
    *,
    option_labels: Sequence[str],
    use_poll_config: bool = True,
) -> Tuple[List[str], bool]:
    """Return display labels without changing category encoding or calculations.

    If the categories are numeric IDs and respondent/poll_config.json contains
    matching option names, use those option names for display only.  String
    labels already present in the uploaded data are preserved.
    """

    fallback = [str(label) for label in cmap.labels]
    if not use_poll_config or not option_labels:
        return fallback, False

    # Numeric synthetic columns and real JSONL exports can both arrive as
    # 0..k-1 ids. The internal CategoryMap is left unchanged; only the text
    # presented in plots/tables is substituted.
    mapped: List[str] = []
    for pos, raw_label in enumerate(cmap.labels):
        raw = str(raw_label).strip()
        idx = pos if raw == f"(missing_{pos})" else _category_index_from_value(raw_label)
        if idx is None or idx >= len(option_labels):
            return fallback, False
        mapped.append(str(option_labels[idx]))

    return mapped, True


def _category_display_map(
    cmap: "CategoryMap",
    display_labels: Sequence[str],
    option_labels: Sequence[str],
) -> Dict[str, Any]:
    """Build a display-only lookup for category values and internal ids."""

    raw_to_display: Dict[str, str] = {}
    id_to_display: Dict[int, str] = {}

    for internal_id, raw_label in enumerate(cmap.labels):
        display = str(display_labels[internal_id]) if internal_id < len(display_labels) else str(raw_label)
        raw_to_display[str(raw_label).strip()] = display
        id_to_display[int(internal_id)] = display

        raw_idx = _category_index_from_value(raw_label)
        if raw_idx is not None and raw_idx < len(option_labels):
            raw_to_display[str(raw_idx)] = str(option_labels[raw_idx])

    return {
        "raw": raw_to_display,
        "id": id_to_display,
        "option_labels": [str(label) for label in option_labels],
    }


def _format_category_value(value: Any, category_display_map: Dict[str, Any]) -> str:
    """Format one answer/category value for display without altering calculations."""

    raw = str(value).strip()
    raw_lookup = category_display_map.get("raw", {})
    if raw in raw_lookup:
        return str(raw_lookup[raw])

    idx = _category_index_from_value(value)
    if idx is not None:
        option_labels = category_display_map.get("option_labels", [])
        if 0 <= idx < len(option_labels):
            return str(option_labels[idx])
        id_lookup = category_display_map.get("id", {})
        if idx in id_lookup:
            return str(id_lookup[idx])

    return raw


def _format_group_key(
    group_key: Sequence[Any],
    group_cols: Sequence[str],
    category_display_map: Dict[str, Any],
    answer_like_cols: set[str],
) -> str:
    """Format a group tuple, mapping answer-like components to party labels."""

    if not group_key:
        return "(all)"

    parts: List[str] = []
    for i, value in enumerate(group_key):
        col = str(group_cols[i]).strip().lower() if i < len(group_cols) else ""
        if col in answer_like_cols:
            parts.append(_format_category_value(value, category_display_map))
        else:
            parts.append(str(value).strip())
    return " | ".join(parts)


def _answer_like_columns(response_col: str, truth_col: Optional[str]) -> set[str]:
    """Column names that should not be offered as learned-MRP features.

    Answer columns may be selected for auditing/display, but using them as MRP
    covariates would leak target information into the model design matrix.
    """

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



# ---------------------------
# Randomized Response (RR) debias (fallback)
# ---------------------------

def rr_matrix(epsilon: float, k: int) -> np.ndarray:
    eps = float(epsilon)
    k = int(k)
    p_keep = math.exp(eps) / (math.exp(eps) + (k - 1))
    p_flip = (1.0 - p_keep) / (k - 1)

    A = np.full((k, k), p_flip, dtype=float)
    np.fill_diagonal(A, p_keep)
    return A


def rr_debias_from_reported(reported: np.ndarray, epsilon: float, k: int) -> np.ndarray:
    reported = np.asarray(reported, dtype=int).reshape(-1)
    n = int(reported.size)
    if n == 0:
        return np.full(k, 1.0 / k, dtype=float)

    counts = np.bincount(reported, minlength=k).astype(float)
    q = counts / max(1.0, float(n))

    A = rr_matrix(epsilon, k)
    # q = A^T p  => p = (A^T)^-1 q
    p_hat = np.linalg.solve(A.T, q)

    p_hat = np.clip(p_hat, 0.0, 1.0)
    s = float(p_hat.sum())
    if s <= 0.0 or not np.isfinite(s):
        p_hat = np.full(k, 1.0 / k, dtype=float)
    else:
        p_hat = p_hat / s
    return p_hat


def estimate_distribution(reported: np.ndarray, epsilon: float, k: int) -> np.ndarray:
    """
    Preferred path: fairvote.privacy.estimate_distribution
    Fallback: closed-form RR debias with solve.
    """
    if _HAS_FAIRVOTE_PRIVACY and fv_estimate_distribution is not None:
        out = fv_estimate_distribution(reported, epsilon=epsilon, k=k)
        return np.asarray(out, dtype=float)
    return rr_debias_from_reported(reported, epsilon=epsilon, k=k)


def bootstrap_ci(
    reported: np.ndarray,
    epsilon: float,
    k: int,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    reported = np.asarray(reported, dtype=int).reshape(-1)
    n = int(reported.size)
    if n == 0:
        lo = np.full(k, float("nan"))
        hi = np.full(k, float("nan"))
        return lo, hi

    boot = np.empty((n_boot, k), dtype=float)
    for b in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        boot[b] = estimate_distribution(reported[idx], epsilon=epsilon, k=k)

    lo = np.quantile(boot, alpha / 2.0, axis=0)
    hi = np.quantile(boot, 1.0 - alpha / 2.0, axis=0)
    return lo, hi


# ---------------------------
# Category mapping + grouping
# ---------------------------

@dataclass
class CategoryMap:
    labels: List[str]
    to_int: Dict[str, int]


def build_category_map(values: Sequence[str], *, k_override: Optional[int] = None) -> CategoryMap:
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


def group_keys(row: Dict[str, str], cols: Sequence[str]) -> Tuple[str, ...]:
    return tuple(str(row.get(c, "")).strip() for c in cols)


def filter_valid(reported: np.ndarray, truth: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    mask = reported >= 0
    rep = reported[mask]
    tru = truth[mask] if truth is not None else None
    return rep, tru, mask


# ---------------------------
# Post-strat (direct) using population table
# ---------------------------

def read_population_weights(rows: List[Dict[str, str]], key_cols: Sequence[str], count_col: str) -> Dict[Tuple[str, ...], float]:
    weights: Dict[Tuple[str, ...], float] = {}
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
    group_estimates: Dict[Tuple[str, ...], np.ndarray],
    pop_weights: Dict[Tuple[str, ...], float],
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


# ---------------------------
# Plotting helpers
# ---------------------------

def _fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    return buf.getvalue()


def _apply_readable_grid(ax, orientation: str = "vertical") -> None:
    """Apply subtle, readable major/minor grids without changing plotted values.

    This helper is cosmetic only; it does not change estimates, fairness metrics,
    or exported numerical results.
    """

    ax.set_axisbelow(True)

    if orientation == "horizontal":
        # Horizontal bar chart: the numeric axis is x.
        if AutoMinorLocator is not None:
            try:
                ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            except Exception:
                pass
        ax.grid(True, which="major", axis="x", alpha=0.35, linewidth=0.8)
        ax.grid(True, which="minor", axis="x", alpha=0.18, linewidth=0.5)
        ax.grid(True, which="major", axis="y", alpha=0.12, linewidth=0.5)
    elif orientation == "both":
        if AutoMinorLocator is not None:
            try:
                ax.xaxis.set_minor_locator(AutoMinorLocator(2))
                ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            except Exception:
                pass
        ax.grid(True, which="major", axis="both", alpha=0.30, linewidth=0.8)
        ax.grid(True, which="minor", axis="both", alpha=0.15, linewidth=0.5)
    else:
        # Vertical bar chart: the numeric axis is y.
        if AutoMinorLocator is not None:
            try:
                ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            except Exception:
                pass
        ax.grid(True, which="major", axis="y", alpha=0.35, linewidth=0.8)
        ax.grid(True, which="minor", axis="y", alpha=0.18, linewidth=0.5)
        ax.grid(True, which="major", axis="x", alpha=0.12, linewidth=0.5)


def _plot_overall_distributions(labels: Sequence[str], series: Sequence[Tuple[str, np.ndarray]], title: str) -> Optional[bytes]:
    if not _HAS_MPL or plt is None:
        return None

    labels = list(labels)
    k = len(labels)
    if k == 0 or not series:
        return None

    series = [(name, np.asarray(p, dtype=float).reshape(-1)) for name, p in series]

    fig, ax = plt.subplots(figsize=(max(6, 0.6 * k), 3.8))
    x = np.arange(k)
    width = 0.8 / max(1, len(series))

    for j, (name, p) in enumerate(series):
        p = np.clip(p, 0.0, 1.0)
        ax.bar(x + (j - (len(series) - 1) / 2) * width, p, width, label=name)

    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Estimated vote share")
    _apply_readable_grid(ax, orientation="vertical")
    ax.legend()

    png = _fig_to_png_bytes(fig)
    plt.close(fig)
    return png


def _plot_group_bars(group_rows: List[Dict[str, Any]], title: str, metric_key: str, top_n: int = 20) -> Optional[bytes]:
    if not _HAS_MPL or plt is None or not group_rows:
        return None

    rows = sorted(group_rows, key=lambda r: float(r.get("mass", 0.0)), reverse=True)[: int(top_n)]
    groups = [str(r.get("group", "")) for r in rows]
    vals = [float(r.get(metric_key, float("nan"))) for r in rows]

    fig, ax = plt.subplots(figsize=(10, max(3.6, 0.32 * len(groups))))
    y = np.arange(len(groups))
    ax.barh(y, vals)
    ax.set_yticks(y)
    ax.set_yticklabels(groups)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(metric_key)
    _apply_readable_grid(ax, orientation="horizontal")

    png = _fig_to_png_bytes(fig)
    plt.close(fig)
    return png


# ---------------------------
# Group metric utilities (fairness dashboard)
# ---------------------------

def _group_metric_summary(
    group_rows: List[Dict[str, Any]],
    metric_key: str,
    major_only: bool,
    major_mass: float,
) -> Dict[str, float]:
    """
    Computes summary stats over groups for a given metric_key in each row:
    - worst (max)
    - p90 (90th percentile)
    - weighted mean (mass-weighted)
    - error ratio (max/min disparity)
    """
    if not group_rows:
        return {"worst": float("nan"), "p90": float("nan"), "weighted": float("nan"), "error_ratio": float("nan")}

    rows = []
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


def _overall_metrics(p_hat: np.ndarray, p_true: Optional[np.ndarray]) -> Dict[str, float]:
    if p_true is None:
        return {"overall_l1": float("nan"), "overall_mae": float("nan"), "correct_winner": 0.0}
    p_hat = np.asarray(p_hat, dtype=float)
    p_true = np.asarray(p_true, dtype=float)
    return {
        "overall_l1": float(np.sum(np.abs(p_hat - p_true))),
        "overall_mae": float(np.mean(np.abs(p_hat - p_true))),
        "correct_winner": float(np.argmax(p_hat) == np.argmax(p_true)),
    }


LEARNED_MRP_METHODS = {
    "Linear RR-aware MRP",
    "Neural RR-aware MRP",
    "Misreport-aware RR-MRP",
}


def _is_learned_mrp_method(method: str) -> bool:
    return str(method) in LEARNED_MRP_METHODS


def _method_prefix(method: str) -> str:
    if method == "Neural RR-aware MRP":
        return "neural_mrp"
    if method == "Misreport-aware RR-MRP":
        return "misreport_mrp"
    if method == "Linear RR-aware MRP":
        return "linear_mrp"
    return "baseline"


def _method_short_label(method: str) -> str:
    if method == "Neural RR-aware MRP":
        return "Neural RR-aware MRP"
    if method == "Misreport-aware RR-MRP":
        return "Misreport-aware RR-MRP"
    if method == "Linear RR-aware MRP":
        return "Linear RR-aware MRP"
    return "RR debiasing"


def _parse_hidden_layers(text: str) -> Tuple[int, ...]:
    """Parse a compact hidden-layer string such as '32,16'."""

    raw = str(text).strip()
    if not raw:
        raise ValueError("hidden layer sizes must not be empty")
    parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
    widths: List[int] = []
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


def _normalised_mean_probability(P: np.ndarray) -> np.ndarray:
    p = np.mean(np.asarray(P, dtype=float), axis=0)
    p = np.clip(p, 0.0, 1.0)
    s = float(p.sum())
    if s > 0.0 and np.isfinite(s):
        p /= s
    return p


def _poststratify_probabilities(P_pop: np.ndarray, weights: Sequence[float]) -> np.ndarray:
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


# ---------------------------
# UI
# ---------------------------

st.set_page_config(page_title="FairVote-AI", layout="wide")
root = _project_root()
outputs_dir = root / "experiments" / "outputs"
outputs_dir.mkdir(parents=True, exist_ok=True)

st.title("FairVote-AI")

tabs = st.tabs(["Upload & Estimate", "Scenario Simulator", "Simulation & Runs", "Recommendations", "About"])


# ===========================
# Tab 1: Upload & Estimate
# ===========================

with tabs[0]:
    st.subheader("Upload a poll CSV and estimate under randomized response")

    if not _HAS_MPL:
        st.warning("matplotlib not installed. Install it to enable plots: pip install matplotlib")

    col1, col2 = st.columns([1, 1])
    with col1:
        poll_file = st.file_uploader("Poll CSV or JSONL (responses)", type=["csv", "jsonl"], key="poll_file")
    with col2:
        pop_file = st.file_uploader("Population CSV (optional, for post-stratification)", type=["csv"], key="pop_csv")

    poll_rows: List[Dict[str, str]] = []
    pop_rows: List[Dict[str, str]] = []

    is_jsonl_upload = False
    if poll_file is not None:
        try:
            if poll_file.name.endswith(".jsonl"):
                is_jsonl_upload = True
                poll_rows = _read_uploaded_jsonl(poll_file)
                st.success(f"Loaded poll JSONL: {len(poll_rows)} rows, {len(_columns(poll_rows))} columns")
            else:
                poll_rows = _read_uploaded_csv(poll_file)
                st.success(f"Loaded poll CSV: {len(poll_rows)} rows, {len(_columns(poll_rows))} columns")
        except Exception as e:
            st.error(f"Failed to read poll file: {e}")

    if pop_file is not None:
        try:
            pop_rows = _read_uploaded_csv(pop_file)
            st.success(f"Loaded population CSV: {len(pop_rows)} rows, {len(_columns(pop_rows))} columns")
        except Exception as e:
            st.error(f"Failed to read population CSV: {e}")

    if not poll_rows:
        st.info("Upload a poll CSV or JSONL to begin.")
    else:
        cols = _columns(poll_rows)

        # ---------- Sidebar config ----------
        with st.sidebar:
            st.header("Poll configuration")

            response_col = st.selectbox(
                "Reported choice column (privatised)",
                options=cols,
                index=_find_best_col(cols, ["perturbed_answer", "reported_choice", "reported", "response", "vote", "choice"]),
            )

            truth_col = st.selectbox(
                "Optional: true choice column (only for evaluation / synthetic data)",
                options=["(none)"] + cols,
                index=(1 + [c.lower() for c in cols].index("true_choice")) if any(c.lower() == "true_choice" for c in cols) else 0,
            )
            truth_col = None if truth_col == "(none)" else truth_col

            st.divider()

            st.subheader("Method")
            method_options = ["RR debiasing", "Linear RR-aware MRP", "Neural RR-aware MRP"]
            if _HAS_MISREPORT_RR_MRP:
                method_options.append("Misreport-aware RR-MRP")
            method = st.radio(
                "Estimation method",
                options=method_options,
                index=0,
                help=(
                    "RR debiasing estimates aggregate shares directly. Learned MRP methods fit "
                    "P(true choice | demographics) while accounting for the RR observation process."
                ),
            )
            if method == "Linear RR-aware MRP" and not _HAS_RR_MRP:
                st.warning("Linear MRP module not found. Falling back to RR debiasing.")
                method = "RR debiasing"
            if method == "Misreport-aware RR-MRP" and not _HAS_MISREPORT_RR_MRP:
                st.warning("Misreport-aware MRP module not found. Falling back to RR debiasing.")
                method = "RR debiasing"
            if method == "Neural RR-aware MRP" and not _torch_available():
                st.warning('PyTorch is not installed, so Neural RR-aware MRP cannot run. Install with `pip install -e ".[dev,ai,streamlit,respondent]"`, or use a non-neural estimator. Falling back to RR debiasing.')
                method = "RR debiasing"
            if method == "Neural RR-aware MRP":
                st.warning(
                    "Neural MRP is a learned model. Compare it against RR debiasing and linear MRP; "
                    "it can overfit or underperform on small/noisy privatized samples."
                )

            st.divider()

            # epsilon default: if column exists, try first non-empty value
            eps_default = 1.0
            if any(c.lower() == "epsilon" for c in cols):
                for r in poll_rows:
                    v = str(r.get("epsilon", "")).strip()
                    if v:
                        try:
                            eps_default = float(v)
                            break
                        except Exception:
                            pass

            st.subheader("Privacy")
            epsilon = st.number_input("epsilon", min_value=0.01, max_value=10.0, value=float(eps_default), step=0.05)

            st.divider()

            st.subheader("Categories (k)")
            raw_labels = [r.get(response_col, "") for r in poll_rows]
            uniq_labels = sorted({str(v).strip() for v in raw_labels if str(v).strip() != ""})
            st.caption(f"Unique labels in reported column: {len(uniq_labels)}")
            if len(uniq_labels) > 50 and len(uniq_labels) > 0.2 * len(poll_rows):
                st.warning("Selected response column has very high cardinality. This often means you picked an ID column.")
            k_override_val = st.number_input("Override k (optional)", min_value=0, max_value=200, value=0, step=1)
            k_override = None if int(k_override_val) <= 0 else int(k_override_val)
            sidebar_cmap = build_category_map(raw_labels, k_override=k_override)
            sidebar_display_labels, _ = _display_labels_for_categories(
                sidebar_cmap,
                option_labels=_load_poll_option_labels(root),
                use_poll_config=True,
            )

            st.divider()

            st.subheader("Audit settings")
            group_options = [c for c in cols if c != response_col]
            default_groups = [c for c in ["region", "age_band"] if c in group_options]
            group_cols = st.multiselect(
                "Group columns for auditing (e.g., region, age_band)",
                options=group_options,
                default=_valid_multiselect_defaults(default_groups, group_options),
            )
            major_mass = st.number_input("Major group mass threshold", min_value=0.0, max_value=1.0, value=0.02, step=0.01)
            n_boot = st.number_input("Bootstrap resamples (baseline only; 0 disables)", min_value=0, max_value=5000, value=300, step=50)
            boot_seed = st.number_input("Bootstrap seed", min_value=0, max_value=10_000_000, value=123, step=1)

            st.divider()

            st.subheader("Post-stratification (optional)")
            if pop_rows:
                pop_cols = _columns(pop_rows)
                post_options = [c for c in pop_cols if c in cols]
                default_post = [c for c in group_cols if c in pop_cols]
                post_cols = st.multiselect(
                    "Post-strat key columns (must exist in BOTH poll and population CSV)",
                    options=post_options,
                    default=_valid_multiselect_defaults(default_post, post_options),
                )
                count_col = st.selectbox("Population count column", options=pop_cols, index=_find_best_col(pop_cols, ["count", "n", "pop", "population"]))
            else:
                post_cols = []
                count_col = None

            if _is_learned_mrp_method(method):
                st.divider()
                st.subheader("Learned MRP settings")
                answer_like_cols = _answer_like_columns(response_col, truth_col)
                mrp_feature_options = [c for c in cols if c.strip().lower() not in answer_like_cols]
                mrp_default_cols = group_cols if group_cols else default_groups
                mrp_feature_cols = st.multiselect(
                    "Feature columns (categorical demographics)",
                    options=mrp_feature_options,
                    default=_valid_multiselect_defaults(mrp_default_cols, mrp_feature_options),
                    help=(
                        "Learned MRP predicts latent true choice from these features, then trains through "
                        "the randomized-response observation model using only reported labels."
                    ),
                )
                mrp_lr = st.number_input("Learning rate", min_value=0.0001, max_value=1.0, value=0.02 if method == "Neural RR-aware MRP" else 0.05, step=0.005)
                mrp_steps = st.number_input("Training steps", min_value=25, max_value=20000, value=500 if method == "Neural RR-aware MRP" else 2000, step=25)
                mrp_batch = st.number_input("Batch size", min_value=16, max_value=8192, value=512, step=64)
                mrp_seed = st.number_input("Model seed", min_value=0, max_value=10_000_000, value=0, step=1)

                if method == "Linear RR-aware MRP":
                    mrp_l2 = st.number_input("L2 regularization", min_value=0.0, max_value=100.0, value=1.0, step=0.5)

                if method == "Misreport-aware RR-MRP":
                    mrp_l2 = st.number_input("L2 regularization", min_value=0.0, max_value=100.0, value=1.0, step=0.5)
                    shy_category_options = list(range(len(sidebar_cmap.labels))) if sidebar_cmap.labels else [0]

                    def _format_shy_category(idx: int) -> str:
                        raw = sidebar_cmap.labels[idx] if 0 <= idx < len(sidebar_cmap.labels) else str(idx)
                        disp = sidebar_display_labels[idx] if 0 <= idx < len(sidebar_display_labels) else raw
                        return disp if disp == raw else f"{disp} ({raw})"

                    misreport_shy_category = st.selectbox(
                        "Shy/misreported category",
                        options=shy_category_options,
                        index=min(1, len(shy_category_options) - 1),
                        format_func=_format_shy_category,
                    )
                    misreport_honesty = st.slider(
                        "Honesty for that category before RR",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.8,
                        step=0.01,
                        help="Only this simple shy-voter misreport model is exposed in the dashboard.",
                    )

                if method == "Neural RR-aware MRP":
                    neural_size = st.selectbox(
                        "Neural model size",
                        options=["Small: 16", "Medium: 32,16", "Custom"],
                        index=0,
                        help="Keep this small unless you have enough respondents. Larger networks can overfit RR noise.",
                    )
                    if neural_size == "Small: 16":
                        neural_hidden_layers_text = "16"
                    elif neural_size == "Medium: 32,16":
                        neural_hidden_layers_text = "32,16"
                    else:
                        neural_hidden_layers_text = st.text_input("Hidden layer sizes", value="32,16", help="Comma-separated widths, e.g. 64,32")
                    neural_dropout = st.slider("Dropout", min_value=0.0, max_value=0.8, value=0.0, step=0.05)
                    neural_weight_decay = st.number_input("Weight decay", min_value=0.0, max_value=10.0, value=0.0001, step=0.0001, format="%.5f")

        # ---------- Build category map ----------
        raw_labels = [r.get(response_col, "") for r in poll_rows]
        cmap = build_category_map(raw_labels, k_override=k_override)
        k = len(cmap.labels)
        option_labels = _load_poll_option_labels(root)
        display_labels, using_poll_option_labels = _display_labels_for_categories(
            cmap,
            option_labels=option_labels,
            use_poll_config=True,
        )
        category_display_map = _category_display_map(cmap, display_labels, option_labels)
        display_answer_like_cols = _answer_like_columns(response_col, truth_col)

        reported_raw = [r.get(response_col, "") for r in poll_rows]
        reported = encode_categories(reported_raw, cmap)

        truth = None
        if truth_col is not None:
            truth_raw = [r.get(truth_col, "") for r in poll_rows]
            truth = encode_categories(truth_raw, cmap)

        reported, truth, valid_mask = filter_valid(reported, truth)
        n = int(reported.size)

        if n == 0:
            st.error("No valid rows after category mapping. Check your reported choice column.")
        else:
            valid_indices = np.where(valid_mask)[0].tolist()
            poll_rows_valid = [poll_rows[i] for i in valid_indices]

            # ---------- Overall baseline estimate ----------
            # Validate k (number of categories)
            try:
                k_int = int(k)
            except Exception:
                k_int = 0
            if k_int < 2:
                st.error(
                    "k must be >= 2. Your current settings/data imply only 0–1 categories. "
                    "Pick the correct 'Reported choice' column (not respondent_id), or set 'Override k' to 2+."
                )
                st.stop()
            k = k_int

            p_baseline = estimate_distribution(reported, epsilon=float(epsilon), k=int(k))

            # Truth distribution (if available)
            p_true = None
            if truth is not None and truth.size == n:
                truth_arr = np.asarray(truth, dtype=int).reshape(-1)
                bad_truth = int(np.sum(truth_arr < 0))
                if bad_truth > 0:
                    st.warning(
                        f"Truth column contains {bad_truth} unmapped/invalid values (likely wrong reported-choice column "
                        f"or label mismatch). Skipping truth-based accuracy metrics."
                    )
                else:
                    p_true = np.bincount(truth_arr, minlength=k).astype(float)
                    denom = float(p_true.sum()) if float(p_true.sum()) > 0 else 1.0
                    p_true = p_true / denom
            # Bootstrap CI (baseline only)
            p_lo, p_hi = (None, None)
            if int(n_boot) > 0:
                p_lo, p_hi = bootstrap_ci(reported, epsilon=float(epsilon), k=int(k), n_boot=int(n_boot), seed=int(boot_seed))

            st.subheader("Overall estimate")
            if using_poll_option_labels and is_jsonl_upload:
                st.caption(
                    "Real respondent exports store randomized category indices; "
                    "the dashboard maps them to poll option labels for display only. "
                    "Underlying data and calculations remain numeric."
                )
            table_rows = []
            for i, lab in enumerate(display_labels):
                row = {"category_id": i, "label": lab, "rr_debias_p": float(p_baseline[i])}
                if p_lo is not None and p_hi is not None:
                    row["ci_low"] = float(p_lo[i])
                    row["ci_high"] = float(p_hi[i])
                if p_true is not None:
                    row["true_p"] = float(p_true[i])
                table_rows.append(row)
            st.dataframe(table_rows, use_container_width=True)

            # ---------- Group audit (baseline) ----------
            group_rows: List[Dict[str, Any]] = []
            group_estimates: Dict[Tuple[str, ...], np.ndarray] = {}
            group_true: Dict[Tuple[str, ...], np.ndarray] = {}

            if group_cols:
                st.subheader("Group audit (baseline)")

                group_to_idx: Dict[Tuple[str, ...], List[int]] = {}
                for pos, original_i in enumerate(valid_indices):
                    key = group_keys(poll_rows[int(original_i)], group_cols)
                    group_to_idx.setdefault(key, []).append(pos)

                for g, idxs in group_to_idx.items():
                    idx_arr = np.asarray(idxs, dtype=int)
                    rep_g = reported[idx_arr]
                    p_g = estimate_distribution(rep_g, epsilon=float(epsilon), k=int(k))
                    group_estimates[g] = p_g

                    mass = float(rep_g.size) / float(n)
                    major = mass >= float(major_mass)

                    # If truth exists, compute group truth distribution and L1 error
                    l1_err = float("nan")
                    if p_true is not None and truth is not None:
                        tru_g = truth[idx_arr]
                        p_true_g = np.bincount(tru_g, minlength=k).astype(float) / max(1.0, float(tru_g.size))
                        group_true[g] = p_true_g
                        l1_err = float(np.sum(np.abs(p_g - p_true_g)))

                    key_str = _format_group_key(g, group_cols, category_display_map, display_answer_like_cols)
                    group_rows.append(
                        {
                            "group": key_str,
                            "n": int(rep_g.size),
                            "mass": mass,
                            "major": bool(major),
                            "baseline_l1": l1_err,
                        }
                    )

                group_rows.sort(key=lambda r: float(r.get("mass", 0.0)), reverse=True)
                st.dataframe(group_rows, use_container_width=True)

            # ---------- Direct post-strat (baseline) ----------
            p_post_direct = None
            pop_weights = None
            if pop_rows and post_cols and count_col is not None:
                pop_weights = read_population_weights(pop_rows, post_cols, str(count_col))
                if pop_weights:
                    # Need estimates per post-strat group keys:
                    if list(post_cols) == list(group_cols) and group_estimates:
                        post_est = group_estimates
                    else:
                        post_to_idx: Dict[Tuple[str, ...], List[int]] = {}
                        for pos, original_i in enumerate(valid_indices):
                            key = group_keys(poll_rows[int(original_i)], post_cols)
                            post_to_idx.setdefault(key, []).append(pos)

                        post_est: Dict[Tuple[str, ...], np.ndarray] = {}
                        for g, idxs in post_to_idx.items():
                            rep_g = reported[np.asarray(idxs, dtype=int)]
                            post_est[g] = estimate_distribution(rep_g, epsilon=float(epsilon), k=int(k))

                    p_post_direct = poststratify_from_groups(post_est, pop_weights, fallback=p_baseline)

                    st.subheader("Post-stratified estimate (direct baseline)")
                    post_rows = [{"category_id": i, "label": lab, "poststrat_p": float(p_post_direct[i])} for i, lab in enumerate(display_labels)]
                    st.dataframe(post_rows, use_container_width=True)

            # ---------- Learned RR-aware MRP methods (optional) ----------
            p_mrp_post = None
            p_mrp_sample = None
            group_rows_mrp: Optional[List[Dict[str, Any]]] = None  # learned-model fairness comparisons
            learned_method_label = _method_short_label(method)
            learned_prefix = _method_prefix(method)
            learned_l1_key = f"{learned_prefix}_l1"
            learned_sample_col = f"{learned_prefix}_sample_p"
            learned_post_col = f"{learned_prefix}_poststrat_p"

            if _is_learned_mrp_method(method):
                st.subheader(learned_method_label)
                if not _HAS_RR_MRP or DesignMatrix is None:
                    st.warning("Design-matrix support is unavailable, so learned MRP cannot run.")
                elif "mrp_feature_cols" not in locals() or not mrp_feature_cols:
                    st.error("Select at least one feature column for learned MRP in the sidebar.")
                else:
                    design = DesignMatrix(mrp_feature_cols).fit(poll_rows_valid)
                    X = design.transform(poll_rows_valid)
                    P_true = None

                    try:
                        if method == "Linear RR-aware MRP":
                            model = MRPRRMultinomialModel(k=int(k), epsilon=float(epsilon), l2=float(mrp_l2), seed=int(mrp_seed))
                            with st.spinner("Fitting linear RR-aware MRP model..."):
                                info = model.fit(
                                    X,
                                    reported,
                                    lr=float(mrp_lr),
                                    steps=int(mrp_steps),
                                    batch_size=int(mrp_batch),
                                    verbose_every=0,
                                    keep_history=False,
                                )
                            st.write(f"Fitted linear RR-aware MRP: steps={info.steps}, final_loss={_fmt(info.final_loss, 6)}")
                            P_true = model.predict_true_proba(X)

                        elif method == "Neural RR-aware MRP":
                            hidden_layers = _parse_hidden_layers(neural_hidden_layers_text)
                            RRNeuralMRPModel = _load_neural_mrp_model()
                            model = RRNeuralMRPModel(
                                k=int(k),
                                epsilon=float(epsilon),
                                hidden_layers=hidden_layers,
                                dropout=float(neural_dropout),
                                weight_decay=float(neural_weight_decay),
                                seed=int(mrp_seed),
                            )
                            with st.spinner("Fitting neural RR-aware MRP model..."):
                                info = model.fit(
                                    X,
                                    reported,
                                    lr=float(mrp_lr),
                                    steps=int(mrp_steps),
                                    batch_size=int(mrp_batch),
                                    keep_history=False,
                                    verbose_every=0,
                                )
                            st.write(
                                f"Fitted neural RR-aware MRP: hidden_layers={hidden_layers}, "
                                f"steps={info.steps}, final_loss={_fmt(info.final_loss, 6)}"
                            )
                            P_true = model.predict_true_proba(X)

                        elif method == "Misreport-aware RR-MRP":
                            shy_category = int(misreport_shy_category)
                            M = shy_misreport_matrix(int(k), shy_category=shy_category, honesty=float(misreport_honesty))
                            model = MisreportRRMultinomialModel(k=int(k), l2=float(mrp_l2), seed=int(mrp_seed), misreport=M)
                            with st.spinner("Fitting misreport-aware RR-MRP model..."):
                                model.fit(
                                    X,
                                    reported,
                                    eps=float(epsilon),
                                    lr=float(mrp_lr),
                                    steps=int(mrp_steps),
                                    batch_size=int(mrp_batch),
                                    verbose_every=0,
                                )
                            st.write(
                                f"Fitted misreport-aware RR-MRP: shy_category={display_labels[shy_category] if 0 <= shy_category < len(display_labels) else cmap.labels[shy_category]}, "
                                f"honesty={float(misreport_honesty):.2f}, steps={int(mrp_steps)}"
                            )
                            P_true = model.predict_theta(X)

                    except Exception as e:
                        st.error(f"{learned_method_label} failed to fit: {e}")
                        P_true = None

                    if P_true is not None:
                        # Sample-averaged latent true probabilities; this works without a population file.
                        p_mrp_sample = _normalised_mean_probability(P_true)

                        st.caption(f"{learned_method_label} sample-averaged estimate (not post-stratified):")
                        st.dataframe(
                            [{"category_id": i, "label": lab, learned_sample_col: float(p_mrp_sample[i])} for i, lab in enumerate(display_labels)],
                            use_container_width=True,
                        )

                        # Learned MRP post-strat requires population file and matching keys.
                        if pop_rows and post_cols and count_col is not None:
                            if sorted(post_cols) != sorted(mrp_feature_cols):
                                st.warning(
                                    "For learned MRP post-stratification, set post-strat key columns to match the feature columns.\n\n"
                                    f"MRP features: {mrp_feature_cols}\nPost-strat keys: {post_cols}"
                                )
                            else:
                                pop_cells: List[Dict[str, str]] = []
                                pop_counts: List[float] = []
                                for r in pop_rows:
                                    try:
                                        cval = float(r.get(str(count_col), "nan"))
                                    except Exception:
                                        cval = float("nan")
                                    if not np.isfinite(cval) or cval <= 0.0:
                                        continue
                                    pop_cells.append({c: str(r.get(c, "")).strip() for c in mrp_feature_cols})
                                    pop_counts.append(float(cval))

                                if pop_cells:
                                    X_pop = design.transform(pop_cells)
                                    w = np.asarray(pop_counts, dtype=float)
                                    if hasattr(model, "poststratify"):
                                        p_mrp_post = model.poststratify(X_pop, w)
                                    elif hasattr(model, "predict_theta"):
                                        p_mrp_post = _poststratify_probabilities(model.predict_theta(X_pop), w)
                                    else:
                                        raise RuntimeError("learned model does not support poststratification")
                                    st.subheader(f"{learned_method_label} post-stratified estimate")
                                    st.dataframe(
                                        [{"category_id": i, "label": lab, learned_post_col: float(p_mrp_post[i])} for i, lab in enumerate(display_labels)],
                                        use_container_width=True,
                                    )
                                else:
                                    st.error("Population CSV yielded no valid rows for the chosen count/key columns.")

                        # Group-level learned-model predictions for the fairness dashboard.
                        if group_cols:
                            group_rows_mrp = []
                            group_to_idx: Dict[Tuple[str, ...], List[int]] = {}
                            for pos, original_i in enumerate(valid_indices):
                                key = group_keys(poll_rows[int(original_i)], group_cols)
                                group_to_idx.setdefault(key, []).append(pos)

                            for g, idxs in group_to_idx.items():
                                idx_arr = np.asarray(idxs, dtype=int)
                                p_g_mrp = _normalised_mean_probability(P_true[idx_arr])
                                mass = float(idx_arr.size) / float(n)
                                major = mass >= float(major_mass)

                                # In real polling mode, do not require true labels. Use divergence from model overall as a proxy.
                                if p_true is not None and truth is not None:
                                    tru_g = truth[idx_arr]
                                    p_true_g = np.bincount(tru_g, minlength=k).astype(float) / max(1.0, float(tru_g.size))
                                    learned_l1 = float(np.sum(np.abs(p_g_mrp - p_true_g)))
                                else:
                                    learned_l1 = float(np.sum(np.abs(p_g_mrp - p_mrp_sample)))

                                key_str = _format_group_key(g, group_cols, category_display_map, display_answer_like_cols)
                                group_rows_mrp.append(
                                    {
                                        "group": key_str,
                                        "n": int(idx_arr.size),
                                        "mass": mass,
                                        "major": bool(major),
                                        learned_l1_key: learned_l1,
                                    }
                                )

                            group_rows_mrp.sort(key=lambda r: float(r.get("mass", 0.0)), reverse=True)

            # ===========================
            # Fairness / worst-group dashboard + report plots + bundle export
            # ===========================

            st.subheader("Fairness & worst-group dashboard")

            if not group_cols or not group_rows:
                st.info("Select group columns in the sidebar to enable fairness / worst-group metrics.")
            else:
                has_truth = p_true is not None and truth is not None
                metric_label = "L1 error vs truth" if has_truth else "L1 divergence vs overall (proxy)"
                st.caption(
                    "If you upload synthetic data with a true_choice column, these are true errors. "
                    "Otherwise we show divergence vs the overall estimate as a robustness proxy."
                )

                # Add baseline proxy metric when truth is absent: baseline_l1 := ||p_g - p_overall||_1
                if not has_truth:
                    p_ref = p_baseline
                    for r in group_rows:
                        key_str = r["group"]
                        # Recover the group key by matching string is hard; instead compute proxy during build:
                        # Here we approximate using baseline_l1 already set nan; recompute by matching on group string
                        # by rebuilding from group_estimates dict (safe and small).
                    # Rebuild a map from display group string -> l1 proxy
                    disp_to_proxy: Dict[str, float] = {}
                    for g, p_g in group_estimates.items():
                        key_str = _format_group_key(g, group_cols, category_display_map, display_answer_like_cols)
                        disp_to_proxy[key_str] = float(np.sum(np.abs(p_g - p_ref)))
                    for r in group_rows:
                        r["baseline_l1"] = disp_to_proxy.get(str(r["group"]), float("nan"))

                # Controls
                colA, colB, colC = st.columns([1, 1, 1])
                with colA:
                    major_only = st.checkbox("Major groups only", value=True, help="Only include groups with mass >= major_mass.")
                with colB:
                    show_top = st.number_input("Show top N groups (by mass)", min_value=5, max_value=100, value=20, step=5)
                with colC:
                    compare_options = ["RR debiasing"]
                    if group_rows_mrp is not None:
                        compare_options.append(learned_method_label)
                    compare_method = st.selectbox("Compare method", options=compare_options)

                # Pick which group rows to use
                if compare_method == learned_method_label and group_rows_mrp is not None:
                    g_rows = group_rows_mrp
                    key = learned_l1_key
                    title_prefix = f"{learned_method_label} ({metric_label})"
                else:
                    g_rows = group_rows
                    key = "baseline_l1"
                    title_prefix = f"RR debiasing ({metric_label})"

                # Apply the major-group filter for display. It may remove every
                # group, so the empty case is handled explicitly below.
                g_rows_show = []
                for r in g_rows:
                    if major_only and float(r.get("mass", 0.0)) < float(major_mass):
                        continue
                    g_rows_show.append(r)
                g_rows_show = sorted(g_rows_show, key=lambda r: float(r.get("mass", 0.0)), reverse=True)[: int(show_top)]

                if not g_rows_show:
                    # Do not attempt to render unavailable subgroup metrics as
                    # if they were valid estimates. The rest of the dashboard
                    # remains usable when this filter is too strict.
                    st.warning(
                        "No groups meet the current major-group mass threshold. "
                        "Lower the threshold or disable 'Major groups only' to view group metrics."
                    )
                else:
                    st.dataframe(g_rows_show, use_container_width=True)

                # Summary metrics
                summ = _group_metric_summary(g_rows, metric_key=key, major_only=major_only, major_mass=float(major_mass))
                if all(not np.isfinite(float(summ.get(m, float("nan")))) for m in ("worst", "p90", "weighted")):
                    st.info("Group summary metrics are unavailable for the current filter settings.")
                else:
                    st.write(
                        f"**Worst-group {metric_label}:** {_fmt(summ.get('worst', float('nan')), 6)}   |   "
                        f"**P90 {metric_label}:** {_fmt(summ.get('p90', float('nan')), 6)}   |   "
                        f"**Mass-weighted mean:** {_fmt(summ.get('weighted', float('nan')), 6)}   |   "
                        f"**Error ratio:** {_fmt(summ.get('error_ratio', float('nan')), 3)}"
                    )

                # Overall metrics (if truth)
                if has_truth:
                    overall_base = _overall_metrics(p_baseline, p_true)
                    st.write(f"RR debiasing overall: L1={_fmt(overall_base['overall_l1'], 6)}, MAE={_fmt(overall_base['overall_mae'], 6)}, Correct winner: {bool(overall_base['correct_winner'])}")
                    if p_post_direct is not None:
                        overall_post = _overall_metrics(p_post_direct, p_true)
                        st.write(f"Direct post-strat overall: L1={_fmt(overall_post['overall_l1'], 6)}, MAE={_fmt(overall_post['overall_mae'], 6)}, Correct winner: {bool(overall_post['correct_winner'])}")
                    if p_mrp_post is not None:
                        overall_mrp = _overall_metrics(p_mrp_post, p_true)
                        st.write(f"{learned_method_label} post-strat overall: L1={_fmt(overall_mrp['overall_l1'], 6)}, MAE={_fmt(overall_mrp['overall_mae'], 6)}, Correct winner: {bool(overall_mrp['correct_winner'])}")

            # ---------- Plots (for report) ----------
            st.subheader("Plots (for report)")

            plot_bytes: Dict[str, bytes] = {}

            series = [("baseline", p_baseline)]
            if p_post_direct is not None:
                series.append(("direct_poststrat", p_post_direct))
            if p_mrp_post is not None:
                series.append((f"{learned_prefix}_poststrat", p_mrp_post))
            if p_true is not None:
                series.append(("truth", p_true))

            overall_png = _plot_overall_distributions(
                labels=display_labels,
                series=series,
                title="Overall vote share estimate (comparison)",
            )
            if overall_png is not None:
                try:
                    st.image(overall_png, caption="Overall estimate comparison", use_container_width=True)
                except Exception as e:
                    st.warning(
                        f"Could not render overall plot image (too large). Re-generating at lower resolution. ({e})"
                    )
                    # Regenerate a smaller PNG if fig_overall exists
                    if 'fig_overall' in locals() and fig_overall is not None:
                        _buf = io.BytesIO()
                        fig_overall.savefig(_buf, format="png", dpi=120, bbox_inches="tight")
                        st.image(
                            _buf.getvalue(),
                            caption="Overall estimate comparison (downsampled)",
                            use_container_width=True,
                        )
                    else:
                        st.info("Use the download buttons to view the plot file locally.")
                plot_bytes["overall_comparison.png"] = overall_png
                st.download_button("Download overall plot (PNG)", data=overall_png, file_name="overall_comparison.png", mime="image/png")

            # Group plot (baseline or mrp)
            if group_cols and group_rows:
                has_truth = p_true is not None and truth is not None
                metric_key = "baseline_l1"
                if not has_truth:
                    # ensure proxy exists
                    p_ref = p_baseline
                    disp_to_proxy: Dict[str, float] = {}
                    for g, p_g in group_estimates.items():
                        key_str = _format_group_key(g, group_cols, category_display_map, display_answer_like_cols)
                        disp_to_proxy[key_str] = float(np.sum(np.abs(p_g - p_ref)))
                    for r in group_rows:
                        r["baseline_l1"] = disp_to_proxy.get(str(r["group"]), float("nan"))

                title = "Top groups by mass: L1 error (baseline vs truth)" if has_truth else "Top groups by mass: L1 divergence vs overall (baseline proxy)"
                grp_png = _plot_group_bars(group_rows, title=title, metric_key=metric_key, top_n=20)
                if grp_png is not None:
                    st.image(grp_png, caption="Group metric (baseline)", use_container_width=True)
                    plot_bytes["group_metric_baseline.png"] = grp_png
                    st.download_button("Download group plot (PNG)", data=grp_png, file_name="group_metric_baseline.png", mime="image/png")

            if plot_bytes:
                zbuf = io.BytesIO()
                with zipfile.ZipFile(zbuf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for name, data in plot_bytes.items():
                        zf.writestr(name, data)
                st.download_button("Download all plots (ZIP)", data=zbuf.getvalue(), file_name="fairvote_plots.zip", mime="application/zip")

            # ---------- Results bundle (ZIP) ----------
            st.subheader("Export results bundle (ZIP)")

            # Build overall estimates CSV
            overall_csv = io.StringIO()
            fieldnames = ["category_id", "label", "rr_debias_p"]
            if p_lo is not None and p_hi is not None:
                fieldnames += ["ci_low", "ci_high"]
            if p_post_direct is not None:
                fieldnames += ["direct_poststrat_p"]
            if p_mrp_post is not None:
                fieldnames += [learned_post_col]
            if p_mrp_sample is not None:
                fieldnames += [learned_sample_col]
            if p_true is not None:
                fieldnames += ["true_p"]

            w = csv.DictWriter(overall_csv, fieldnames=fieldnames)
            w.writeheader()
            for i, lab in enumerate(display_labels):
                row = {"category_id": i, "label": lab, "rr_debias_p": float(p_baseline[i])}
                if p_lo is not None and p_hi is not None:
                    row["ci_low"] = float(p_lo[i])
                    row["ci_high"] = float(p_hi[i])
                if p_post_direct is not None:
                    row["direct_poststrat_p"] = float(p_post_direct[i])
                if p_mrp_post is not None:
                    row[learned_post_col] = float(p_mrp_post[i])
                if p_mrp_sample is not None:
                    row[learned_sample_col] = float(p_mrp_sample[i])
                if p_true is not None:
                    row["true_p"] = float(p_true[i])
                w.writerow(row)
            overall_csv_bytes = overall_csv.getvalue().encode("utf-8")

            # Group audit CSV
            group_csv_bytes = b""
            if group_rows:
                group_csv = io.StringIO()
                # include learned-model group metric if available
                fn = ["group", "n", "mass", "major", "baseline_l1"]
                mrp_map = {}
                if group_rows_mrp is not None:
                    mrp_map = {str(r["group"]): float(r.get(learned_l1_key, float("nan"))) for r in group_rows_mrp}
                    fn.append(learned_l1_key)
                wg = csv.DictWriter(group_csv, fieldnames=fn)
                wg.writeheader()
                for r in group_rows:
                    out = {
                        "group": r["group"],
                        "n": int(r.get("n", 0)),
                        "mass": float(r.get("mass", 0.0)),
                        "major": bool(r.get("major", False)),
                        "baseline_l1": float(r.get("baseline_l1", float("nan"))),
                    }
                    if mrp_map:
                        out[learned_l1_key] = mrp_map.get(str(r["group"]), float("nan"))
                    wg.writerow(out)
                group_csv_bytes = group_csv.getvalue().encode("utf-8")

            # Markdown summary
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            has_truth = p_true is not None and truth is not None
            metric_label = "L1 error vs truth" if has_truth else "L1 divergence vs overall (proxy)"
            md_lines = [
                "# FairVote-AI Results Summary",
                "",
                f"- Generated: {now}",
                f"- n_rows_used: {n}",
                f"- epsilon: {float(epsilon)}",
                f"- k: {k}",
                f"- method: {method}",
                f"- group_cols: {group_cols}",
                "",
                "## Overall estimates",
                "- See: overall_estimates.csv",
                "",
                "## Group / fairness metrics",
                f"- Metric shown: {metric_label}",
            ]
            if group_rows:
                base_s = _group_metric_summary(group_rows, metric_key="baseline_l1", major_only=True, major_mass=float(major_mass))
                md_lines.append(f"- RR debiasing worst-major: {base_s['worst']:.6f}, p90-major: {base_s['p90']:.6f}, weighted-major: {base_s['weighted']:.6f}")
                if group_rows_mrp is not None:
                    mrp_s = _group_metric_summary(group_rows_mrp, metric_key=learned_l1_key, major_only=True, major_mass=float(major_mass))
                    md_lines.append(f"- {learned_method_label} worst-major: {mrp_s['worst']:.6f}, p90-major: {mrp_s['p90']:.6f}, weighted-major: {mrp_s['weighted']:.6f}")
            if has_truth:
                om = _overall_metrics(p_baseline, p_true)
                md_lines.append("")
                md_lines.append("## Truth-based overall metrics")
                md_lines.append(f"- RR debiasing: overall_l1={om['overall_l1']:.6f}, overall_mae={om['overall_mae']:.6f}")
                if p_post_direct is not None:
                    om2 = _overall_metrics(p_post_direct, p_true)
                    md_lines.append(f"- Direct post-strat: overall_l1={om2['overall_l1']:.6f}, overall_mae={om2['overall_mae']:.6f}")
                if p_mrp_post is not None:
                    om3 = _overall_metrics(p_mrp_post, p_true)
                    md_lines.append(f"- {learned_method_label} post-strat: overall_l1={om3['overall_l1']:.6f}, overall_mae={om3['overall_mae']:.6f}")
            md_lines.append("")
            md_lines.append("## Plots")
            if plot_bytes:
                for name in sorted(plot_bytes.keys()):
                    md_lines.append(f"- {name}")
            else:
                md_lines.append("- (No plots generated; install matplotlib.)")
            md_lines.append("")

            summary_md_bytes = ("\n".join(md_lines)).encode("utf-8")

            # Metadata JSON
            meta = {
                "generated_at": now,
                "n_rows_used": n,
                "epsilon": float(epsilon),
                "k": int(k),
                "method": method,
                "response_col": response_col,
                "truth_col": truth_col,
                "group_cols": group_cols,
                "major_mass": float(major_mass),
                "has_truth": bool(has_truth),
                "has_population": bool(bool(pop_rows)),
            }
            if _is_learned_mrp_method(method) and "mrp_feature_cols" in locals():
                meta["learned_method"] = learned_method_label
                meta["mrp_feature_cols"] = mrp_feature_cols
                meta["mrp_lr"] = float(mrp_lr)
                meta["mrp_steps"] = int(mrp_steps)
                meta["mrp_batch"] = int(mrp_batch)
                meta["mrp_seed"] = int(mrp_seed)
                if "mrp_l2" in locals():
                    meta["mrp_l2"] = float(mrp_l2)
                if method == "Neural RR-aware MRP":
                    meta["neural_hidden_layers"] = list(_parse_hidden_layers(neural_hidden_layers_text))
                    meta["neural_dropout"] = float(neural_dropout)
                    meta["neural_weight_decay"] = float(neural_weight_decay)
                if method == "Misreport-aware RR-MRP":
                    shy_idx = int(misreport_shy_category)
                    meta["misreport_shy_label"] = display_labels[shy_idx] if 0 <= shy_idx < len(display_labels) else str(shy_idx)
                    meta["misreport_honesty"] = float(misreport_honesty)
            if pop_rows and post_cols and count_col is not None:
                meta["post_cols"] = post_cols
                meta["count_col"] = str(count_col)

            meta_bytes = json.dumps(meta, indent=2).encode("utf-8")

            bundle = io.BytesIO()
            with zipfile.ZipFile(bundle, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("overall_estimates.csv", overall_csv_bytes)
                if group_csv_bytes:
                    zf.writestr("group_audit.csv", group_csv_bytes)
                zf.writestr("results_summary.md", summary_md_bytes)
                zf.writestr("metadata.json", meta_bytes)
                # plots
                for name, data in plot_bytes.items():
                    zf.writestr(f"plots/{name}", data)

            st.download_button(
                "Download Results Bundle (ZIP)",
                data=bundle.getvalue(),
                file_name="fairvote_results_bundle.zip",
                mime="application/zip",
            )



# ===========================
# Tab 2: Scenario Simulator
# ===========================

with tabs[1]:
    st.subheader("Scenario simulator (one-click demo)")
    st.write(
        "Generate a synthetic poll (with optional real-world biases) and compare methods end-to-end. "
        "This is ideal for a live demo + screenshots for your report."
    )

    # -----------------------
    # Synthetic generator
    # -----------------------
    def _party_labels_for_k(k: int) -> List[str]:
        if k == 5:
            return ["Labour", "Conservative", "Reform", "LibDem", "Green"]
        return [f"Party_{i}" for i in range(k)]

    def _region_labels(n: int) -> List[str]:
        base = ["London", "North", "Midlands", "South", "East", "Wales", "Scotland", "NI"]
        return base[:n] if n <= len(base) else base + [f"Region_{i}" for i in range(len(base), n)]

    def _age_labels(n: int) -> List[str]:
        base = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
        return base[:n] if n <= len(base) else base + [f"Age_{i}" for i in range(len(base), n)]

    def _softmax(logits: np.ndarray) -> np.ndarray:
        x = logits - np.max(logits)
        e = np.exp(x)
        return e / np.sum(e)

    def _prefs_for_cell(party_labels: List[str], region: str, age: str, rng: np.random.Generator) -> np.ndarray:
        """
        Structured-but-simple preference model:
        - Starts from a realistic base share (for k=5) then adds region/age effects.
        - Keeps it interpretable for write-up.
        """
        k = len(party_labels)
        if k == 5 and party_labels == ["Labour", "Conservative", "Reform", "LibDem", "Green"]:
            base = np.array([0.34, 0.30, 0.12, 0.08, 0.16], dtype=float)
            logits = np.log(base + 1e-12)

            # Age effects (very rough, but plausible)
            if age in ("18-24", "25-34"):
                logits += np.array([0.10, -0.10, 0.05, 0.10, -0.05])
            elif age in ("55-64", "65+"):
                logits += np.array([-0.10, 0.15, 0.05, -0.10, 0.00])

            # Region effects (also rough)
            if region == "London":
                logits += np.array([0.10, -0.05, 0.10, 0.00, -0.05])
            if region == "North":
                logits += np.array([0.05, -0.05, 0.00, 0.00, 0.00])
            if region == "South":
                logits += np.array([-0.05, 0.10, 0.05, 0.00, -0.10])
            if region == "Scotland":
                # We keep the demo to the canonical five-party option list.
                logits += np.array([0.00, 0.00, 0.00, 0.00, 0.15])

            # Small random cell noise so each cell isn't identical
            logits += rng.normal(0.0, 0.05, size=5)
            return _softmax(logits)

        # Generic fallback: structured Dirichlet around uniform
        alpha = np.ones(k, dtype=float) * 3.0
        p = rng.dirichlet(alpha)
        return p

    def _generate_population(regions: List[str], ages: List[str], total_pop: int, rng: np.random.Generator) -> List[Dict[str, str]]:
        # Mildly non-uniform weights (fixed for reproducibility across reruns)
        reg_w = np.linspace(1.3, 0.7, num=len(regions))
        reg_w = reg_w / reg_w.sum()
        age_w = np.linspace(1.2, 0.8, num=len(ages))
        age_w = age_w / age_w.sum()

        rows = []
        for r_i, r in enumerate(regions):
            for a_i, a in enumerate(ages):
                w = float(reg_w[r_i] * age_w[a_i])
                c = max(1, int(round(total_pop * w / (reg_w.reshape(-1, 1) * age_w.reshape(1, -1)).sum())))
                rows.append({"region": r, "age_band": a, "count": str(c)})
        return rows

    def _sample_from_population(pop_rows: List[Dict[str, str]], n: int, rng: np.random.Generator) -> List[Tuple[str, str]]:
        keys = []
        weights = []
        for r in pop_rows:
            keys.append((str(r.get("region", "")).strip(), str(r.get("age_band", "")).strip()))
            try:
                c = float(r.get("count", "0"))
            except Exception:
                c = 0.0
            weights.append(max(0.0, c))
        w = np.asarray(weights, dtype=float)
        w = w / w.sum() if w.sum() > 0 else np.full(len(keys), 1.0 / len(keys))
        idx = rng.choice(len(keys), size=int(n), replace=True, p=w)
        return [keys[i] for i in idx]

    def _apply_nonresponse(region: str, age: str, base: float, rng: np.random.Generator) -> bool:
        # Return True if respondent stays (responds)
        nr = base
        if age in ("18-24", "25-34"):
            nr += 0.12
        if age in ("65+", "55-64"):
            nr -= 0.05
        if region in ("London",):
            nr += 0.05
        if region in ("North", "Midlands"):
            nr += 0.03
        nr = float(np.clip(nr, 0.02, 0.60))
        return rng.random() >= nr

    def _apply_misreport(true_idx: int, k: int, honesty: float, rng: np.random.Generator) -> int:
        honesty = float(np.clip(honesty, 0.0, 1.0))
        if rng.random() < honesty:
            return int(true_idx)
        # pick uniformly among others
        other = [j for j in range(k) if j != true_idx]
        return int(rng.choice(other))

    def _apply_shy_effect(true_idx: int, shy_idx: int, k: int, shy_base: float, epsilon: float, rng: np.random.Generator) -> int:
        """
        'Privacy helps' model:
        - When epsilon is high (less privacy), shy voters misreport more (social desirability).
        - When epsilon is low (more privacy), misreport drops.
        """
        if true_idx != shy_idx:
            return int(true_idx)
        # misreport probability increases with epsilon
        p = float(shy_base) * (float(epsilon) / (float(epsilon) + 1.0))
        p = float(np.clip(p, 0.0, 0.95))
        if rng.random() >= p:
            return int(true_idx)
        # misreport to a uniformly random alternative category
        other = [j for j in range(k) if j != true_idx]
        return int(rng.choice(other))

    def _to_csv_bytes(rows: List[Dict[str, Any]], fieldnames: List[str]) -> bytes:
        s = io.StringIO()
        w = csv.DictWriter(s, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
        return s.getvalue().encode("utf-8")

    # -----------------------
    # Controls
    # -----------------------
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        scenario = st.selectbox("Scenario", ["no_bias", "nonresponse", "shy_privacy_helps", "misreport"], index=1)
    with c2:
        n_resp = st.number_input("Respondents (n)", min_value=500, max_value=50000, value=5000, step=500)
    with c3:
        eps = st.selectbox("epsilon", [0.2, 0.5, 1.0, 2.0], index=2)
    with c4:
        seed = st.number_input("Seed", min_value=0, max_value=10_000_000, value=123, step=1)

    c5, c6, c7, c8 = st.columns([1, 1, 1, 1])
    with c5:
        k = st.selectbox("k parties", [3, 4, 5], index=2)
    with c6:
        n_regions = st.selectbox("regions", [4, 6, 8], index=2)
    with c7:
        age_options = [3, 4, 5, 6]
    default_age = 5
    n_ages = st.selectbox("age bands", age_options, index=age_options.index(default_age) if default_age in age_options else 0)
    with c8:
        total_pop = st.number_input("Population total (for post-strat)", min_value=50_000, max_value=10_000_000, value=1_000_000, step=50_000)

    # Scenario params
    st.markdown("#### Bias parameters")
    b1, b2, b3 = st.columns([1, 1, 1])
    with b1:
        nonresponse_base = st.slider("Base nonresponse rate", 0.0, 0.5, 0.15, 0.01, help="Only used in nonresponse scenario.")
    with b2:
        shy_base = st.slider("Shy misreport base", 0.0, 0.8, 0.40, 0.02, help="Only used in shy_privacy_helps scenario.")
    with b3:
        honesty = st.slider("Misreport honesty", 0.0, 1.0, 0.80, 0.01, help="Only used in misreport scenario.")

    run = st.button("Generate & run comparison", type="primary")

    if run:
        rng = np.random.default_rng(int(seed))
        parties = _party_labels_for_k(int(k))
        regions = _region_labels(int(n_regions))
        ages = _age_labels(int(n_ages))

        pop_rows = _generate_population(regions, ages, int(total_pop), rng)

        # sample demographics from population
        demos = _sample_from_population(pop_rows, int(n_resp), rng)

        # choose shy party index (if exists)
        shy_idx = 1 if ("Conservative" in parties) else 0  # default: party 1 if possible

        poll_rows = []
        for (reg, age) in demos:
            # response filtering (nonresponse)
            if scenario == "nonresponse":
                if not _apply_nonresponse(reg, age, float(nonresponse_base), rng):
                    continue

            p = _prefs_for_cell(parties, reg, age, rng)
            true_idx = int(rng.choice(len(parties), p=p))

            declared_idx = true_idx
            if scenario == "misreport":
                declared_idx = _apply_misreport(true_idx, len(parties), float(honesty), rng)
            elif scenario == "shy_privacy_helps":
                declared_idx = _apply_shy_effect(true_idx, shy_idx, len(parties), float(shy_base), float(eps), rng)

            from fairvote.privacy.mechanisms.kary_rr import privatize_one
            reported_idx = privatize_one(declared_idx, float(eps), len(parties), rng)

            poll_rows.append(
                {
                    "region": reg,
                    "age_band": age,
                    "true_choice": parties[true_idx],
                    "declared_choice": parties[declared_idx],
                    "reported_choice": parties[reported_idx],
                    "epsilon": str(float(eps)),
                }
            )

        if len(poll_rows) < 100:
            st.error("Too few respondents after bias (increase n or reduce nonresponse).")
        else:
            st.success(f"Generated poll: {len(poll_rows)} respondents (after bias), k={len(parties)}, eps={eps}")

            # Downloads
            poll_csv = _to_csv_bytes(poll_rows, ["region", "age_band", "true_choice", "declared_choice", "reported_choice", "epsilon"])
            pop_csv = _to_csv_bytes(pop_rows, ["region", "age_band", "count"])
            st.download_button("Download synthetic poll CSV", data=poll_csv, file_name="synthetic_poll.csv", mime="text/csv")
            st.download_button("Download synthetic population CSV", data=pop_csv, file_name="synthetic_population.csv", mime="text/csv")

            # --------------------------------
            # Run the same analysis pipeline as Upload tab
            # (fixed columns: reported_choice, true_choice, region, age_band)
            # --------------------------------
            cmap = CategoryMap(labels=parties, to_int={lab: i for i, lab in enumerate(parties)})
            k_eff = len(parties)

            reported = encode_categories([r["reported_choice"] for r in poll_rows], cmap)
            truth = encode_categories([r["true_choice"] for r in poll_rows], cmap)
            reported, truth, valid_mask = filter_valid(reported, truth)
            n_eff = int(reported.size)

            p_baseline = estimate_distribution(reported, epsilon=float(eps), k=k_eff)
            p_true = np.bincount(truth, minlength=k_eff).astype(float) / max(1.0, float(n_eff))
            
            from fairvote.privacy.mechanisms.laplace_mechanism import estimate_distribution_central_dp
            p_central_dp = estimate_distribution_central_dp(truth, epsilon=float(eps), k=k_eff, rng=rng)

            # Group audit by region|age_band
            group_cols = ["region", "age_band"]
            major_mass = 0.02
            group_rows = []
            group_estimates = {}
            for r in poll_rows:
                pass  # No per-row work is needed here; grouped indices are built below.

            group_to_idx = {}
            for i, row in enumerate(poll_rows):
                key = (row["region"], row["age_band"])
                group_to_idx.setdefault(key, []).append(i)

            for g, idxs in group_to_idx.items():
                idx_arr = np.asarray(idxs, dtype=int)
                rep_g = reported[idx_arr]
                tru_g = truth[idx_arr]
                p_g = estimate_distribution(rep_g, epsilon=float(eps), k=k_eff)
                p_tg = np.bincount(tru_g, minlength=k_eff).astype(float) / max(1.0, float(tru_g.size))
                group_estimates[g] = p_g
                mass = float(rep_g.size) / float(n_eff)
                group_rows.append(
                    {
                        "group": f"{g[0]} | {g[1]}",
                        "n": int(rep_g.size),
                        "mass": mass,
                        "major": bool(mass >= major_mass),
                        "baseline_l1": float(np.sum(np.abs(p_g - p_tg))),
                    }
                )
            group_rows.sort(key=lambda r: float(r["mass"]), reverse=True)

            # Direct post-strat (keys match)
            pop_weights = read_population_weights(pop_rows, ["region", "age_band"], "count")
            p_post_direct = poststratify_from_groups(group_estimates, pop_weights, fallback=p_baseline) if pop_weights else None

            # RR-aware MRP (optional, if available)
            p_mrp_post = None
            group_rows_mrp = None
            p_mrp_sample = None
            if _HAS_RR_MRP and MRPRRMultinomialModel is not None and DesignMatrix is not None:
                # Fit MRP with region + age_band
                design = DesignMatrix(["region", "age_band"]).fit(poll_rows)
                X = design.transform(poll_rows)
                model = MRPRRMultinomialModel(k=k_eff, epsilon=float(eps), l2=1.0, seed=int(seed))
                with st.spinner("Fitting RR-aware MRP..."):
                    model.fit(X, reported, lr=0.05, steps=2000, batch_size=512, verbose_every=0, keep_history=False)

                P_true = model.predict_true_proba(X)
                p_mrp_sample = np.mean(P_true, axis=0)
                p_mrp_sample = np.clip(p_mrp_sample, 0.0, 1.0)
                s = float(p_mrp_sample.sum())
                if s > 0:
                    p_mrp_sample /= s

                # Post-strat with population
                pop_cells = [{"region": r["region"], "age_band": r["age_band"]} for r in pop_rows]
                pop_counts = np.asarray([float(r["count"]) for r in pop_rows], dtype=float)
                Xp = design.transform(pop_cells)
                p_mrp_post = model.poststratify(Xp, pop_counts)

                # Group-level MRP L1 vs truth
                group_rows_mrp = []
                for g, idxs in group_to_idx.items():
                    idx_arr = np.asarray(idxs, dtype=int)
                    p_g_mrp = np.mean(P_true[idx_arr], axis=0)
                    p_g_mrp = np.clip(p_g_mrp, 0.0, 1.0)
                    s = float(p_g_mrp.sum())
                    if s > 0:
                        p_g_mrp /= s
                    tru_g = truth[idx_arr]
                    p_tg = np.bincount(tru_g, minlength=k_eff).astype(float) / max(1.0, float(tru_g.size))
                    mass = float(idx_arr.size) / float(n_eff)
                    group_rows_mrp.append(
                        {
                            "group": f"{g[0]} | {g[1]}",
                            "n": int(idx_arr.size),
                            "mass": mass,
                            "major": bool(mass >= major_mass),
                            "mrp_l1": float(np.sum(np.abs(p_g_mrp - p_tg))),
                        }
                    )
                group_rows_mrp.sort(key=lambda r: float(r["mass"]), reverse=True)

            # -----------------------
            # Display: overall table
            # -----------------------
            st.subheader("Overall comparison (truth known)")
            rows = []
            for i, lab in enumerate(parties):
                row = {"label": lab, "true_p": float(p_true[i]), "baseline_p": float(p_baseline[i]), "central_dp_p": float(p_central_dp[i])}
                if p_post_direct is not None:
                    row["direct_poststrat_p"] = float(p_post_direct[i])
                if p_mrp_post is not None:
                    row["mrp_poststrat_p"] = float(p_mrp_post[i])
                rows.append(row)
            st.dataframe(rows, use_container_width=True)

            # -----------------------
            # Fairness summary
            # -----------------------
            st.subheader("Worst-group / fairness metrics (truth known)")
            base_s = _group_metric_summary(group_rows, metric_key="baseline_l1", major_only=True, major_mass=major_mass)
            st.write(
                f"Baseline major-groups: worst={_fmt(base_s['worst'],6)} | p90={_fmt(base_s['p90'],6)} | weighted={_fmt(base_s['weighted'],6)}"
            )
            if group_rows_mrp is not None:
                mrp_s = _group_metric_summary(group_rows_mrp, metric_key="mrp_l1", major_only=True, major_mass=major_mass)
                st.write(
                    f"MRP major-groups: worst={_fmt(mrp_s['worst'],6)} | p90={_fmt(mrp_s['p90'],6)} | weighted={_fmt(mrp_s['weighted'],6)}"
                )
            st.caption("Group table below is sorted by mass (largest groups first).")
            st.dataframe(group_rows, use_container_width=True)
            if group_rows_mrp is not None:
                st.dataframe(group_rows_mrp, use_container_width=True)

            # -----------------------
            # Plots + bundle
            # -----------------------
            st.subheader("Report-ready plots + bundle")
            plot_bytes = {}
            series = [("truth", p_true), ("baseline (LDP)", p_baseline), ("central DP", p_central_dp)]
            if p_post_direct is not None:
                series.append(("direct_poststrat", p_post_direct))
            if p_mrp_post is not None:
                series.append(("mrp_poststrat", p_mrp_post))

            overall_png = _plot_overall_distributions(parties, series, "Overall vote share (truth vs methods)")
            if overall_png is not None:
                st.image(overall_png, use_container_width=True)
                plot_bytes["overall_comparison.png"] = overall_png

            grp_png = _plot_group_bars(group_rows, "Top groups by mass: baseline L1 vs truth", "baseline_l1", top_n=20)
            if grp_png is not None:
                st.image(grp_png, use_container_width=True)
                plot_bytes["group_baseline_l1.png"] = grp_png

            # Bundle ZIP
            overall_csv = _to_csv_bytes(rows, list(rows[0].keys()))
            group_csv = _to_csv_bytes(group_rows, ["group", "n", "mass", "major", "baseline_l1"])
            md = "\n".join(
                [
                    "# FairVote-AI Scenario Simulator Run",
                    "",
                    f"- scenario: {scenario}",
                    f"- n: {len(poll_rows)} (after bias)",
                    f"- epsilon: {eps}",
                    f"- k: {k_eff}",
                    "",
                    "## Major-group fairness metrics",
                    f"- baseline: worst={base_s['worst']:.6f}, p90={base_s['p90']:.6f}, weighted={base_s['weighted']:.6f}",
                ]
            )
            if group_rows_mrp is not None:
                mrp_s = _group_metric_summary(group_rows_mrp, metric_key="mrp_l1", major_only=True, major_mass=major_mass)
                md += "\n" + f"- mrp: worst={mrp_s['worst']:.6f}, p90={mrp_s['p90']:.6f}, weighted={mrp_s['weighted']:.6f}"

            meta = {
                "scenario": scenario,
                "n_after_bias": len(poll_rows),
                "epsilon": float(eps),
                "seed": int(seed),
                "k": int(k_eff),
                "regions": regions,
                "ages": ages,
                "nonresponse_base": float(nonresponse_base),
                "shy_base": float(shy_base),
                "honesty": float(honesty),
                "has_mrp": bool(p_mrp_post is not None),
            }

            bundle = io.BytesIO()
            with zipfile.ZipFile(bundle, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("synthetic_poll.csv", poll_csv)
                zf.writestr("synthetic_population.csv", pop_csv)
                zf.writestr("overall_comparison.csv", overall_csv)
                zf.writestr("group_audit.csv", group_csv)
                zf.writestr("summary.md", md.encode("utf-8"))
                zf.writestr("metadata.json", json.dumps(meta, indent=2).encode("utf-8"))
                for name, data in plot_bytes.items():
                    zf.writestr(f"plots/{name}", data)

            st.download_button(
                "Download scenario bundle (ZIP)",
                data=bundle.getvalue(),
                file_name="fairvote_scenario_bundle.zip",
                mime="application/zip",
            )
            st.success("Done. Use the ZIP contents directly in your report.")


# ===========================
# Tab 3: Simulation & Runs
# ===========================

with tabs[2]:
    st.subheader("Simulation & runs")
    st.write("Run the experiment pipeline and inspect outputs.")

    def _list_runs(base: Path) -> List[Path]:
        if not base.exists():
            return []
        runs = [p for p in base.iterdir() if p.is_dir()]
        runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return runs

    def _run_module(module: str, argv: List[str], cwd: Path) -> Tuple[int, str]:
        cmd = [sys.executable, "-m", module, *argv]
        proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
        out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        return proc.returncode, out

    def _read_csv_rows(path: Path) -> List[dict]:
        with path.open("r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    with st.sidebar:
        st.header("Simulation configuration")
        sim_trials = st.number_input("Trials", min_value=1, max_value=200, value=10, step=1, key="sim_trials")
        sim_eps = st.text_input("Eps list", value="0.2,0.5,1.0,2.0", key="sim_eps")
        sim_major_mass = st.number_input("Major mass", min_value=0.0, max_value=1.0, value=0.02, step=0.01, key="sim_major_mass")

    colA, colB = st.columns([1, 1])

    with colA:
        if st.button("Run mrp_vs_baselines", type="primary"):
            with st.spinner("Running mrp_vs_baselines..."):
                rc, out = _run_module(
                    "experiments.mrp_vs_baselines",
                    ["--trials", str(int(sim_trials)), "--eps", str(sim_eps), "--major_mass", str(float(sim_major_mass))],
                    cwd=root,
                )
            st.text_area("Output", out, height=280)
            if rc == 0:
                st.success("Run completed.")
            else:
                st.error("Run failed. See output above.")

    with colB:
        st.write("Existing run folders:")
        runs = _list_runs(outputs_dir)
        choice = st.selectbox("Run folder", options=[p.name for p in runs] if runs else ["(none)"])
        if choice != "(none)":
            run_dir = outputs_dir / choice
            st.code(str(run_dir.as_posix()))
            summary_csv = run_dir / "summary.csv"
            if summary_csv.exists():
                st.dataframe(_read_csv_rows(summary_csv), use_container_width=True)
            else:
                st.info("No summary.csv found in this run folder.")


# ===========================
# Tab 4: Recommendations
# ===========================

with tabs[3]:
    st.subheader("Optimisation & Recommendations")
    st.markdown("Upload a `summary.csv` from an experiment run to find the optimal privacy-utility configuration that satisfies your requirements.")
    
    _rec_csv = st.file_uploader("Upload summary.csv from an experiment run", type=["csv"], key="rec_csv")
    if _rec_csv is not None:
        try:
            from fairvote.optimisation.recommend import read_summary_csv, Constraints, Objective, recommend_per_scenario
            import tempfile
            from pathlib import Path

            raw_csv = _rec_csv.getvalue()
            with tempfile.NamedTemporaryFile("wb", delete=False, suffix=".csv") as f:
                f.write(raw_csv)
                temp_path = Path(f.name)
            
            cands = read_summary_csv(temp_path)
            temp_path.unlink()

            if not cands:
                st.warning("No valid candidates found in CSV.")
            else:
                st.success(f"Loaded {len(cands)} candidates across {len(set(c.scenario for c in cands))} scenarios.")

                colR1, colR2 = st.columns(2)
                with colR1:
                    st.markdown("**Privacy Constraints**")
                    eps_max = st.number_input("Max Epsilon (epsilon_max)", min_value=0.0, max_value=10.0, value=10.0, step=0.1)
                    
                    st.markdown("**Utility Constraints**")
                    l1_max = st.number_input("Max Overall L1 Error", min_value=0.0, max_value=2.0, value=2.0, step=0.01)
                    
                with colR2:
                    st.markdown("**Fairness Constraints**")
                    w_reg_l1 = st.number_input("Max Worst Region L1 (Major)", min_value=0.0, max_value=2.0, value=2.0, step=0.01)
                    w_age_l1 = st.number_input("Max Worst Age L1 (Major)", min_value=0.0, max_value=2.0, value=2.0, step=0.01)

                obj_primary = st.selectbox("Optimize Objective (minimise)", ["mean_overall_l1", "mean_worst_region_l1_major", "mean_worst_age_l1_major", "mean_overall_mae", "epsilon"])

                if st.button("Generate Recommendations", type="primary"):
                    cons = Constraints(
                        epsilon_max=float(eps_max) if eps_max < 9.9 else None,
                        overall_l1_max=float(l1_max) if l1_max < 1.99 else None,
                        worst_region_l1_major_max=float(w_reg_l1) if w_reg_l1 < 1.99 else None,
                        worst_age_l1_major_max=float(w_age_l1) if w_age_l1 < 1.99 else None,
                    )
                    objective = Objective(primary=obj_primary)
                    
                    recs = recommend_per_scenario(cands, constraints=cons, objective=objective)
                    
                    for rec in recs:
                        st.markdown(f"### Scenario: `{rec.scenario}`")
                        if rec.chosen:
                            st.success(f"**Recommended Method:** `{rec.chosen.method}` at **$\\epsilon={rec.chosen.epsilon}$**")
                            
                            c_dict = {
                                "Metric": ["Overall L1 Error", "Worst Region L1 (Major)", "Worst Age L1 (Major)"],
                                "Value": [f"{rec.chosen.mean_overall_l1:.4f} ± {rec.chosen.std_overall_l1:.4f}",
                                          f"{rec.chosen.mean_worst_region_l1_major:.4f} ± {rec.chosen.std_worst_region_l1_major:.4f}",
                                          f"{rec.chosen.mean_worst_age_l1_major:.4f} ± {rec.chosen.std_worst_age_l1_major:.4f}"]
                            }
                            st.table(c_dict)
                        else:
                            st.error(f"No feasible configuration found. Reason: {rec.reason_if_none}")
                        st.markdown(f"*(Feasible candidates: {rec.feasible_count} / {rec.total_count})*")
                        st.divider()

        except Exception as e:
            st.error(f"Error parsing recommendations: {e}")

# ===========================
# Tab 5: About
# ===========================

with tabs[4]:
    st.subheader("What to demo for marks")
    st.markdown(
        """
### Real-world demo (Upload & Estimate)
- Upload a poll dataset (RR-privatised responses)
- Show baseline RR-debias + group audit
- Upload population counts and show post-stratification
- Switch to RR-aware MRP and compare estimates (MRP may help under nonresponse / demographic skew, but is not guaranteed to improve results)
- Export the Results Bundle ZIP (plots + tables + markdown) and paste into your report

### Marks-bearing experiments
- Use `experiments.mrp_vs_baselines` for controlled trials across epsilons and bias scenarios
- Use the report table generator + recommendation scripts for the engineering decision story
"""
    )

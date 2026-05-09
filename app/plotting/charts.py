"""Matplotlib plotting helpers for report-ready dashboard images."""

from __future__ import annotations

import io
from typing import Any, Optional, Sequence

import numpy as np

HAS_MPL = True
try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator

    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 150
    plt.rcParams["figure.figsize"] = (8, 4)
except Exception:
    HAS_MPL = False
    plt = None  # type: ignore[assignment]
    AutoMinorLocator = None  # type: ignore[assignment]


def fig_to_png_bytes(fig: Any) -> bytes:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    return buf.getvalue()


def apply_readable_grid(ax: Any, orientation: str = "vertical") -> None:
    """Apply subtle readable grids without changing plotted values."""

    ax.set_axisbelow(True)

    if orientation == "horizontal":
        if AutoMinorLocator is not None:
            try:
                ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            except Exception:
                pass
        ax.grid(True, which="major", axis="x", alpha=0.35, linewidth=0.8)
        ax.grid(True, which="minor", axis="x", alpha=0.18, linewidth=0.5)
        ax.grid(True, which="major", axis="y", alpha=0.12, linewidth=0.5)
    elif orientation == "vertical":
        if AutoMinorLocator is not None:
            try:
                ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            except Exception:
                pass
        ax.grid(True, which="major", axis="y", alpha=0.35, linewidth=0.8)
        ax.grid(True, which="minor", axis="y", alpha=0.18, linewidth=0.5)
        ax.grid(True, which="major", axis="x", alpha=0.12, linewidth=0.5)


def plot_overall_distributions(
    labels: Sequence[str],
    series: Sequence[tuple[str, np.ndarray]],
    title: str,
) -> Optional[bytes]:
    if not HAS_MPL or plt is None:
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
    apply_readable_grid(ax, orientation="vertical")
    ax.legend()

    png = fig_to_png_bytes(fig)
    plt.close(fig)
    return png


def plot_group_bars(
    group_rows: list[dict[str, Any]],
    title: str,
    metric_key: str,
    top_n: int = 20,
) -> Optional[bytes]:
    if not HAS_MPL or plt is None or not group_rows:
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
    apply_readable_grid(ax, orientation="horizontal")

    png = fig_to_png_bytes(fig)
    plt.close(fig)
    return png

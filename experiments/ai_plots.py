from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import plot_tree

from experiments.plots import METHOD_COLOURS
from fairvote.ai.features import FEATURE_COLUMNS
from fairvote.ai.selector import FittedSelector
from fairvote.study import METHOD_LABELS, METHODS

plt.switch_backend("Agg")

BASELINE_PLOT_NAME = "ai_selector_baselines.png"
TREE_PLOT_PREFIX = "ai_selector_tree_"
MODEL_COLOUR = "#444444"
ORACLE_COLOUR = "#999999"


def make_ai_plots(
    metrics: dict[str, object],
    selector: FittedSelector,
    plots_dir: Path,
) -> list[Path]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    paths = [plots_dir / BASELINE_PLOT_NAME]
    _plot_baselines(metrics, paths[0])
    for method in METHODS:
        path = plots_dir / f"{TREE_PLOT_PREFIX}{method}.png"
        _plot_tree(selector, method, path)
        paths.append(path)
    return paths


def _baseline_series(metrics: dict[str, object]) -> tuple[list[str], list[float], list[float], list[str]]:
    grouped = metrics["grouped_cross_validation"]["recommendation"]
    held_out = metrics["leave_one_epsilon_out"]["recommendation"]
    baselines = metrics["fixed_baselines"]

    labels = ["Recommender (grouped CV)", "Recommender (held-out epsilon)"]
    selected = [float(grouped["mean_selected_l1"]), float(held_out["mean_selected_l1"])]
    regret = [float(grouped["mean_regret"]), float(held_out["mean_regret"])]
    colours = [MODEL_COLOUR, MODEL_COLOUR]

    for method in METHODS:
        labels.append(f"Always: {METHOD_LABELS[method]}")
        selected.append(float(baselines[method]["mean_selected_l1"]))
        regret.append(float(baselines[method]["mean_regret"]))
        colours.append(METHOD_COLOURS[method])

    labels.append("Oracle best per poll")
    selected.append(float(baselines["oracle"]["mean_selected_l1"]))
    regret.append(float(baselines["oracle"]["mean_regret"]))
    colours.append(ORACLE_COLOUR)
    return labels, selected, regret, colours


def _plot_baselines(metrics: dict[str, object], path: Path) -> None:
    labels, selected, regret, colours = _baseline_series(metrics)
    positions = np.arange(len(labels), dtype=float)

    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))
    for axis, values, title in (
        (axes[0], selected, "Mean L1 error of the selected estimator"),
        (axes[1], regret, "Mean regret against the best estimator per poll"),
    ):
        axis.barh(positions, values, color=colours, height=0.62)
        axis.set_yticks(positions)
        axis.set_yticklabels(labels, fontsize="small")
        axis.invert_yaxis()
        axis.set_xlabel("L1 error")
        axis.set_title(title, fontsize="medium")
        axis.grid(alpha=0.3, axis="x")
        span = max(values) if max(values) > 0 else 1.0
        for position, value in zip(positions, values, strict=True):
            axis.text(value + span * 0.02, position, f"{value:.4f}", va="center", fontsize="small")
        axis.set_xlim(0.0, span * 1.22)

    figure.suptitle("AI-assisted estimator recommendation against fixed baselines")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_tree(selector: FittedSelector, method: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(28.0, 15.0))
    plot_tree(
        selector.models[method],
        feature_names=list(FEATURE_COLUMNS),
        filled=True,
        rounded=True,
        precision=4,
        fontsize=8,
        proportion=False,
        ax=axis,
    )
    axis.set_title(
        f"Predicted L1 error: {METHOD_LABELS[method]}",
        fontsize="large",
        pad=16,
    )
    figure.subplots_adjust(
        left=0.02,
        right=0.98,
        bottom=0.02,
        top=0.92,
    )
    figure.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.2,
    )
    plt.close(figure)

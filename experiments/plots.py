from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from fairvote.study import METHOD_LABELS, METHODS, SUPPORTED_BIAS_LEVELS

# The non-interactive Agg backend allows experiment figures to be generated without opening a graphical window.
plt.switch_backend("Agg")

# These reference settings are used when holding one experiment dimension fixed in the statistical plots.
REFERENCE_EPSILON = 1.0
REFERENCE_SAMPLE_SIZE = 1000

# Each estimator is assigned the corresponding colour from the currently configured Matplotlib axes colour cycle.
_DEFAULT_COLOURS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
METHOD_COLOURS = {method: _DEFAULT_COLOURS[index] for index, method in enumerate(METHODS)}


def make_plots(results: pd.DataFrame, plots_dir: Path) -> list[Path]:
    # The plots directory is created before the three statistical figure paths are prepared.
    plots_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        plots_dir / "l1_vs_epsilon.png",
        plots_dir / "l1_vs_sample_size.png",
        plots_dir / "l1_by_bias.png",
    ]

    # Existing versions of these three owned plot files are removed before regeneration.
    for path in paths:
        path.unlink(missing_ok=True)

    # Each plotting function receives the same estimator-level results table and writes one PNG file.
    _plot_l1_vs_epsilon(results, paths[0])
    _plot_l1_vs_sample_size(results, paths[1])
    _plot_l1_by_bias(results, paths[2])
    return paths


def _plot_l1_vs_epsilon(results: pd.DataFrame, path: Path) -> None:
    # This plot varies epsilon while holding the respondent count at the shared reference sample size.
    subset = results[results["n_respondents"] == REFERENCE_SAMPLE_SIZE]
    repetitions = _repetition_count(subset)
    bias_levels = _available_bias_levels(subset)

    # One panel is created for each bias condition present in the filtered results, with a shared vertical scale.
    figure, axes = plt.subplots(
        1,
        len(bias_levels),
        figsize=(4.6 * len(bias_levels), 4.1),
        sharey=True,
    )
    axes = np.atleast_1d(axes)

    # Within each bias panel, one curve shows the mean L1 error of each estimator at every available epsilon.
    for axis, bias in zip(axes, bias_levels, strict=True):
        bias_subset = subset[subset["bias"] == bias]
        for method in METHODS:
            curve = bias_subset[bias_subset["method"] == method].groupby("epsilon", sort=True)["l1_error"].mean()
            axis.plot(
                curve.index,
                curve.to_numpy(),
                marker="o",
                color=METHOD_COLOURS[method],
                label=METHOD_LABELS[method],
            )
        axis.set_title(f"Bias: {bias}")
        axis.set_xlabel("Privacy budget (epsilon)")
        axis.grid(alpha=0.3)

    # The shared figure labels record the plotted error measure, reference sample size and repetition count.
    axes[0].set_ylabel("Mean L1 error")
    axes[-1].legend(fontsize="small")
    figure.suptitle(f"L1 error versus epsilon at n = {REFERENCE_SAMPLE_SIZE} ({repetitions} repetitions)")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_l1_vs_sample_size(results: pd.DataFrame, path: Path) -> None:
    # This plot varies respondent count while holding epsilon at the shared reference privacy budget.
    subset = results[np.isclose(results["epsilon"], REFERENCE_EPSILON)]
    repetitions = _repetition_count(subset)
    bias_levels = _available_bias_levels(subset)

    # One panel is created for each bias condition present in the filtered results, with a shared vertical scale.
    figure, axes = plt.subplots(
        1,
        len(bias_levels),
        figsize=(4.6 * len(bias_levels), 4.1),
        sharey=True,
    )
    axes = np.atleast_1d(axes)

    # Within each bias panel, one curve shows the mean L1 error of each estimator at every available sample size.
    for axis, bias in zip(axes, bias_levels, strict=True):
        bias_subset = subset[subset["bias"] == bias]
        for method in METHODS:
            curve = bias_subset[bias_subset["method"] == method].groupby("n_respondents", sort=True)["l1_error"].mean()
            axis.plot(
                curve.index,
                curve.to_numpy(),
                marker="o",
                color=METHOD_COLOURS[method],
                label=METHOD_LABELS[method],
            )
        axis.set_title(f"Bias: {bias}")
        axis.set_xlabel("Number of respondents")
        axis.grid(alpha=0.3)

    # The shared figure labels record the plotted error measure, reference epsilon and repetition count.
    axes[0].set_ylabel("Mean L1 error")
    axes[-1].legend(fontsize="small")
    figure.suptitle(f"L1 error versus sample size at epsilon = {REFERENCE_EPSILON} ({repetitions} repetitions)")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_l1_by_bias(results: pd.DataFrame, path: Path) -> None:
    # The boxplot compares bias conditions at the common reference epsilon and sample size.
    subset = results[
        np.isclose(results["epsilon"], REFERENCE_EPSILON) & (results["n_respondents"] == REFERENCE_SAMPLE_SIZE)
    ]
    repetitions = _repetition_count(subset)
    bias_levels = _available_bias_levels(subset)

    # Positions are built explicitly so the three estimator boxplots form a visually separated group for each bias condition.
    positions: list[float] = []
    distributions: list[np.ndarray] = []
    group_centres: list[float] = []
    cursor = 1.0

    # Each box contains the repetition-level L1 errors for one estimator under one bias condition.
    for bias in bias_levels:
        start = cursor
        for method in METHODS:
            values = subset[(subset["bias"] == bias) & (subset["method"] == method)]["l1_error"].to_numpy()
            distributions.append(values)
            positions.append(cursor)
            cursor += 1.0
        group_centres.append((start + cursor - 1.0) / 2.0)
        cursor += 0.8

    # The boxplot displays means, medians, whiskers, caps and outliers for the collected repetition-level distributions.
    figure, axis = plt.subplots(figsize=(10.5, 4.8))
    boxplot = axis.boxplot(
        distributions,
        positions=positions,
        widths=0.65,
        showmeans=True,
        patch_artist=True,
        meanprops={
            "marker": "^",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markeredgewidth": 1.2,
            "markersize": 7,
        },
        medianprops={
            "color": "black",
            "linewidth": 1.6,
        },
        whiskerprops={
            "color": "black",
            "linewidth": 1.1,
        },
        capprops={
            "color": "black",
            "linewidth": 1.1,
        },
        flierprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markeredgewidth": 1.0,
            "markersize": 4,
            "alpha": 0.8,
        },
    )

    # Boxes repeat the shared estimator colour mapping within every bias group.
    for index, box in enumerate(boxplot["boxes"]):
        method = METHODS[index % len(METHODS)]
        box.set_facecolor(METHOD_COLOURS[method])
        box.set_edgecolor("black")
        box.set_linewidth(1.0)

    # Group centres provide one x-axis label per bias condition rather than one label per estimator box.
    axis.set_xticks(group_centres)
    axis.set_xticklabels(bias_levels)
    axis.set_xlabel("Sampling bias")
    axis.set_ylabel("L1 error")
    axis.set_title(
        f"L1 error by sampling bias at epsilon = {REFERENCE_EPSILON}, "
        f"n = {REFERENCE_SAMPLE_SIZE} ({repetitions} repetitions)"
    )
    axis.grid(alpha=0.3, axis="y")

    # The legend maps each box colour back to the corresponding estimator label.
    axis.legend(
        handles=[Patch(facecolor=METHOD_COLOURS[method], label=METHOD_LABELS[method]) for method in METHODS],
        fontsize="small",
    )
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _available_bias_levels(results: pd.DataFrame) -> list[str]:
    # Bias conditions are returned only when present in the data and remain in the shared supported order.
    present = set(results["bias"])
    return [bias for bias in SUPPORTED_BIAS_LEVELS if bias in present]


def _repetition_count(results: pd.DataFrame) -> int:
    # An empty reference subset cannot provide a repetition count for a plot title.
    if results.empty:
        raise ValueError("The requested reference configuration is absent from the results.")

    # Unique seeds are counted within each experiment-setting and estimator group.
    counts = results.groupby(["epsilon", "n_respondents", "bias", "method"])["seed"].nunique()

    # The minimum group count is used as the repetition count displayed in the figure title.
    return int(counts.min())

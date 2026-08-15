from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from experiments.ai_plots import make_ai_plots
from fairvote.ai.evaluation import (
    build_metrics,
    evaluate_grouped,
    evaluate_leave_one_epsilon_out,
)
from fairvote.ai.features import (
    build_selector_dataset,
    validate_selector_dataset,
)
from fairvote.ai.selector import train_selector

DATASET_NAME = "selector_dataset.csv"
PREDICTIONS_NAME = "selector_predictions.csv"
METRICS_NAME = "selector_metrics.json"


def run(quick: bool, output_dir: Path) -> dict[str, object]:
    ai_dir = output_dir / "ai"
    plots_dir = output_dir / "plots"
    ai_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_selector_dataset(quick=quick)
    validate_selector_dataset(dataset, quick=quick)

    grouped_predictions, grouped_metrics = evaluate_grouped(dataset)
    held_out_predictions, held_out_metrics = evaluate_leave_one_epsilon_out(dataset)
    predictions = pd.concat([grouped_predictions, held_out_predictions], ignore_index=True)

    metrics = build_metrics(dataset, grouped_metrics, held_out_metrics)
    selector = train_selector(dataset)

    dataset_path = ai_dir / DATASET_NAME
    predictions_path = ai_dir / PREDICTIONS_NAME
    metrics_path = ai_dir / METRICS_NAME
    for stale in (dataset_path, predictions_path, metrics_path):
        stale.unlink(missing_ok=True)

    dataset.to_csv(dataset_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    plot_paths = make_ai_plots(metrics, selector, plots_dir)

    return {
        "dataset": dataset,
        "metrics": metrics,
        "paths": [dataset_path, predictions_path, metrics_path, *plot_paths],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate the FairVote-AI estimator recommender.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use the small reference grid with 3 repetitions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for the AI dataset, metrics and plots (default: results).",
    )
    args = parser.parse_args(argv)

    outcome = run(quick=args.quick, output_dir=args.output_dir)
    dataset = outcome["dataset"]
    metrics = outcome["metrics"]
    grouped = metrics["grouped_cross_validation"]["recommendation"]
    held_out = metrics["leave_one_epsilon_out"]["recommendation"]
    baselines = metrics["fixed_baselines"]

    mode = "quick" if args.quick else "full"
    print(f"FairVote-AI selector finished ({mode} mode).")
    print(f"  dataset rows            : {len(dataset)}")
    print(f"  grouped folds           : {metrics['grouped_cross_validation']['n_splits']}")
    print(f"  tie threshold           : {metrics['final_model_tie_threshold']:.4f}")
    print(f"  grouped argmin accuracy : {grouped['argmin_accuracy']:.3f}")
    print(f"  grouped mean regret     : {grouped['mean_regret']:.4f}")
    print(f"  held-out mean regret    : {held_out['mean_regret']:.4f}")
    print("  mean selected L1 error:")
    print(f"    recommender             {grouped['mean_selected_l1']:.4f}")
    for method, values in baselines.items():
        prefix = "oracle best per poll" if method == "oracle" else f"always {method}"
        print(f"    {prefix:<23} {values['mean_selected_l1']:.4f}")
    for path in outcome["paths"]:
        print(f"  wrote                   : {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

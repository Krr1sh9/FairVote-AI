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

# These filenames identify the three recommender data files written under the output directory's ai subdirectory.
DATASET_NAME = "selector_dataset.csv"
PREDICTIONS_NAME = "selector_predictions.csv"
METRICS_NAME = "selector_metrics.json"


def run(quick: bool, output_dir: Path) -> dict[str, object]:
    # Recommender data files and generated plots are kept in separate subdirectories beneath the selected output directory.
    ai_dir = output_dir / "ai"
    plots_dir = output_dir / "plots"
    ai_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # The selector dataset is generated from the shared experiment design and then checked against the expected schema and run mode.
    dataset = build_selector_dataset(quick=quick)
    validate_selector_dataset(dataset, quick=quick)

    # The same dataset is evaluated with grouped cross-validation by seed and with leave-one-epsilon-out validation.
    grouped_predictions, grouped_metrics = evaluate_grouped(dataset)
    held_out_predictions, held_out_metrics = evaluate_leave_one_epsilon_out(dataset)

    # Prediction rows from both evaluation procedures are combined into one output table while retaining their evaluation labels.
    predictions = pd.concat([grouped_predictions, held_out_predictions], ignore_index=True)

    # The metrics document combines both evaluations with dataset metadata, fixed baselines and final-model information.
    metrics = build_metrics(dataset, grouped_metrics, held_out_metrics)

    # A final selector is fitted on the complete generated dataset for the exported model visualisations.
    selector = train_selector(dataset)

    # These paths are the recommender-owned CSV and JSON outputs for the selected output directory.
    dataset_path = ai_dir / DATASET_NAME
    predictions_path = ai_dir / PREDICTIONS_NAME
    metrics_path = ai_dir / METRICS_NAME

    # Existing versions of the three recommender data files are removed before the new outputs are written.
    for stale in (dataset_path, predictions_path, metrics_path):
        stale.unlink(missing_ok=True)

    # Tables are written without DataFrame index columns, and the metrics dictionary is written as indented UTF-8 JSON.
    dataset.to_csv(dataset_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    # AI plot generation uses the combined metrics and the selector fitted on the complete dataset.
    plot_paths = make_ai_plots(metrics, selector, plots_dir)

    # The return value keeps the in-memory dataset and metrics together with every path produced by this pipeline.
    return {
        "dataset": dataset,
        "metrics": metrics,
        "paths": [dataset_path, predictions_path, metrics_path, *plot_paths],
    }


def main(argv: list[str] | None = None) -> int:
    # The command-line interface supports either the full selector pipeline or the reduced quick grid.
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

    # Running the pipeline returns the generated dataset, combined metrics and output paths.
    outcome = run(quick=args.quick, output_dir=args.output_dir)
    dataset = outcome["dataset"]
    metrics = outcome["metrics"]

    # These nested dictionaries provide the recommendation metrics and fixed-strategy baselines used in the console summary.
    grouped = metrics["grouped_cross_validation"]["recommendation"]
    held_out = metrics["leave_one_epsilon_out"]["recommendation"]
    baselines = metrics["fixed_baselines"]

    # The console output reports the run mode and selected headline diagnostics from the generated metrics.
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

    # Fixed-strategy means are printed alongside the grouped recommender result, with the oracle given a distinct label.
    for method, values in baselines.items():
        prefix = "oracle best per poll" if method == "oracle" else f"always {method}"
        print(f"    {prefix:<23} {values['mean_selected_l1']:.4f}")

    # Every generated data or plot path returned by the pipeline is listed at the end of the summary.
    for path in outcome["paths"]:
        print(f"  wrote                   : {path}")
    return 0


if __name__ == "__main__":
    # Running the module directly exits with the integer status returned by main.
    raise SystemExit(main())

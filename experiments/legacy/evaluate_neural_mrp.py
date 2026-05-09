"""Justification experiment for RR-aware Neural MRP.

This script is deliberately evaluative, not promotional. It runs the existing
MRP/baseline pipeline across privacy levels, sample sizes, and bias scenarios,
then writes comparison tables that make it easy to see when the neural model
helps, when it does not, and whether its extra runtime/complexity is defensible.

The neural model is trained only on privatized Randomized Response reports via
the RR observation likelihood implemented in ``RRNeuralMRPModel``. True labels
are used only by the simulator to compute evaluation metrics after fitting.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

# Reuse parser and experiment helpers from the main baseline script so this
# constrained evidence wrapper follows the same configuration semantics.
from experiments.mrp_vs_baselines import (
    _parse_eps_list,
    _parse_hidden_layers,
    _parse_list,
    _parse_multipliers,
    _ts_run_dir,
    _write_csv,
    _write_json,
    run_experiment,
)


@dataclass(frozen=True)
class NeuralMRPPreset:
    """Named experiment preset.

    ``small`` is for CI/smoke testing. ``full`` is for final project evidence.
    The full preset is intentionally still finite and CPU-oriented; increase
    trials or training steps only if you have time to report the cost honestly.
    """

    eps: str
    sample_sizes: Tuple[int, ...]
    scenarios: str
    population_n: int
    trials: int
    mrp_steps: int
    neural_steps: int
    neural_hidden_layers: str
    neural_batch_size: int
    mrp_batch_size: int


PRESETS: Dict[str, NeuralMRPPreset] = {
    "small": NeuralMRPPreset(
        eps="0.5,1.0",
        sample_sizes=(120, 250),
        scenarios="simple_linear,nonlinear_interaction",
        population_n=1_000,
        trials=1,
        mrp_steps=5,
        neural_steps=5,
        neural_hidden_layers="8",
        neural_batch_size=128,
        mrp_batch_size=128,
    ),
    "full": NeuralMRPPreset(
        eps="0.2,0.5,1.0,2.0",
        sample_sizes=(500, 1_000, 5_000, 10_000),
        scenarios="simple_linear,nonlinear_interaction,education_urbanicity_interaction,sparse_minority_curve,nonresponse,privacy_helps",
        population_n=100_000,
        trials=5,
        mrp_steps=1_200,
        neural_steps=300,
        neural_hidden_layers="32,16",
        neural_batch_size=512,
        mrp_batch_size=2_048,
    ),
}

SCENARIO_ALIASES: Dict[str, str] = {
    "no_bias": "no_bias",
    "simple_linear": "simple_linear",
    "nonlinear_interaction": "nonlinear_interaction",
    "education_urbanicity_interaction": "education_urbanicity_interaction",
    "sparse_minority_curve": "sparse_minority_curve",
    "nonlinear_response": "nonlinear_response",
    "privacy_noise_sparse": "privacy_noise_sparse",
    "nonresponse": "nonresponse",
    "shy_voter": "shy_fixed",
    "shy_fixed": "shy_fixed",
    "privacy_helps": "shy_privacy_helps",
    "privacy_tradeoff": "privacy_tradeoff",
    "shy_privacy_helps": "shy_privacy_helps",
}

DISPLAY_SCENARIO_NAMES: Dict[str, str] = {
    "no_bias": "no_bias",
    "simple_linear": "simple_linear",
    "nonlinear_interaction": "nonlinear_interaction",
    "education_urbanicity_interaction": "education_urbanicity_interaction",
    "sparse_minority_curve": "sparse_minority_curve",
    "nonlinear_response": "nonlinear_response",
    "privacy_noise_sparse": "privacy_noise_sparse",
    "nonresponse": "nonresponse",
    "shy_fixed": "shy_voter",
    "shy_privacy_helps": "privacy_helps",
    "privacy_tradeoff": "privacy_tradeoff",
}

CORE_METHODS = [
    "baseline_rr_debias",
    "mrp_rr_poststrat",
    "mrp_misreport_rr_poststrat",
    "neural_rr_mrp",
]

COMPARISON_BASELINES = [
    "baseline_rr_debias",
    "mrp_rr_poststrat",
    "mrp_misreport_rr_poststrat",
    "mrp_learned_misreport_rr_poststrat",
]

LOWER_IS_BETTER_METRICS = [
    "mean_overall_l1",
    "mean_overall_mae",
    "mean_worst_region_l1_major",
    "mean_worst_age_l1_major",
    "mean_weighted_region_l1",
    "mean_weighted_age_l1",
    "mean_p90_region_l1_major",
    "mean_p90_age_l1_major",
    "mean_runtime_sec",
]

HIGHER_IS_BETTER_METRICS = ["mean_winner_correct"]


def _parse_sample_sizes(s: str) -> List[int]:
    vals = [int(x.strip()) for x in str(s).split(",") if x.strip()]
    if not vals:
        raise ValueError("Provide --sample_sizes like '500,1000,5000'.")
    if any(v < 1 for v in vals):
        raise ValueError("All sample sizes must be positive integers.")
    return vals


def _normalise_scenarios(scenarios: Sequence[str]) -> List[str]:
    out: List[str] = []
    for raw in scenarios:
        key = raw.strip()
        if not key:
            continue
        if key not in SCENARIO_ALIASES:
            allowed = ", ".join(sorted(SCENARIO_ALIASES))
            raise ValueError(f"Unknown scenario '{key}'. Allowed names/aliases: {allowed}")
        canonical = SCENARIO_ALIASES[key]
        if canonical not in out:
            out.append(canonical)
    if not out:
        raise ValueError("At least one scenario is required.")
    return out


def _display_scenario(canonical: str) -> str:
    return DISPLAY_SCENARIO_NAMES.get(canonical, canonical)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _float_or_nan(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _index_summary(summary_rows: Sequence[Mapping[str, object]]) -> Dict[Tuple[str, float, int, str], Mapping[str, object]]:
    indexed: Dict[Tuple[str, float, int, str], Mapping[str, object]] = {}
    for row in summary_rows:
        if int(row.get("n_rows", 0) or 0) <= 0:
            continue
        key = (
            str(row["scenario"]),
            float(row["epsilon"]),
            int(row["n_sample"]),
            str(row["method"]),
        )
        indexed[key] = row
    return indexed


def build_neural_comparison(summary_rows: Sequence[Mapping[str, object]]) -> List[dict]:
    """Compare neural RR-MRP against each baseline for every condition.

    Deltas are defined as ``neural - baseline``. For error/runtime metrics,
    negative values favour the neural model. For winner correctness, positive
    values favour the neural model.
    """

    indexed = _index_summary(summary_rows)
    conditions = sorted({(s, e, n) for (s, e, n, _m) in indexed.keys()}, key=lambda x: (x[0], x[2], x[1]))
    out: List[dict] = []

    for scenario, eps, n_sample in conditions:
        neural = indexed.get((scenario, eps, n_sample, "neural_rr_mrp"))
        if neural is None:
            continue
        for baseline in COMPARISON_BASELINES:
            base = indexed.get((scenario, eps, n_sample, baseline))
            if base is None:
                continue
            row: dict = {
                "scenario": scenario,
                "scenario_label": _display_scenario(scenario),
                "epsilon": eps,
                "n_sample": n_sample,
                "baseline_method": baseline,
                "neural_method": "neural_rr_mrp",
            }
            for metric in LOWER_IS_BETTER_METRICS:
                nv = _float_or_nan(neural.get(metric))
                bv = _float_or_nan(base.get(metric))
                # Negative delta means neural is better for error metrics.
                delta = nv - bv
                row[f"neural_{metric}"] = nv
                row[f"baseline_{metric}"] = bv
                row[f"delta_{metric}"] = delta
                row[f"neural_better_{metric}"] = int(np.isfinite(delta) and delta < 0.0)
            for metric in HIGHER_IS_BETTER_METRICS:
                nv = _float_or_nan(neural.get(metric))
                bv = _float_or_nan(base.get(metric))
                delta = nv - bv
                row[f"neural_{metric}"] = nv
                row[f"baseline_{metric}"] = bv
                row[f"delta_{metric}"] = delta
                row[f"neural_better_{metric}"] = int(np.isfinite(delta) and delta > 0.0)

            linear_runtime = _float_or_nan(base.get("mean_runtime_sec"))
            neural_runtime = _float_or_nan(neural.get("mean_runtime_sec"))
            if np.isfinite(linear_runtime) and linear_runtime > 0 and np.isfinite(neural_runtime):
                row["runtime_ratio_neural_vs_baseline"] = neural_runtime / linear_runtime
            else:
                row["runtime_ratio_neural_vs_baseline"] = float("nan")

            # A blunt complexity flag: neural must improve both overall error and
            # at least one population-weighted subgroup metric to offset the extra
            # computational cost. This deliberately sets a high bar.
            row["complexity_supported_basic"] = int(
                row.get("neural_better_mean_overall_l1", 0) == 1
                and (
                    row.get("neural_better_mean_weighted_region_l1", 0) == 1
                    or row.get("neural_better_mean_weighted_age_l1", 0) == 1
                )
            )
            out.append(row)
    return out


def build_method_rankings(summary_rows: Sequence[Mapping[str, object]]) -> List[dict]:
    """Rank methods within each condition by key metrics.

    Rank 1 is best. This makes failure cases visible: if neural is not rank 1,
    the output says so without narrative spin.
    """

    grouped: Dict[Tuple[str, float, int], List[Mapping[str, object]]] = {}
    for row in summary_rows:
        if int(row.get("n_rows", 0) or 0) <= 0:
            continue
        key = (str(row["scenario"]), float(row["epsilon"]), int(row["n_sample"]))
        grouped.setdefault(key, []).append(row)

    out: List[dict] = []
    rank_metrics = [
        "mean_overall_l1",
        "mean_weighted_region_l1",
        "mean_weighted_age_l1",
        "mean_runtime_sec",
    ]
    for (scenario, eps, n_sample), rows in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][2], x[0][1])):
        for metric in rank_metrics:
            vals = []
            for row in rows:
                val = _float_or_nan(row.get(metric))
                if np.isfinite(val):
                    vals.append((val, str(row["method"])))
            vals.sort(key=lambda x: x[0])
            for rank, (val, method) in enumerate(vals, start=1):
                out.append({
                    "scenario": scenario,
                    "scenario_label": _display_scenario(scenario),
                    "epsilon": eps,
                    "n_sample": n_sample,
                    "metric": metric,
                    "rank": rank,
                    "method": method,
                    "value": val,
                    "is_neural": int(method == "neural_rr_mrp"),
                })
    return out


def build_neural_verdict(comparison_rows: Sequence[Mapping[str, object]]) -> List[dict]:
    """Aggregate machine-readable verdict rows by baseline method.

    This does not decide that neural is globally justified. It reports win rates
    and average deltas so the dissertation/report can make a cautious claim.
    """

    out: List[dict] = []
    baselines = sorted({str(r["baseline_method"]) for r in comparison_rows})
    for baseline in baselines:
        rows = [r for r in comparison_rows if r["baseline_method"] == baseline]
        if not rows:
            continue
        verdict = {
            "baseline_method": baseline,
            "n_conditions": len(rows),
        }
        for metric in [
            "mean_overall_l1",
            "mean_weighted_region_l1",
            "mean_weighted_age_l1",
            "mean_p90_region_l1_major",
            "mean_p90_age_l1_major",
            "mean_runtime_sec",
        ]:
            delta_key = f"delta_{metric}"
            better_key = f"neural_better_{metric}"
            deltas = np.array([_float_or_nan(r.get(delta_key)) for r in rows], dtype=float)
            deltas = deltas[np.isfinite(deltas)]
            verdict[f"mean_delta_{metric}"] = float(np.mean(deltas)) if deltas.size else float("nan")
            verdict[f"median_delta_{metric}"] = float(np.median(deltas)) if deltas.size else float("nan")
            verdict[f"neural_win_rate_{metric}"] = float(np.mean([int(r.get(better_key, 0)) for r in rows]))
        verdict["complexity_supported_rate_basic"] = float(
            np.mean([int(r.get("complexity_supported_basic", 0)) for r in rows])
        )
        out.append(verdict)
    return out


def _add_common_metadata(rows: Iterable[dict], *, preset: str, n_sample: int) -> List[dict]:
    out: List[dict] = []
    for row in rows:
        r = dict(row)
        r["preset"] = preset
        r["n_sample"] = int(n_sample)
        r["scenario_label"] = _display_scenario(str(r.get("scenario", "")))
        out.append(r)
    return out


def run_justification_experiment(args: argparse.Namespace) -> Path:
    preset = PRESETS[args.preset]

    eps = args.eps if args.eps is not None else preset.eps
    sample_sizes = _parse_sample_sizes(args.sample_sizes) if args.sample_sizes else list(preset.sample_sizes)
    scenarios_raw = _parse_list(args.scenarios if args.scenarios is not None else preset.scenarios)
    scenarios = _normalise_scenarios(scenarios_raw)

    population_n = args.population_n if args.population_n is not None else preset.population_n
    trials = args.trials if args.trials is not None else preset.trials
    mrp_steps = args.mrp_steps if args.mrp_steps is not None else preset.mrp_steps
    neural_steps = args.neural_steps if args.neural_steps is not None else preset.neural_steps
    neural_hidden_layers_raw = args.neural_hidden_layers or preset.neural_hidden_layers
    neural_batch_size = args.neural_batch_size if args.neural_batch_size is not None else preset.neural_batch_size
    mrp_batch_size = args.mrp_batch_size if args.mrp_batch_size is not None else preset.mrp_batch_size

    # Preserve the requested sweep grid exactly; this wrapper chooses smaller
    # defaults but does not alter the experiment methodology.
    eps_list = _parse_eps_list(eps)
    neural_hidden_layers = _parse_hidden_layers(neural_hidden_layers_raw)
    feature_order = _parse_list(args.features)
    strata = _parse_list(args.strata)
    multipliers = _parse_multipliers(args.multipliers)

    run_dir = _ts_run_dir(Path(args.out_dir), f"neural_mrp_justification_{args.preset}")

    config = {
        "purpose": "Evaluate whether RR-aware Neural MRP is justified against simpler baselines.",
        "preset": args.preset,
        "k": args.k,
        "eps": eps_list,
        "sample_sizes": sample_sizes,
        "scenarios_requested": scenarios_raw,
        "scenarios_canonical": scenarios,
        "population_n": population_n,
        "trials": trials,
        "seed": args.seed,
        "sampling": args.sampling,
        "strata": strata,
        "allocation": args.allocation,
        "min_per_stratum": args.min_per_stratum,
        "biased_feature": args.biased_feature,
        "multipliers": multipliers,
        "features": feature_order,
        "shy_category": args.shy_category,
        "shy_honesty": args.shy_honesty,
        "mrp_steps": mrp_steps,
        "mrp_lr": args.mrp_lr,
        "mrp_l2": args.mrp_l2,
        "mrp_batch_size": mrp_batch_size,
        "enable_neural": not args.disable_neural,
        "neural_hidden_layers": list(neural_hidden_layers),
        "neural_steps": neural_steps,
        "neural_lr": args.neural_lr,
        "neural_batch_size": neural_batch_size,
        "neural_seed": args.neural_seed,
        "neural_dropout": args.neural_dropout,
        "neural_weight_decay": args.neural_weight_decay,
        "major_mass": args.major_mass,
        "core_methods_expected": CORE_METHODS,
        "comparison_baselines": COMPARISON_BASELINES,
        "notes": [
            "Training uses privatized reported RR labels; true simulator labels are used only for evaluation.",
            "Negative error deltas in neural_comparison.csv favour neural; positive runtime deltas mean neural is slower.",
            "Do not claim neural is better unless the output tables support that for the chosen scenarios.",
        ],
    }
    _write_json(run_dir / "config.json", config)

    # Rows remain machine-readable so downstream tables can be reproduced from
    # the saved CSV files without re-running the experiment.
    all_rows: List[dict] = []
    all_summary: List[dict] = []

    for n_sample in sample_sizes:
        rows, summary = run_experiment(
            k=args.k,
            eps_list=eps_list,
            scenarios=scenarios,
            population_n=population_n,
            n_sample=n_sample,
            trials=trials,
            seed=args.seed,
            sampling=args.sampling,
            strata=strata,
            allocation=args.allocation,
            min_per_stratum=args.min_per_stratum,
            biased_feature=args.biased_feature,
            biased_multipliers=multipliers,
            feature_order=feature_order,
            shy_category=args.shy_category,
            shy_honesty=args.shy_honesty,
            mrp_steps=mrp_steps,
            mrp_lr=args.mrp_lr,
            mrp_l2=args.mrp_l2,
            mrp_batch_size=mrp_batch_size,
            verbose_every=args.verbose_every,
            enable_neural=not args.disable_neural,
            neural_hidden_layers=neural_hidden_layers,
            neural_steps=neural_steps,
            neural_lr=args.neural_lr,
            neural_batch_size=neural_batch_size,
            neural_seed=args.neural_seed,
            neural_dropout=args.neural_dropout,
            neural_weight_decay=args.neural_weight_decay,
            major_mass=args.major_mass,
        )
        all_rows.extend(_add_common_metadata(rows, preset=args.preset, n_sample=n_sample))
        all_summary.extend(_add_common_metadata(summary, preset=args.preset, n_sample=n_sample))

    comparison = build_neural_comparison(all_summary)
    rankings = build_method_rankings(all_summary)
    verdict = build_neural_verdict(comparison)

    # Save all artefacts in machine-readable formats so downstream reporting
    # (tables, verdict documents) can be regenerated from the CSV/JSONL files
    # without re-running the full experiment.
    _write_csv(run_dir / "results_trials.csv", all_rows)
    _write_jsonl(run_dir / "results_trials.jsonl", all_rows)
    _write_csv(run_dir / "summary.csv", all_summary)
    _write_jsonl(run_dir / "summary.jsonl", all_summary)
    _write_csv(run_dir / "neural_comparison.csv", comparison)
    _write_jsonl(run_dir / "neural_comparison.jsonl", comparison)
    _write_csv(run_dir / "method_rankings.csv", rankings)
    _write_csv(run_dir / "neural_verdict.csv", verdict)
    _write_json(run_dir / "neural_verdict.json", {"rows": verdict})

    return run_dir


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate whether RR-aware Neural MRP is justified against simpler baselines."
    )
    p.add_argument("--preset", choices=sorted(PRESETS), default="small")
    p.add_argument("--out_dir", default="experiments/outputs")

    # Optional overrides. If omitted, preset values are used.
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--eps", type=str, default=None, help="Override preset eps list, e.g. '0.2,0.5,1.0,2.0'.")
    p.add_argument("--sample_sizes", type=str, default=None, help="Override preset sample sizes, e.g. '500,1000'.")
    p.add_argument("--scenarios", type=str, default=None, help="Override preset scenarios. Aliases: shy_voter, privacy_helps.")
    p.add_argument("--population_n", type=int, default=None)
    p.add_argument("--trials", type=int, default=None)
    p.add_argument("--seed", type=int, default=123)

    # Sampling and bias controls, kept aligned with mrp_vs_baselines.py.
    p.add_argument("--sampling", choices=["srs", "stratified", "biased"], default="srs")
    p.add_argument("--strata", type=str, default="region")
    p.add_argument("--allocation", choices=["proportional", "equal", "sqrt"], default="proportional")
    p.add_argument("--min_per_stratum", type=int, default=0)
    p.add_argument("--biased_feature", type=str, default="region")
    p.add_argument("--multipliers", type=str, default="")
    p.add_argument("--shy_category", type=int, default=0)
    p.add_argument("--shy_honesty", type=float, default=0.80)

    # Model controls.
    p.add_argument("--features", type=str, default="region,age_group,education,gender,urbanicity")
    p.add_argument("--mrp_steps", type=int, default=None)
    p.add_argument("--mrp_lr", type=float, default=0.05)
    p.add_argument("--mrp_l2", type=float, default=1.0)
    p.add_argument("--mrp_batch_size", type=int, default=None)
    p.add_argument("--verbose_every", type=int, default=0)

    p.add_argument("--disable_neural", action="store_true", help="For infrastructure debugging only; defeats the main purpose.")
    p.add_argument("--neural_hidden_layers", type=str, default=None)
    p.add_argument("--neural_steps", type=int, default=None)
    p.add_argument("--neural_lr", type=float, default=0.01)
    p.add_argument("--neural_batch_size", type=int, default=None)
    p.add_argument("--neural_seed", type=int, default=321)
    p.add_argument("--neural_dropout", type=float, default=0.0)
    p.add_argument("--neural_weight_decay", type=float, default=1e-4)

    p.add_argument("--major_mass", type=float, default=0.02)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = run_justification_experiment(args)
    print(f"Saved neural-MRP justification experiment to: {run_dir}")
    print(f"- {run_dir / 'config.json'}")
    print(f"- {run_dir / 'results_trials.csv'}")
    print(f"- {run_dir / 'summary.csv'}")
    print(f"- {run_dir / 'neural_comparison.csv'}")
    print(f"- {run_dir / 'method_rankings.csv'}")
    print(f"- {run_dir / 'neural_verdict.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

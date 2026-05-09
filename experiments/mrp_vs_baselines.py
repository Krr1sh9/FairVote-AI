"""Compare RR debiasing, MRP baselines, and neural RR-MRP on simulations.

This module is now a thin CLI/compatibility layer. The professional experiment
pipeline lives in ``experiments.pipeline`` where config parsing, population
construction, sampling, perturbation, method runners, metrics, output writing,
and summary generation are separated into auditable modules.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from experiments.pipeline.config import ExperimentConfig, default_methods, resolve_methods
from experiments.pipeline.io import write_csv as _write_csv
from experiments.pipeline.io import write_experiment_outputs, write_json as _write_json
from experiments.pipeline.methods import METHOD_REGISTRY, misreport_to_matrix as _misreport_to_matrix
from experiments.pipeline.parsing import (
    parse_eps_list as _parse_eps_list,
    parse_hidden_layers as _parse_hidden_layers,
    parse_int_list as _parse_int_list,
    parse_list as _parse_list,
    parse_multipliers as _parse_multipliers,
)
from experiments.pipeline.plotting import plot_summary as _plot_summary
from experiments.pipeline.presets import get_preset, preset_names
from experiments.pipeline.runner import execute_experiment


def _ts_run_dir(base: Path, name: str) -> Path:
    """Create a unique timestamped output directory.

    Kept here for backwards compatibility with older tests and scripts that
    monkeypatch the ``datetime`` symbol in this module.
    """
    base.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    for i in range(1000):
        suffix = "" if i == 0 else f"_{i:03d}"
        run_dir = base / f"{ts}_{name}{suffix}"
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            (run_dir / "plots").mkdir(parents=True, exist_ok=True)
            return run_dir
        except FileExistsError:
            continue
    ts2 = datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")
    run_dir = base / f"{ts2}_{name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    return run_dir


def _experiment_methods(enable_neural: bool) -> List[str]:
    """Backward-compatible wrapper around the canonical method list."""
    return default_methods(enable_neural)


def run_experiment(
    *,
    k: int,
    eps_list: List[float],
    scenarios: List[str],
    population_n: int,
    n_sample: int,
    trials: int,
    seed: int,
    sampling: str,
    strata: List[str],
    allocation: str,
    min_per_stratum: int,
    biased_feature: str,
    biased_multipliers: Dict[str, float],
    feature_order: List[str],
    shy_category: int,
    shy_honesty: float,
    mrp_steps: int,
    mrp_lr: float,
    mrp_l2: float,
    mrp_batch_size: int,
    verbose_every: int,
    enable_neural: bool,
    neural_hidden_layers: Tuple[int, ...],
    neural_steps: int,
    neural_lr: float,
    neural_batch_size: int,
    neural_seed: int,
    neural_dropout: float,
    neural_weight_decay: float,
    major_mass: float,
    neural_validation_fraction: float = 0.2,
    neural_patience: int = 20,
    methods: List[str] | None = None,
    sample_sizes: List[int] | None = None,
) -> Tuple[List[dict], List[dict]]:
    """Run an experiment grid and return raw trial rows plus summary rows.

    The signature is preserved for ``evaluate_neural_mrp.py`` and older docs.
    New code should prefer constructing ``ExperimentConfig`` and calling
    ``experiments.pipeline.runner.execute_experiment`` directly.
    """
    config = ExperimentConfig(
        k=k,
        eps_list=eps_list,
        scenarios=scenarios,
        population_n=population_n,
        n_sample=n_sample,
        trials=trials,
        seed=seed,
        sampling=sampling,
        strata=strata,
        allocation=allocation,
        min_per_stratum=min_per_stratum,
        biased_feature=biased_feature,
        biased_multipliers=biased_multipliers,
        feature_order=feature_order,
        shy_category=shy_category,
        shy_honesty=shy_honesty,
        mrp_steps=mrp_steps,
        mrp_lr=mrp_lr,
        mrp_l2=mrp_l2,
        mrp_batch_size=mrp_batch_size,
        verbose_every=verbose_every,
        enable_neural=enable_neural,
        neural_hidden_layers=neural_hidden_layers,
        neural_steps=neural_steps,
        neural_lr=neural_lr,
        neural_batch_size=neural_batch_size,
        neural_seed=neural_seed,
        neural_dropout=neural_dropout,
        neural_weight_decay=neural_weight_decay,
        major_mass=major_mass,
        neural_validation_fraction=neural_validation_fraction,
        neural_patience=neural_patience,
        methods=methods or [],
        sample_sizes=sample_sizes or [],
    )
    result = execute_experiment(config)
    return result.rows, result.summary


def _explicit_options(argv: list[str]) -> set[str]:
    """Return argparse option names explicitly present on the command line."""
    out: set[str] = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        name = token[2:].split("=", 1)[0].replace("-", "_")
        out.add(name)
    return out


def _apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    preset = get_preset(getattr(args, "preset", "custom"))
    if preset is None:
        args.preset = "custom"
        return args
    explicit = set(getattr(args, "_explicit_options", set()))
    for key, value in preset.overrides.items():
        if key not in explicit:
            setattr(args, key, value)
    args.preset = preset.name
    return args


def _config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    args = _apply_preset(args)
    eps_list = _parse_eps_list(args.eps)
    scenarios = _parse_list(args.scenarios)
    sample_sizes = _parse_int_list(args.sample_sizes) if str(args.sample_sizes).strip() else [int(args.n_sample)]
    return ExperimentConfig(
        k=args.k,
        eps_list=eps_list,
        scenarios=scenarios,
        population_n=args.population_n,
        n_sample=int(sample_sizes[0]),
        trials=args.trials,
        seed=args.seed,
        sampling=args.sampling,
        strata=_parse_list(args.strata),
        allocation=args.allocation,
        min_per_stratum=args.min_per_stratum,
        biased_feature=args.biased_feature,
        biased_multipliers=_parse_multipliers(args.multipliers),
        feature_order=_parse_list(args.features),
        shy_category=args.shy_category,
        shy_honesty=args.shy_honesty,
        mrp_steps=args.mrp_steps,
        mrp_lr=args.mrp_lr,
        mrp_l2=args.mrp_l2,
        mrp_batch_size=args.mrp_batch_size,
        verbose_every=args.verbose_every,
        enable_neural=not args.disable_neural,
        neural_hidden_layers=_parse_hidden_layers(args.neural_hidden_layers),
        neural_steps=args.neural_steps,
        neural_lr=args.neural_lr,
        neural_batch_size=args.neural_batch_size,
        neural_seed=args.neural_seed,
        neural_dropout=args.neural_dropout,
        neural_weight_decay=args.neural_weight_decay,
        major_mass=args.major_mass,
        neural_validation_fraction=args.neural_validation_fraction,
        neural_patience=args.neural_patience,
        methods=resolve_methods(args.methods, enable_neural=not args.disable_neural),
        sample_sizes=sample_sizes,
        preset=args.preset,
        continue_on_error=not args.fail_fast,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare baseline RR debias vs RR-aware MRP + poststratification.")
    parser.add_argument("--preset", choices=["custom", *preset_names()], default="custom", help="Run preset: smoke_test, medium_evidence, final_evidence, or custom.")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--eps", type=str, default="0.2,0.5,1.0,2.0")
    parser.add_argument("--sample_sizes", type=str, default="", help="Comma-separated respondent sample sizes, e.g. '500,1000,2500'.")
    parser.add_argument("--scenarios", type=str, default="simple_linear,nonlinear_interaction,nonresponse,shy_privacy_helps")
    parser.add_argument("--population_n", type=int, default=100_000)
    parser.add_argument("--n_sample", type=int, default=5_000)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out_dir", type=str, default="experiments/outputs")
    parser.add_argument("--sampling", choices=["srs", "stratified", "biased"], default="srs")
    parser.add_argument("--strata", type=str, default="region")
    parser.add_argument("--allocation", choices=["proportional", "equal", "sqrt"], default="proportional")
    parser.add_argument("--min_per_stratum", type=int, default=0)
    parser.add_argument("--biased_feature", type=str, default="region")
    parser.add_argument("--multipliers", type=str, default="")
    parser.add_argument("--shy_category", type=int, default=0)
    parser.add_argument("--shy_honesty", type=float, default=0.80)
    parser.add_argument("--features", type=str, default="region,age_group,education,gender,urbanicity")
    parser.add_argument("--mrp_steps", type=int, default=1200)
    parser.add_argument("--mrp_lr", type=float, default=0.05)
    parser.add_argument("--mrp_l2", type=float, default=1.0)
    parser.add_argument("--mrp_batch_size", type=int, default=2048)
    parser.add_argument("--verbose_every", type=int, default=0, help="Set >0 to print training logs periodically.")
    parser.add_argument("--disable_neural", action="store_true", help="Disable the PyTorch RR-aware neural MRP estimator.")
    parser.add_argument("--neural_hidden_layers", type=str, default="16")
    parser.add_argument("--neural_steps", type=int, default=200)
    parser.add_argument("--neural_lr", type=float, default=0.01)
    parser.add_argument("--neural_batch_size", type=int, default=512)
    parser.add_argument("--neural_seed", type=int, default=321)
    parser.add_argument("--neural_dropout", type=float, default=0.0)
    parser.add_argument("--neural_weight_decay", type=float, default=1e-4)
    parser.add_argument("--neural_validation_fraction", type=float, default=0.2)
    parser.add_argument("--neural_patience", type=int, default=20)
    parser.add_argument(
        "--methods",
        type=str,
        default="default",
        help="Method preset (default, research, all) or comma-separated method names.",
    )
    parser.add_argument(
        "--major_mass",
        type=float,
        default=0.02,
        help="Only treat groups with population share >= major_mass as major groups for worst/p90 metrics.",
    )
    parser.add_argument("--fail_fast", action="store_true", help="Stop immediately if one estimator fails instead of writing partial-run failure rows.")
    parser.add_argument("--skip_plots", action="store_true", help="Skip optional matplotlib plots; CSV/JSON evidence is still written.")
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = _build_parser().parse_args(argv)
    args._explicit_options = _explicit_options(argv)
    config = _config_from_args(args)
    base_out = Path(args.out_dir)
    run_dir = _ts_run_dir(base_out, "mrp_vs_baselines")

    result = execute_experiment(config)
    write_experiment_outputs(run_dir, result)
    if not args.skip_plots:
        _plot_summary(run_dir, result.summary)

    print(f"Saved run to: {run_dir}")
    print(f"- {run_dir / 'config.json'}")
    print(f"- {run_dir / 'manifest.json'}")
    print(f"- {run_dir / 'summary_with_ci.csv'}")
    print(f"- {run_dir / 'raw_trials.csv'}")
    print(f"- {run_dir / 'paired_comparisons.csv'}")
    print(f"- {run_dir / 'ablations.csv'}")
    print(f"- {run_dir / 'runtime_profile.csv'}")
    print(f"- {run_dir / 'README.md'}")
    if not args.skip_plots:
        print(f"- {run_dir / 'plots'} (if matplotlib installed)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

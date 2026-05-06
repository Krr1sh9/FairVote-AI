# Experiment Pipeline Architecture

The main MRP/baseline experiment is deliberately split into a thin CLI wrapper and testable pipeline modules. This makes the evidence generation auditable: a marker can see where the population is created, where sampling and Randomized Response happen, where each estimator is run, and where metrics are written.

## Before the refactor

```text
experiments/mrp_vs_baselines.py
  - CLI parsing
  - population generation
  - sampling
  - bias scenarios
  - RR perturbation
  - all estimator implementations
  - metric calculations
  - summary aggregation
  - CSV/JSON writing
  - plotting
```

This made the main experiment hard to inspect because one large `run_experiment` function mixed configuration, simulation, method execution, metrics, and output writing.

## Current structure

```text
experiments/
  mrp_vs_baselines.py          # thin CLI and backwards-compatible wrappers
  pipeline/
    config.py                  # ExperimentConfig, TrialConfig, MethodResult, ExperimentResult
    context.py                 # population generation and poststratification context
    io.py                      # JSON/CSV/run-directory/README writing
    methods/                   # split method registry and estimator runners
    metrics.py                 # metric and result-row generation
    parsing.py                 # CLI parsing helpers
    perturbation.py            # misreporting + RR perturbation stage
    plotting.py                # optional matplotlib plots
    runner.py                  # experiment-grid orchestration
    sampling.py                # sample-frame and nonresponse stages
    summary.py                 # summary/CI, paired-comparison, ablation, runtime aggregation
```

## Reproducibility guarantees

Every raw result row includes the key configuration needed to trace the result:

- `config_seed`
- `random_seed`
- `sample_seed`
- `scenario`
- `epsilon`
- `sample_size`
- `population_n`
- `sampling`
- `method`
- `runtime_sec`
- metric columns such as `overall_l1`, `overall_mae`, `winner_correct`, and subgroup error metrics

Each run now writes the final-evidence artefacts directly:

```text
config.json
manifest.json
raw_trials.csv
summary_with_ci.csv
paired_comparisons.csv
ablations.csv
runtime_profile.csv
failures.csv
README.md
plots/
```

For backwards compatibility, the writer also emits `results_trials.csv` and `summary.csv` as aliases of the raw and CI summary tables. The `manifest.json` records the full config, requested scenarios, epsilons, sample-size grid, methods, row counts, output filenames, failures, and total runtime. This prevents the evidence pack from becoming a collection of unexplained CSVs.

## Method registry

Estimator methods are plugged in through `experiments/pipeline/methods/`:

```text
METHOD_REGISTRY = {
  "oracle_true_sample_distribution": ...,
  "raw_reported_distribution": ...,
  "baseline_rr_debias": ...,
  "linear_rr_no_poststrat": ...,
  "mrp_rr_poststrat": ...,
  "oracle_true_linear_mrp_poststrat": ...,
  "mrp_misreport_rr_poststrat": ...,
  "oracle_known_misreport_rr_mrp": ...,
  "mrp_learned_misreport_rr_poststrat": ...,
  "neural_rr_mrp": ...,
  "neural_naive_reported_mrp": ...,
}
```

Adding a new estimator now means adding a runner function and registry entry. It should not require editing a giant experiment function.

## Regression tests

`tests/test_experiment_pipeline.py` checks that:

- all requested methods appear in raw and summary outputs;
- all requested scenarios, epsilons, and sample sizes appear;
- required metric/config columns are present;
- summary means match the raw trial rows;
- the manifest matches the run config, preset, sample-size grid, and output row counts;
- research scenarios and oracle/ablation methods can be requested;
- paired neural-minus-linear comparison tables report deltas and win rates.

These tests are intended to protect the experiment evidence from silent schema drift.

## Research preset

Use `--methods research` to include oracle diagnostics, no-poststratification ablations, the canonical RR-aware linear poststratification/MRP baseline, the RR-aware neural model, and the naive neural-without-RR-aware-loss ablation. The run writes `paired_comparisons.csv` whenever both `mrp_rr_poststrat` and `neural_rr_mrp` are present.

See `docs/research_contribution.md` for the research question, scenario definitions, and interpretation rules.


## Evidence presets

The CLI supports three named evidence presets:

```bash
python -m experiments.mrp_vs_baselines --preset smoke_test
python -m experiments.mrp_vs_baselines --preset medium_evidence
python -m experiments.mrp_vs_baselines --preset final_evidence
```

- `smoke_test` is only a sanity check and deliberately uses minimal settings.
- `medium_evidence` is for development and draft-report tables.
- `final_evidence` uses epsilons `0.2,0.5,1.0,2.0`, sample sizes `500,1000,2500`, scenarios `simple_linear,nonresponse,nonlinear_interaction,shy_fixed`, and at least 30 trials unless explicitly overridden. It does not use 5-step minimal training.

The final preset can be made stronger with, for example, `--trials 50` or `--trials 100` if runtime permits.

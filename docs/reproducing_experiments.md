# Reproducing Experiments

This is the canonical command guide for the current modular experiment pipeline. Historical development scripts remain in the repository for compatibility, but final-submission evidence should be generated through `experiments.mrp_vs_baselines` and the named presets below.

## Install

From a clean virtual environment:

```bash
pip install -e ".[experiments,neural]"
```

For full local verification, use:

```bash
pip install -e ".[dev]"
```

## Smoke test

Use this only to verify that the pipeline runs and writes files:

```bash
python -m experiments.mrp_vs_baselines --preset smoke_test
```

The `smoke_test` preset is intentionally small and uses minimal training settings. It must not be used for final quantitative claims.

## Medium evidence run

Use this for development checks or draft-report tables when compute time is limited:

```bash
python -m experiments.mrp_vs_baselines --preset medium_evidence
```

## Final evidence run

Use this for final report evidence:

```bash
python -m experiments.mrp_vs_baselines --preset final_evidence
```

The full final preset uses the wider robustness grid defined in `experiments/pipeline/presets.py`:

```text
epsilons:     0.2, 0.5, 1.0, 2.0, 4.0
sample sizes: 500, 1000, 2500, 5000
scenarios:    full robustness set, including simple, nonlinear, sparse, privacy-noise and misreport scenarios
trials:       30 by default
methods:      research method set
```

The submitted canonical evidence run is a smaller CPU-sized final-style custom run at `evidence/final/2026-05-06_004647_mrp_vs_baselines/`. It should be cited as custom/reduced evidence, not as the full preset.

Increase trial count if runtime allows:

```bash
python -m experiments.mrp_vs_baselines --preset final_evidence --trials 50
python -m experiments.mrp_vs_baselines --preset final_evidence --trials 100
```

## Custom research run

Example focused neural-vs-linear run:

```bash
python -m experiments.mrp_vs_baselines \
  --methods research \
  --scenarios simple_linear,nonlinear_interaction,nonresponse,shy_fixed \
  --eps 0.5,1.0 \
  --sample_sizes 500,1000 \
  --trials 10
```

## Output files

Each run writes a timestamped directory under `experiments/outputs/` with:

| File | Purpose |
|---|---|
| `raw_trials.csv` | One row per method/condition/trial with config, seed, scenario, epsilon, sample size, method, metrics and runtime. |
| `summary_with_ci.csv` | Mean metrics and 95% confidence intervals over repeated trials. |
| `paired_comparisons.csv` | Paired neural-minus-linear deltas, win rates and paired bootstrap intervals. |
| `ablations.csv` | Comparisons against the canonical RR-aware linear poststratification/MRP baseline. |
| `runtime_profile.csv` | Runtime summaries by method and condition. |
| `failures.csv` | Method-level failures if `continue_on_error` is enabled. |
| `config.json` | Exact reproducible configuration. |
| `manifest.json` | Run manifest with row counts, methods, scenarios and output filenames. |
| `README.md` | Run-local explanation of the preset and interpretation cautions. |
| `plots/` | Optional plots generated from the summary. |

For compatibility, the pipeline may also write `results_trials.csv` and `summary.csv`. Prefer the newer final-evidence files for report tables.

## Failure handling and reproducibility

- Seeds are stored in `config.json`, `manifest.json`, and each raw trial row.
- Method runtime is stored per row and summarised in `runtime_profile.csv`.
- If a method fails and `continue_on_error` is true, the failure is logged in `failures.csv` and the rest of the run continues.
- Use `--fail_fast` when debugging a failing estimator, or for final evidence when you want strict failure handling so hidden partial failures cannot pass as complete evidence.

Example:

```bash
python -m experiments.mrp_vs_baselines --preset smoke_test --fail_fast
```

## Interpreting evidence

Use [`evidence_interpretation.md`](evidence_interpretation.md). In short:

- Lower L1 error is better.
- Lower worst-group L1 is better for subgroup robustness.
- Negative neural-minus-linear deltas favour RR-aware Neural MRP.
- Paired comparisons are more informative than unpaired mean differences.
- Runtime is part of the practical engineering trade-off.
- Synthetic evidence does not prove real-election accuracy.

## Legacy scripts

Some older helper scripts remain for compatibility or specialised analysis, including `experiments.evaluate_neural_mrp`, `experiments.add_uncertainty_summaries`, `experiments.sweep_eps`, and `experiments.make_report_tables`. They are not the canonical final-evidence path. The canonical path is:

```bash
python -m experiments.mrp_vs_baselines --preset final_evidence
```

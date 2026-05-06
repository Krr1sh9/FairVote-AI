# FairVote-AI experiment run

Preset: `custom`

This run is intended as CPU-sized statistical evidence; inspect confidence intervals, iteration counts, and paired deltas before making final claims.

## Configuration summary

- Seed: `123`
- Scenarios: `nonlinear_interaction, sparse_minority_curve, privacy_noise_sparse, simple_linear`
- Epsilons: `0.5, 1.0, 2.0`
- Sample sizes: `500, 1000`
- Trials per cell: `20`
- Population size: `10000`
- Methods: `baseline_rr_debias, mrp_rr_poststrat, hierarchical_rr_mrp_poststrat, neural_rr_mrp`
- MRP steps: `800`
- Neural steps: `200`
- Neural patience: `30`

## Files

- `raw_trials.csv`: one row per method × sample size × scenario × epsilon × trial.
- `summary_with_ci.csv`: means, standard deviations and 95% bootstrap CIs over trials.
- `paired_comparisons.csv`: paired neural-minus-linear deltas, bootstrap CIs and win rates.
- `ablations.csv`: paired ablation deltas against the canonical linear RR-aware MRP baseline.
- `runtime_profile.csv`: runtime/failure/skipped counts by method and condition.
- `config.json`: complete reproducible configuration.
- `manifest.json`: run manifest, output map, row counts, failures and total runtime.
- `environment.json`: Python/platform/package provenance copied from the manifest.
- `sha256sums.txt`: SHA-256 hashes for output integrity.
- `failures.csv`: method-level errors if `continue_on_error=True` allowed a partial run.

## Interpretation rules

Negative neural-minus-linear deltas mean neural RR-MRP had lower error than the linear RR-aware MRP baseline.
Treat a claim as supported only when the paired CI is mostly on the same side of zero and the win rate is convincing.
Do not treat oracle baselines as deployable methods; they use synthetic true labels or known misreport structure.

## Run status

- Result rows: `1920`
- Summary rows: `96`
- Paired comparison rows: `24`
- Ablation rows: `24`
- Runtime profile rows: `96`
- Failures: `0`
- Total runtime seconds: `488.924`

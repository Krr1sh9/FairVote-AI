# FairVote-AI experiment run

Preset: `smoke_test`

This is a smoke/sanity run only; do not use it as final statistical evidence.

## Configuration summary

- Seed: `123`
- Scenarios: `simple_linear`
- Epsilons: `1.0`
- Sample sizes: `120`
- Trials per cell: `1`
- Population size: `1500`
- Methods: `raw_reported_distribution, baseline_rr_debias, mrp_rr_poststrat, mrp_misreport_rr_poststrat, mrp_learned_misreport_rr_poststrat`
- MRP steps: `5`
- Neural steps: `5`
- Neural patience: `2`

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

- Result rows: `5`
- Summary rows: `5`
- Paired comparison rows: `0`
- Ablation rows: `2`
- Runtime profile rows: `5`
- Failures: `0`
- Total runtime seconds: `0.027`

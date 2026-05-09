# FairVote-AI canonical final-style evidence run

Preset: `custom`

This is the **canonical report-ready evidence run** for the submitted archive. It is a CPU-sized final-style run designed to be inspectable and rerunnable on ordinary hardware. It should not be described as the full `final_evidence` preset.

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
- Failure policy: strict fail-fast semantics (`continue_on_error=false`); the run completed with zero recorded failures.

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
- `failures.csv`: method-level errors if any occurred. This file is empty for this run.

## Run status

- Result rows: `1920`
- Summary rows: `96`
- Paired comparison rows: `24`
- Ablation rows: `24`
- Runtime profile rows: `96`
- Failures: `0`
- Total runtime seconds: `488.924`

## Interpretation rules

Negative neural-minus-linear deltas mean neural RR-MRP had lower error than the linear RR-aware MRP baseline. Treat a neural improvement claim as supported only when the paired CI is mostly on the same side of zero and the win rate is convincing.

The main conclusion from this run is mixed and conditional:

- Neural RR-MRP often improves over the linear `mrp_rr_poststrat` baseline: for overall L1, 22 of 24 paired cells have negative mean deltas and 18 of 24 have confidence intervals wholly below zero.
- Neural RR-MRP is not the overall best method in most conditions when compared with all methods. For best mean overall L1, `baseline_rr_debias` is best in 13 of 24 cells, `hierarchical_rr_mrp_poststrat` in 10 of 24, and `neural_rr_mrp` in 1 of 24.
- Hierarchical partial pooling is the strongest subgroup-error method in this run: for major worst-group L1, `hierarchical_rr_mrp_poststrat` is best in 23 of 24 cells.
- Therefore the report should state that neural modelling is a valid privacy-compatible learned estimator and often improves over the linear MRP baseline in paired comparisons, but should not claim universal neural superiority.

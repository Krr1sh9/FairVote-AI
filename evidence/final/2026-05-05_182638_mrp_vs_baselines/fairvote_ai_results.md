# FairVote-AI: RR-aware MRP for locally private polling

## Abstract

FairVote-AI evaluates locally private polling estimators under k-ary Randomized Response, sampling bias, sparse subgroup structure, nonlinear response patterns and strategic misreporting. The submitted implementation includes a browser-side LDP respondent prototype, raw-answer rejection on the server, strict analyst upload validation, analytical RR debiasing, linear RR-aware MRP, hierarchical partial-pooling RR-aware MRP, optional neural RR-MRP, and a reproducible evidence pipeline.

## Method summary

The central modelling assumption is explicit: training observes only randomized-response reports. RR-aware MRP estimators optimise reported-label likelihood through the known RR transition matrix and then poststratify estimated latent choice probabilities over population cells. The hierarchical estimator adds feature-level varying effects with shrinkage, targeting sparse-cell stability without storing raw answers.

## Evidence run used for this paper draft

- Run directory: `evidence/final/2026-05-05_182638_mrp_vs_baselines`
- Preset: `custom`
- Command: `not recorded`

## Evidence quality checks

| Check | Status | Evidence |
| --- | --- | --- |
| Run is not labelled smoke/non-final | PASS | README wording |
| No recorded failures | PASS | 0 failure rows |
| At least 20 non-skipped trials per summary cell | PASS | minimum n_rows=20; configured trials=20 |
| No skipped result cells | PASS | max n_skipped=0 |
| Neural-vs-linear paired comparisons are available | WARN | 0 paired rows |

## Main results

| scenario | method | best_epsilon | sample_size | mean_overall_l1 | ci95_low | ci95_high | n_rows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| privacy_helps | baseline_rr_debias | 2.0 | 120 | 0.2520 | 0.2101 | 0.2982 | 20 |
| privacy_helps | hierarchical_rr_mrp_poststrat | 2.0 | 120 | 0.4918 | 0.4834 | 0.4995 | 20 |
| privacy_helps | mrp_rr_poststrat | 2.0 | 120 | 0.4597 | 0.4354 | 0.4849 | 20 |

## Claims and support status

| Claim | Evidence file | Status | Required interpretation rule |
| --- | --- | --- | --- |
| RR-aware MRP improves over direct RR debiasing | summary_with_ci.csv / ablations.csv | SUPPORTED TO ANALYSE | Compare mean_overall_l1 and CI by scenario; do not claim improvement unless deltas are negative and stable. |
| Hierarchical partial pooling improves sparse subgroup error | summary_with_ci.csv | MISSING EVIDENCE | Use worst_group_l1_major in sparse scenarios; report failures as mixed results. |
| Neural RR-MRP beats simpler baselines in nonlinear settings | paired_comparisons.csv | MISSING PAIRED COMPARISONS | Only claim this if paired deltas have enough trials and confidence intervals exclude zero where relevant. |
| Privacy can help under strategic misreporting | summary_with_ci.csv | SUPPORTED TO ANALYSE | Look for intermediate-epsilon minima; do not overgeneralise to pure privacy noise scenarios. |

## Limitations

This paper draft is only as strong as the evidence run above. If the run is smoke-labelled, has fewer than 20 repeated trials per cell, has failures, or lacks paired comparisons, the corresponding claims must be reported as preliminary or unsupported. Randomized response protects answer values only; demographic minimisation, rare-cell reporting and export controls are separate privacy boundaries.

## Reproducibility checklist

- `summary_with_ci.csv` or `summary.csv` present.
- `paired_comparisons.csv` present when neural claims are made.
- `failures.csv` empty for final claims.
- `manifest.json`, `environment.json` and `sha256sums.txt` present.
- Acceptance checks above pass before this text is used as a final report-ready result.

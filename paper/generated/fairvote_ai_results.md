# FairVote-AI: RR-aware MRP for locally private polling

## Abstract

FairVote-AI evaluates locally private polling estimators under k-ary Randomized Response, sampling bias, sparse subgroup structure, nonlinear response patterns and privacy-noise stress tests. The wider experiment pipeline also supports strategic-misreporting scenarios, but the canonical evidence run used here should be interpreted as a reduced, CPU-sized final-style benchmark rather than a full robustness sweep. The submitted implementation includes a browser-side LDP respondent prototype, raw-answer rejection on the server, strict analyst upload validation, analytical RR debiasing, linear RR-aware MRP, hierarchical partial-pooling RR-aware MRP, optional neural RR-MRP, and a reproducible evidence pipeline.

## Method summary

The central modelling assumption is explicit: training observes only randomized-response reports. RR-aware MRP estimators optimise reported-label likelihood through the known RR transition matrix and then poststratify estimated latent choice probabilities over population cells. The hierarchical estimator adds feature-level varying effects with shrinkage, targeting sparse-cell stability without storing raw answers.

## Evidence run used for this paper draft

- Run directory: `evidence/final/2026-05-06_004647_mrp_vs_baselines`
- Preset: `custom`
- Command: `python -m experiments.mrp_vs_baselines --trials 20 --population_n 10000 --n_sample 500 --sample_sizes 500,1000 --eps 0.5,1,2 --scenarios nonlinear_interaction,sparse_minority_curve,privacy_noise_sparse,simple_linear --methods baseline_rr_debias,mrp_rr_poststrat,hierarchical_rr_mrp_poststrat,neural_rr_mrp --mrp_steps 800 --neural_steps 200 --neural_hidden_layers 32,16 --neural_lr 0.01 --neural_patience 30 --neural_validation_fraction 0.2 --neural_weight_decay 1e-4 --mrp_l2 1.0 --out_dir evidence/final --fail_fast --verbose_every 100`

## Evidence quality checks

| Check | Status | Evidence |
| --- | --- | --- |
| Run is not labelled smoke/non-final | PASS | README wording |
| No recorded failures | PASS | 0 failure rows |
| At least 20 non-skipped trials per summary cell | PASS | minimum n_rows=20; configured trials=20 |
| No skipped result cells | PASS | max n_skipped=0 |
| Neural-vs-linear paired comparisons are available | PASS | 24 paired rows |

## Main results

| scenario | method | best_epsilon | sample_size | mean_overall_l1 | ci95_low | ci95_high | n_rows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| nonlinear_interaction | baseline_rr_debias | 2.0 | 1000 | 0.0751 | 0.0608 | 0.0916 | 20 |
| nonlinear_interaction | hierarchical_rr_mrp_poststrat | 2.0 | 1000 | 0.0855 | 0.0687 | 0.1029 | 20 |
| nonlinear_interaction | mrp_rr_poststrat | 2.0 | 500 | 0.4406 | 0.4335 | 0.4480 | 20 |
| nonlinear_interaction | neural_rr_mrp | 2.0 | 1000 | 0.1053 | 0.0892 | 0.1241 | 20 |
| privacy_noise_sparse | baseline_rr_debias | 2.0 | 1000 | 0.0867 | 0.0713 | 0.0993 | 20 |
| privacy_noise_sparse | hierarchical_rr_mrp_poststrat | 2.0 | 1000 | 0.0812 | 0.0679 | 0.0976 | 20 |
| privacy_noise_sparse | mrp_rr_poststrat | 2.0 | 500 | 0.2637 | 0.2560 | 0.2705 | 20 |
| privacy_noise_sparse | neural_rr_mrp | 2.0 | 1000 | 0.1317 | 0.1029 | 0.1585 | 20 |
| simple_linear | baseline_rr_debias | 2.0 | 1000 | 0.0913 | 0.0749 | 0.1107 | 20 |
| simple_linear | hierarchical_rr_mrp_poststrat | 2.0 | 1000 | 0.1168 | 0.1009 | 0.1345 | 20 |
| simple_linear | mrp_rr_poststrat | 2.0 | 1000 | 0.5090 | 0.5007 | 0.5168 | 20 |
| simple_linear | neural_rr_mrp | 2.0 | 1000 | 0.1318 | 0.1016 | 0.1675 | 20 |
| sparse_minority_curve | baseline_rr_debias | 2.0 | 1000 | 0.1009 | 0.0832 | 0.1177 | 20 |
| sparse_minority_curve | hierarchical_rr_mrp_poststrat | 2.0 | 1000 | 0.1130 | 0.0943 | 0.1289 | 20 |
| sparse_minority_curve | mrp_rr_poststrat | 2.0 | 1000 | 0.4410 | 0.4353 | 0.4462 | 20 |
| sparse_minority_curve | neural_rr_mrp | 2.0 | 1000 | 0.1366 | 0.1168 | 0.1590 | 20 |

## Method win-count summary

### Best mean overall L1 by cell

| method | best-cell count |
| --- | --- |
| baseline_rr_debias | 13 |
| hierarchical_rr_mrp_poststrat | 10 |
| neural_rr_mrp | 1 |

### Best mean major worst-group L1 by cell

| method | best-cell count |
| --- | --- |
| hierarchical_rr_mrp_poststrat | 23 |
| neural_rr_mrp | 1 |

## Claims and support status

| Claim | Evidence file | Status | Required interpretation rule |
| --- | --- | --- | --- |
| RR-aware MRP is evaluated against direct RR debiasing | summary_with_ci.csv / ablations.csv | SUPPORTED TO ANALYSE | Compare mean_overall_l1 and CIs by scenario; report the direction honestly rather than assuming MRP improves aggregate accuracy. |
| Hierarchical partial pooling is evaluated for sparse subgroup error | summary_with_ci.csv | SUPPORTED TO ANALYSE | Use worst_group_l1_major in sparse scenarios and state whether hierarchical partial pooling is actually best in the reported cells. |
| RR-aware Neural MRP is evaluated conditionally against the linear MRP baseline | paired_comparisons.csv | SUPPORTED TO ANALYSE | Use paired neural-minus-linear deltas, win rates and CIs; do not claim general neural superiority. |
| Privacy-noise/sparse robustness is evaluated | summary_with_ci.csv | SUPPORTED TO ANALYSE | Use this for privacy-noise and sparse-cell robustness only; do not claim that privacy improves honesty unless a misreport/privacy-help scenario is present. |

## Limitations

This paper draft is only as strong as the evidence run above. If the run is smoke-labelled, has fewer than 20 repeated trials per cell, has failures, or lacks paired comparisons, the corresponding claims must be reported as preliminary, conditional or unsupported. Randomized response protects answer values only; demographic minimisation, rare-cell reporting and export controls are separate privacy boundaries.

## Reproducibility checklist

- `summary_with_ci.csv` or `summary.csv` present.
- `paired_comparisons.csv` present when neural improvement claims are made.
- `failures.csv` empty for final claims.
- `manifest.json`, `environment.json` and `sha256sums.txt` present.
- Acceptance checks above pass before this text is used as a final report-ready result.

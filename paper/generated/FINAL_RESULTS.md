# Final evidence interpretation: `2026-05-05_182242_mrp_vs_baselines`

This file is generated from committed evidence files. It does not rerun experiments and it does not infer beyond the CSV/JSON contents.

## Provenance

- Run directory: `evidence/final/2026-05-05_182242_mrp_vs_baselines`
- Preset: `custom`
- Command: `not recorded`

## Evidence acceptance checks

| Check | Status | Evidence |
| --- | --- | --- |
| Run is not labelled smoke/non-final | PASS | README wording |
| No recorded failures | PASS | 0 failure rows |
| At least 20 non-skipped trials per summary cell | PASS | minimum n_rows=20; configured trials=20 |
| No skipped result cells | PASS | max n_skipped=0 |
| Neural-vs-linear paired comparisons are available | PASS | 3 paired rows |

## Main result table: best epsilon by scenario and method

| scenario | method | best_epsilon | sample_size | mean_overall_l1 | ci95_low | ci95_high | n_rows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| privacy_helps | baseline_rr_debias | 1.0 | 120 | 0.5981 | 0.5064 | 0.6989 | 20 |
| privacy_helps | hierarchical_rr_mrp_poststrat | 1.0 | 120 | 0.5463 | 0.5288 | 0.5651 | 20 |
| privacy_helps | mrp_rr_poststrat | 1.0 | 120 | 0.5761 | 0.5683 | 0.5836 | 20 |
| privacy_helps | neural_rr_mrp | 1.0 | 120 | 0.5966 | 0.5486 | 0.6437 | 20 |
| privacy_noise_sparse | baseline_rr_debias | 1.0 | 120 | 0.5540 | 0.4816 | 0.6296 | 20 |
| privacy_noise_sparse | hierarchical_rr_mrp_poststrat | 1.0 | 120 | 0.2902 | 0.2724 | 0.3087 | 20 |
| privacy_noise_sparse | mrp_rr_poststrat | 1.0 | 120 | 0.3207 | 0.3138 | 0.3279 | 20 |
| privacy_noise_sparse | neural_rr_mrp | 1.0 | 120 | 0.3751 | 0.3303 | 0.4220 | 20 |
| sparse_minority_curve | baseline_rr_debias | 1.0 | 120 | 0.4751 | 0.4181 | 0.5374 | 20 |
| sparse_minority_curve | hierarchical_rr_mrp_poststrat | 1.0 | 120 | 0.5062 | 0.4873 | 0.5297 | 20 |
| sparse_minority_curve | mrp_rr_poststrat | 1.0 | 120 | 0.5503 | 0.5416 | 0.5584 | 20 |
| sparse_minority_curve | neural_rr_mrp | 1.0 | 120 | 0.5752 | 0.5247 | 0.6295 | 20 |

## Claim-to-evidence status

| Claim | Evidence file | Status | Required interpretation rule |
| --- | --- | --- | --- |
| RR-aware MRP improves over direct RR debiasing | summary_with_ci.csv / ablations.csv | SUPPORTED TO ANALYSE | Compare mean_overall_l1 and CI by scenario; do not claim improvement unless deltas are negative and stable. |
| Hierarchical partial pooling improves sparse subgroup error | summary_with_ci.csv | SUPPORTED TO ANALYSE | Use worst_group_l1_major in sparse scenarios; report failures as mixed results. |
| Neural RR-MRP beats simpler baselines in nonlinear settings | paired_comparisons.csv | SUPPORTED TO ANALYSE | Only claim this if paired deltas have enough trials and confidence intervals exclude zero where relevant. |
| Privacy can help under strategic misreporting | summary_with_ci.csv | SUPPORTED TO ANALYSE | Look for intermediate-epsilon minima; do not overgeneralise to pure privacy noise scenarios. |

## Suggested report wording

- If any acceptance check is `FAIL`, call this run preliminary evidence, not final evidence.
- If paired comparisons are absent, do not make neural-vs-linear superiority claims.
- If a result is mixed across scenarios, state that it is mixed rather than selecting only the favourable rows.
- Every number above comes from `summary_with_ci.csv` or `summary.csv` in this run directory.

## Supplementary robustness evidence

The primary table above is backed by an additional privacy-help robustness curve at `evidence/final/2026-05-05_182638_mrp_vs_baselines/`. That run covers the strategic-misreporting `privacy_helps` scenario over epsilons 0.2, 0.5, 1.0 and 2.0, with 20 non-skipped trials per summary cell and no recorded failures. Use it when making claims about the relationship between privacy level and strategic misreporting correction.

# Final evidence interpretation: `2026-05-05_182638_mrp_vs_baselines`

This file is generated from committed evidence files. It does not rerun experiments and it does not infer beyond the CSV/JSON contents.

## Provenance

- Run directory: `evidence/final/2026-05-05_182638_mrp_vs_baselines`
- Preset: `custom`
- Command: `not recorded`

## Evidence acceptance checks

| Check | Status | Evidence |
| --- | --- | --- |
| Run is not labelled smoke/non-final | PASS | README wording |
| No recorded failures | PASS | 0 failure rows |
| At least 20 non-skipped trials per summary cell | PASS | minimum n_rows=20; configured trials=20 |
| No skipped result cells | PASS | max n_skipped=0 |
| Neural-vs-linear paired comparisons are available | WARN | 0 paired rows |

## Main result table: best epsilon by scenario and method

| scenario | method | best_epsilon | sample_size | mean_overall_l1 | ci95_low | ci95_high | n_rows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| privacy_helps | baseline_rr_debias | 2.0 | 120 | 0.2520 | 0.2101 | 0.2982 | 20 |
| privacy_helps | hierarchical_rr_mrp_poststrat | 2.0 | 120 | 0.4918 | 0.4834 | 0.4995 | 20 |
| privacy_helps | mrp_rr_poststrat | 2.0 | 120 | 0.4597 | 0.4354 | 0.4849 | 20 |

## Claim-to-evidence status

| Claim | Evidence file | Status | Required interpretation rule |
| --- | --- | --- | --- |
| RR-aware MRP improves over direct RR debiasing | summary_with_ci.csv / ablations.csv | SUPPORTED TO ANALYSE | Compare mean_overall_l1 and CI by scenario; do not claim improvement unless deltas are negative and stable. |
| Hierarchical partial pooling improves sparse subgroup error | summary_with_ci.csv | MISSING EVIDENCE | Use worst_group_l1_major in sparse scenarios; report failures as mixed results. |
| Neural RR-MRP beats simpler baselines in nonlinear settings | paired_comparisons.csv | MISSING PAIRED COMPARISONS | Only claim this if paired deltas have enough trials and confidence intervals exclude zero where relevant. |
| Privacy can help under strategic misreporting | summary_with_ci.csv | SUPPORTED TO ANALYSE | Look for intermediate-epsilon minima; do not overgeneralise to pure privacy noise scenarios. |

## Suggested report wording

- If any acceptance check is `FAIL`, call this run preliminary evidence, not final evidence.
- If paired comparisons are absent, do not make neural-vs-linear superiority claims.
- If a result is mixed across scenarios, state that it is mixed rather than selecting only the favourable rows.
- Every number above comes from `summary_with_ci.csv` or `summary.csv` in this run directory.

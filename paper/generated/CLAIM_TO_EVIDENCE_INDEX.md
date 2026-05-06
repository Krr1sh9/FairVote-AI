# Claim-to-evidence index for `2026-05-05_182242_mrp_vs_baselines`

| Claim | Evidence file | Status | Required interpretation rule |
| --- | --- | --- | --- |
| RR-aware MRP improves over direct RR debiasing | summary_with_ci.csv / ablations.csv | SUPPORTED TO ANALYSE | Compare mean_overall_l1 and CI by scenario; do not claim improvement unless deltas are negative and stable. |
| Hierarchical partial pooling improves sparse subgroup error | summary_with_ci.csv | SUPPORTED TO ANALYSE | Use worst_group_l1_major in sparse scenarios; report failures as mixed results. |
| Neural RR-MRP beats simpler baselines in nonlinear settings | paired_comparisons.csv | SUPPORTED TO ANALYSE | Only claim this if paired deltas have enough trials and confidence intervals exclude zero where relevant. |
| Privacy can help under strategic misreporting | summary_with_ci.csv | SUPPORTED TO ANALYSE | Look for intermediate-epsilon minima; do not overgeneralise to pure privacy noise scenarios. |


## Supplementary robustness evidence

- Strategic-misreporting/privacy-level claims should cite `evidence/final/2026-05-05_182638_mrp_vs_baselines/summary_with_ci.csv` and its `BUNDLE/` directory.

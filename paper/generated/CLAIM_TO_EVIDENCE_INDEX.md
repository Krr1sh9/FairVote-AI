# Claim-to-evidence index for `2026-05-06_004647_mrp_vs_baselines`

| Claim | Evidence file | Status | Required interpretation rule |
| --- | --- | --- | --- |
| RR-aware MRP is evaluated against direct RR debiasing | summary_with_ci.csv / ablations.csv | SUPPORTED TO ANALYSE | Compare mean_overall_l1 and CIs by scenario; report the direction honestly rather than assuming MRP improves aggregate accuracy. |
| Hierarchical partial pooling is evaluated for sparse subgroup error | summary_with_ci.csv | SUPPORTED TO ANALYSE | Use worst_group_l1_major in sparse scenarios and state whether hierarchical partial pooling is actually best in the reported cells. |
| RR-aware Neural MRP is evaluated conditionally against the linear MRP baseline | paired_comparisons.csv | SUPPORTED TO ANALYSE | Use paired neural-minus-linear deltas, win rates and CIs; do not claim general neural superiority. |
| Privacy-noise/sparse robustness is evaluated | summary_with_ci.csv | SUPPORTED TO ANALYSE | Use this for privacy-noise and sparse-cell robustness only; do not claim that privacy improves honesty unless a misreport/privacy-help scenario is present. |


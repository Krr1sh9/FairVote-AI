# Final evidence bundle

This directory contains the committed experiment evidence for the final report. The **canonical report-ready run** for the submitted archive is:

```text
2026-05-06_004647_mrp_vs_baselines/
```

Use this run for final report tables, figures and viva claims unless a newer run is explicitly generated and documented.

## Canonical report-ready evidence run

- `2026-05-06_004647_mrp_vs_baselines/`: primary CPU-sized final-style evidence run. It covers `nonlinear_interaction`, `sparse_minority_curve`, `privacy_noise_sparse` and `simple_linear`; epsilons `0.5`, `1.0` and `2.0`; sample sizes `500` and `1000`; and 20 repeated trials per condition. It includes `baseline_rr_debias`, `mrp_rr_poststrat`, `hierarchical_rr_mrp_poststrat` and `neural_rr_mrp`, with 1920 raw rows, 96 summary rows, 24 paired neural-vs-linear comparison rows, 24 ablation rows, runtime profiles, plots, manifests, environment metadata, hashes and zero recorded failures.

This run is deliberately described as **CPU-sized final-style evidence**, not the full `final_evidence` preset. The full preset remains available in `experiments/pipeline/presets.py`, but is computationally heavier. The committed canonical run is the practical examiner-review evidence bundle.

## Main interpretation of the canonical run

The final evidence is mixed and should be reported honestly:

- RR-aware Neural MRP often improves over the **linear** `mrp_rr_poststrat` baseline in paired comparisons: for overall L1, 22 of 24 paired cells have negative neural-minus-linear mean deltas and 18 of 24 have confidence intervals wholly below zero.
- Neural is **not** generally the best method across all available methods. For best mean overall L1 by scenario/epsilon/sample-size condition, `baseline_rr_debias` is best in 13 of 24 cells, `hierarchical_rr_mrp_poststrat` in 10 of 24, and `neural_rr_mrp` in 1 of 24.
- For major worst-group L1, hierarchical partial pooling is strongest: `hierarchical_rr_mrp_poststrat` is best in 23 of 24 cells and `neural_rr_mrp` in 1 of 24.
- Therefore the report-safe conclusion is conditional: neural modelling is technically valid and can beat the linear MRP baseline in many paired settings, but the best deployable choice depends on metric and scenario; direct RR debiasing is often strongest for aggregate L1 and hierarchical partial pooling is strongest for subgroup error.

## Supporting evidence

- `theory/`: analytic and Monte Carlo validation of randomized-response privacy, debiasing, variance, interval behaviour, epsilon/k grids and clipping-bias diagnostics.

Older exploratory or superseded experiment runs are not included in this archive to avoid ambiguity. The canonical run listed above is the only experiment run used for final report tables, generated result summaries and viva claims.

Smoke/sanity runs are quarantined under `evidence/smoke/` and must not be cited as final statistical evidence.

## Generated report artefacts

The generated report-ready artefacts for the canonical run are included in:

- `paper/generated/FINAL_RESULTS.md`
- `paper/generated/CLAIM_TO_EVIDENCE_INDEX.md`
- `paper/generated/fairvote_ai_results.md`

These files are generated from `2026-05-06_004647_mrp_vs_baselines/`. They should be regenerated if a newer run replaces the canonical evidence.

## Reproduction notes

To regenerate the committed report-ready artefacts from the canonical run:

```bash
python -m experiments.write_publication_result \
  --run_dir evidence/final/2026-05-06_004647_mrp_vs_baselines \
  --out_dir paper/generated
```

Do not cite an unsupported claim: every report claim should point to a file in the canonical run directory or to `paper/generated/CLAIM_TO_EVIDENCE_INDEX.md`.

# Final evidence bundle

This directory contains verifiable artefacts generated after the final consistency review. Smoke and superseded intermediate runs were removed from the submission evidence folder, so the final folder does not present minimal or failed evidence as final evidence.

## Primary report-ready evidence run

- `2026-05-05_182242_mrp_vs_baselines/`: primary CPU-sized final evidence run covering `sparse_minority_curve`, `privacy_noise_sparse`, and `privacy_helps` with 20 non-skipped trials per summary cell at epsilon 1.0. It includes `baseline_rr_debias`, `mrp_rr_poststrat`, `hierarchical_rr_mrp_poststrat`, and `neural_rr_mrp`, plus paired neural-vs-linear comparisons, runtime profile, plots, manifest, environment metadata, hashes, and a clean `BUNDLE/` directory.

## Supporting evidence runs

- `2026-05-05_182638_mrp_vs_baselines/`: privacy-help robustness curve over epsilons 0.2, 0.5, 1.0 and 2.0 for the strategic-misreporting `privacy_helps` scenario, with 20 non-skipped trials per summary cell and analytical/linear/hierarchical methods.
- `theory/`: analytic and Monte Carlo validation of randomized-response privacy, debiasing, variance, interval behaviour, epsilon/k grids, and clipping-bias diagnostics.

Smoke/sanity runs are quarantined under `evidence/smoke/` and must not be cited as final statistical evidence.

## Generated report artefacts

The generated report-ready result is in:

- `paper/generated/FINAL_RESULTS.md`
- `paper/generated/CLAIM_TO_EVIDENCE_INDEX.md`
- `paper/generated/fairvote_ai_results.md`

These files are generated from `2026-05-05_182242_mrp_vs_baselines/` and pass the built-in evidence acceptance checks for non-smoke labelling, no failures, at least 20 non-skipped trials per summary cell, no skipped cells, and paired neural-vs-linear comparisons.

## Reproduction notes

Full `final_evidence` presets remain available, but the committed artefacts here are intentionally CPU-sized so a second examiner can rerun them without GPU resources. Do not cite an unsupported claim: every report claim should point to a file in the relevant run directory or to `paper/generated/CLAIM_TO_EVIDENCE_INDEX.md`.

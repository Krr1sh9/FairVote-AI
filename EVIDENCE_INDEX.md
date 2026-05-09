# FairVote-AI evidence index

This file maps assessment claims to the implementation and evidence files that support them. A claim without a code path, test path and output artefact should not be made in the final report.

## Canonical evidence run

The canonical final report evidence run for this submitted archive is:

```text
evidence/final/2026-05-06_004647_mrp_vs_baselines/
```

This run contains 20 trials per condition, sample sizes `500` and `1000`, epsilons `0.5`, `1.0` and `2.0`, scenarios `nonlinear_interaction`, `sparse_minority_curve`, `privacy_noise_sparse` and `simple_linear`, the four deployable methods `baseline_rr_debias`, `mrp_rr_poststrat`, `hierarchical_rr_mrp_poststrat` and `neural_rr_mrp`, and zero recorded failures.

| Claim | Code path | Tests | Evidence output |
|---|---|---|---|
| Browser polling uses client-side k-ary randomized response before submission | `respondent/static/rr.js`, `respondent/static/app.js` | `tests/test_respondent_rr_js.py`, `tests/test_browser_respondent_privacy.py`, `tests/test_respondent_client_privacy.py` | `evidence/privacy/browser_network_capture.md`; static privacy regression tests |
| Production browser client does not expose selected-answer debug audit values | `respondent/static/app.js`, `respondent/index.html` | `tests/test_respondent_client_privacy.py` | Static privacy regression test |
| Server rejects raw-answer fields before storage | `respondent/privacy.py`, `respondent/app_factory.py`, `respondent/storage.py` | `tests/test_respondent_server.py`, `tests/test_respondent_client_privacy.py` | `evidence/privacy/api_rejection_examples.md` |
| Individual response export is token-gated and rare-cell guarded | `respondent/privacy.py`, `respondent/app_factory.py` | `tests/test_respondent_server.py` | `/api/privacy-report`; k-anonymity block for rare cells |
| Upload parser fails safely on malformed JSONL | `app/parsing/upload.py` | `tests/test_dashboard_modules.py` | Invalid-row report from dashboard upload/export logic |
| Dashboard truth columns require explicit synthetic-evaluation mode | `app/services/upload_analysis.py`, `app/controllers/upload_controller.py` | `tests/test_dashboard_modules.py` | Metadata field `synthetic_evaluation_mode` in dashboard export |
| RR debiasing uses the analytic inverse of k-ary RR | `fairvote/privacy/estimators.py`, `fairvote/privacy/mechanisms/kary_rr.py` | `tests/test_debias.py`, `tests/test_rr_properties.py`, `tests/test_theory_validation.py` | `evidence/final/theory/theory_validation.json` |
| MRP models train on the RR observation likelihood, not latent labels | `fairvote/inference/mrp/likelihood.py`, `fairvote/inference/mrp/linear.py`, `fairvote/inference/mrp/hierarchical.py`, `fairvote/inference/mrp/misreport_rr.py` | `tests/test_mrp_canonical.py`, `tests/test_hierarchical_mrp.py` | `evidence/final/2026-05-06_004647_mrp_vs_baselines/summary_with_ci.csv`, `ablations.csv` |
| Hierarchical partial-pooling MRP is implemented and tested | `fairvote/inference/mrp/hierarchical.py`, `experiments/pipeline/methods/hierarchical_mrp.py` | `tests/test_hierarchical_mrp.py`, `tests/test_experiment_pipeline.py` | Rows for `hierarchical_rr_mrp_poststrat` in `evidence/final/2026-05-06_004647_mrp_vs_baselines/summary_with_ci.csv` |
| Sparse-cell/privacy-noise robustness is measured | `experiments/pipeline/scenarios.py`, `experiments/pipeline/presets.py` | `tests/test_experiment_pipeline.py` | `sparse_minority_curve` and `privacy_noise_sparse` rows in `evidence/final/2026-05-06_004647_mrp_vs_baselines/summary_with_ci.csv` |
| RR-aware Neural MRP contribution is evaluated conditionally against baselines | `fairvote/inference/mrp/neural/`, `experiments/pipeline/methods/neural_mrp.py` | `tests/test_rr_neural_mrp.py`, `tests/test_evaluate_neural_mrp.py` | `evidence/final/2026-05-06_004647_mrp_vs_baselines/paired_comparisons.csv`, `paper/generated/FINAL_RESULTS.md` |
| Theory claims are validated by derivation and Monte Carlo | `experiments/theory_validation.py`, `docs/theory_validation.md` | `tests/test_theory_validation.py` | `evidence/final/theory/theory_validation.json`, `evidence/final/theory/theory_validation.md` |
| Evidence bundle is reproducible | `experiments/pipeline/io.py`, `experiments/pipeline/runner.py`, `experiments/build_results_bundle.py` | `tests/test_experiment_pipeline.py` | `manifest.json`, `environment.json`, `sha256sums.txt`, `README.md` in the canonical evidence run |
| Report-ready contribution is stated clearly and cautiously | `experiments/write_publication_result.py`, `paper/generated/` | Manual report review | `paper/generated/FINAL_RESULTS.md`, `paper/generated/CLAIM_TO_EVIDENCE_INDEX.md`, `paper/generated/fairvote_ai_results.md` |

## Required final bundle contents

The canonical run directory should include:

- `raw_trials.csv`
- `summary_with_ci.csv`
- `paired_comparisons.csv`
- `ablations.csv`
- `runtime_profile.csv`
- `failures.csv`
- `config.json`
- `manifest.json`
- `environment.json`
- `sha256sums.txt`
- `README.md`
- `plots/*.png`

Do not cite a result in the report unless it can be traced to one of these files or to the generated files in `paper/generated/`.

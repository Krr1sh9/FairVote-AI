# Local verification summary

Environment: Python 3.14, CPU-only.

## Commands completed successfully

```bash
python -m compileall -q fairvote app respondent experiments tests
```

```bash
python -m pytest tests/test_dashboard_modules.py tests/test_evaluate_neural_mrp.py tests/test_mrp_canonical.py tests/test_hierarchical_mrp.py tests/test_learned_honesty.py -q -p no:cacheprovider
# 27 passed
```

```bash
python -m pytest tests/test_respondent_client_privacy.py tests/test_theory_validation.py tests/test_generate_poll_csv_script.py tests/test_experiment_pipeline.py tests/test_rr_neural_mrp.py -q -p no:cacheprovider
# 45 passed
```

```bash
python -m pytest tests/test_bootstrap_ci.py tests/test_central_dp.py tests/test_cli_entrypoints.py tests/test_debias.py tests/test_group_metrics.py tests/test_mrp_vs_baselines.py tests/test_new_metrics.py tests/test_privacy_core.py tests/test_respondent_rr_js.py tests/test_rr.py tests/test_rr_statistical.py tests/test_streamlit_app_syntax.py tests/test_uncertainty_summaries.py -q -p no:cacheprovider
# 52 passed, 1 skipped (slow test requires FV_RUN_SLOW=1)
```

```bash
python -m pytest tests/test_respondent_server.py tests/test_browser_respondent_privacy.py tests/test_rr_properties.py tests/test_integration_privacy_dashboard_experiment.py -q -p no:cacheprovider
# 4 skipped locally because Flask/Hypothesis/browser optional dependencies are not installed in this container.
# The CI workflow installs .[dev] and .[respondent,browser] to run these gates.
```


Additional post-report-generator check:

```bash
python -m pytest tests/test_experiment_pipeline.py -q -p no:cacheprovider
# 7 passed
```

## Final evidence generated

Primary report-ready run: `evidence/final/2026-05-05_182242_mrp_vs_baselines/`

Acceptance status from `paper/generated/FINAL_RESULTS.md`:

- non-smoke labelling: PASS
- no recorded failures: PASS
- at least 20 non-skipped trials per summary cell: PASS
- no skipped result cells: PASS
- paired neural-vs-linear comparisons available: PASS

Supplementary privacy-help curve: `evidence/final/2026-05-05_182638_mrp_vs_baselines/` has epsilons 0.2, 0.5, 1.0 and 2.0, 20 non-skipped trials per summary cell, no skipped result cells, and no recorded failures.

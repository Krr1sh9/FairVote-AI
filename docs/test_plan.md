# Test Plan

This test plan documents how FairVote-AI is verified beyond simply making `pytest` pass. The test suite is organised around the risks that matter for the project: privacy boundary failures, statistical mistakes in Randomized Response, invalid probability outputs, experiment reproducibility, and evidence-table drift.

## Test categories

| Category | Main files | What they prove | Mark-scheme relevance |
|---|---|---|---|
| Unit tests | `tests/test_rr.py`, `tests/test_debias.py`, `tests/test_privacy_core.py`, `tests/test_mrp_canonical.py`, `tests/test_rr_neural_mrp.py`, `tests/test_group_metrics.py`, `tests/test_new_metrics.py` | Individual modules produce valid deterministic outputs and reject invalid inputs. | Achievement; testing/evaluation |
| Property-based tests | `tests/test_rr_properties.py` | RR matrices, debiased distributions, probability outputs and poststratification weights satisfy invariants over generated inputs. | Testing rigour; methodology correctness |
| Statistical tests | `tests/test_rr_statistical.py` | The RR mechanism empirically matches its theoretical output distribution, `p_keep` is monotonic in epsilon, and estimator variance increases as privacy strengthens. | Evaluation; privacy/utility reasoning |
| Respondent/server integration tests | `tests/test_respondent_server.py`, `tests/test_integration_privacy_dashboard_experiment.py` | Valid perturbed submissions are accepted, malicious raw-answer payloads are rejected, stored JSONL does not contain raw answers, and aggregate results work. | Privacy boundary; ethical awareness; achievement |
| Dashboard tests | `tests/test_dashboard_modules.py`, `tests/test_integration_privacy_dashboard_experiment.py`, `tests/test_streamlit_app_syntax.py` | CSV/JSONL parsing, invalid input handling, method selection, result summaries and export bundles are testable without launching Streamlit. | Engineering quality; reproducibility |
| Experiment-regression tests | `tests/test_experiment_pipeline.py`, `tests/test_mrp_vs_baselines.py`, `tests/test_bootstrap_ci.py`, `tests/test_uncertainty_summaries.py`, `tests/test_evaluate_neural_mrp.py` | Requested methods/scenarios appear, raw-vs-summary aggregation is consistent, config and manifest match, and uncertainty summaries are not silently broken. | Evaluation/testing; reproducible evidence |
| Browser-level privacy test | `tests/test_browser_respondent_privacy.py` | With Playwright, the respondent page can be opened, an answer selected, and the network submission intercepted to verify that raw answers are not sent. | Privacy boundary; viva demonstration evidence |
| CLI/syntax checks | `tests/test_cli_entrypoints.py`, `tests/test_streamlit_app_syntax.py` | Important entry points import/compile and do not fail because of missing optional dependencies. | Reproducibility; installation reliability |

## Standard commands

Full local developer/examiner setup:

```bash
pip install -e ".[dev]"
python -m pytest -q
```

Coverage run:

```bash
python -m pytest -q \
  --cov=fairvote \
  --cov=experiments \
  --cov=respondent \
  --cov=app \
  --cov-report=term-missing:skip-covered \
  --cov-report=html \
  --cov-report=xml \
  --cov-fail-under=60
```

Optional browser privacy test:

```bash
pip install -e ".[browser]"
python -m playwright install chromium
FV_RUN_BROWSER=1 python -m pytest tests/test_browser_respondent_privacy.py -q
```

## Property-based tests

The Hypothesis tests generate many valid combinations of `epsilon`, `k`, counts and poststratification weights. They check that:

- RR transition matrices are finite, non-negative and row-stochastic;
- debiased distributions are valid probability distributions after clipping/renormalisation;
- `privatize_many` never emits categories outside `[0, k-1]`;
- poststratification weights are non-negative and sum to one;
- `p_keep` is monotonic as epsilon increases.

These tests protect the mathematical core from regressions caused by future refactors.

## Statistical RR tests

The statistical tests use fixed seeds and large synthetic samples. They check:

- observed RR output frequencies are close to the theoretical transition row;
- `p_keep` increases monotonically with epsilon;
- debiased estimator variance is larger at lower epsilon, reflecting the privacy/utility trade-off.

These are not final evidence experiments. Their purpose is to prove that the RR implementation behaves like the intended mechanism.

## Integration tests

Integration tests verify cross-layer behaviour:

- respondent form submission into the Flask server;
- JSONL storage contains only perturbed answers and allowed demographics;
- `/api/results` returns aggregate counts only;
- dashboard JSONL parsing can read respondent output;
- a tiny experiment smoke run writes the expected files.

This is important because the project claim depends on multiple layers working together, not isolated functions.

## Experiment-regression tests

The experiment tests guard the evidence pipeline. They check that:

- all requested methods appear in `raw_trials.csv` and summaries;
- all requested scenarios and epsilons appear;
- `summary_with_ci.csv` matches aggregation from `raw_trials.csv` on a tiny fixture;
- `manifest.json` and `config.json` describe the same run;
- required metric columns are not silently dropped;
- paired and ablation files are produced when the relevant methods are present.

These tests make the evidence tables suitable for report use because the table schema is itself tested.

## Coverage summary

The project uses a risk-based repository-wide coverage threshold of `60%`. This is intentionally lower than a core-only target because the repository contains UI entrypoints, plotting/report-generation scripts, compatibility wrappers, and long-running evidence drivers that are not useful to exercise line-by-line in fast CI. Those surfaces are instead checked by compile/import tests, manual/browser checks, smoke experiments, evidence validators, and generated evidence artefacts.

The coverage denominator deliberately keeps the important implementation areas in scope: the canonical Randomized Response channel, debiasing, MRP estimators, respondent privacy/storage/server logic, experiment-pipeline aggregation, and dashboard parsing/service helpers. The coverage report should therefore be read alongside the property, statistical, integration, and evidence-regression tests rather than as a standalone quality claim.

## Known untested or partially tested risks

| Risk | Current mitigation | Why it remains a limitation |
|---|---|---|
| Full Streamlit interaction is not browser-tested. | Pure parsing/inference/export helpers are unit-tested; syntax/import checks cover the entrypoint. | Browser-level Streamlit automation would add runtime and fragility. |
| Full `final_evidence` preset is not run in CI. | Tiny smoke and regression runs verify schema and aggregation; final runs are reproducible from config. | Full runs are computationally expensive. |
| Real deployment security is not tested. | Respondent server has raw-field rejection, token-protected exports, request-size limits and CORS controls. | TLS, reverse-proxy logs, authentication and hosting metadata are outside the prototype. |
| Real election validity is not testable from synthetic simulations. | The experiment protocol states synthetic assumptions and uses oracle baselines. | Synthetic evidence cannot prove external validity. |
| Neural model quality is tested only on small fixtures in CI. | Final evidence runs expose validation NLL, Brier score, entropy and paired comparisons. | Claims about neural superiority must come from repeated evidence, not unit tests. |

## Relation to older testing document

`docs/testing_plan.md` is retained as a short compatibility pointer. This file is the canonical test plan.

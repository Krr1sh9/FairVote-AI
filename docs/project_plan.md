# Project Plan

This plan records the current controlled execution state of the repository. Dates should be adjusted to the student's actual academic calendar and supervision schedule when used in the submitted progress/final report. The technical status below is consistent with the current repository state.

## Project aim

Build and evaluate a reproducible research prototype and benchmark for comparing Randomized-Response-aware polling estimators under Local Differential Privacy, sampling bias, nonlinear demographic effects and misreporting bias.

## Milestones and deliverables

| Milestone | Deliverables | Dependencies | Current status | Remaining work |
|---|---|---|---|---|
| M1. Problem definition and academic framing | `docs/problem_definition_and_background.md`; research question; objectives; scope and non-goals | Supervisor agreement; initial literature search | Implemented in repository documentation | Final report should expand citations and critical literature review in Harvard style. |
| M2. Core privacy mechanism | Canonical Python k-ary RR implementation; browser-side JS RR; tests for matrix/probability/statistical invariants | Defined categorical poll options; Local Differential Privacy design | Implemented | Keep JS/Python agreement tests current if RR code changes. |
| M3. Respondent app privacy boundary | Flask respondent API; recursive raw-field rejection; demographic validation; aggregate endpoint; protected export endpoint; CSP/CORS/request-size controls | Poll configuration schema; RR client mechanism | Implemented | Manual browser test before demo; record deployment limitations in final report. |
| M4. Canonical RR-aware linear poststratification/MRP | Canonical `LinearRRMRPModel`; design matrix helpers; poststratification validation; diagnostics; metadata export | Canonical RR channel; synthetic population features | Implemented | Use final evidence to show where it is sufficient or insufficient. |
| M5. RR-aware Neural MRP | RR-channel training objective; validation NLL; early stopping; deterministic seed/device handling; calibration helpers; diagnostics | PyTorch optional dependency; canonical RR channel; feature design | Implemented | Run final evidence to test whether neural gains are statistically supported. |
| M6. Modular dashboard | Thin `app/streamlit_app.py`; parsing/services/plotting/ui modules; export helpers; dashboard tests | Core estimators; parsing logic; optional Streamlit dependency | Implemented | Optional UI polish only if it directly helps demonstration clarity. |
| M7. Modular experiment pipeline | Presets; scenario generator; method registry; estimator runners; metrics; output writers; manifests; runtime profiles | Synthetic scenario definitions; estimators; metrics | Implemented | Execute final-evidence preset and archive outputs for report. |
| M8. Research scenarios and ablations | Simple linear, nonlinear interaction, nonresponse, misreport/privacy-tradeoff scenarios; oracle and ablation methods | Experiment pipeline; synthetic population generation | Implemented | Interpret results against hypotheses; avoid cherry-picking. |
| M9. Robust tests | Unit, property, statistical, integration, browser, experiment-regression tests; coverage configuration; test plan | Stable code modules; optional browser dependencies | Implemented | Run full test suite in final environment; save coverage report if useful. |
| M10. Project-management documentation | Requirements, traceability matrix, risk register, project plan, design decisions, experiment protocol, privacy boundary | Stable repository architecture | Implemented by current documentation pass | Keep documents aligned if code changes. |
| M11. Final evidence generation | `raw_trials.csv`, `summary_with_ci.csv`, `paired_comparisons.csv`, `ablations.csv`, `runtime_profile.csv`, `config.json`, `manifest.json`, run README | Completed experiment pipeline; sufficient compute time | Evidence pipeline implemented; final heavy run not bundled by default | Run `--preset final_evidence`; verify no failures; use outputs in final report. |
| M12. Final report and viva preparation | Final report, showcase video, viva answers, demonstration script | Final evidence outputs; supervisor feedback | Not part of repository code work yet | Write report; prepare evidence tables; rehearse privacy/AI/evaluation limitations. |

## Current status by workstream

| Workstream | Status | Evidence in repository |
|---|---|---|
| Problem and background | Implemented documentation scaffold | `docs/problem_definition_and_background.md` |
| Requirements and planning | Implemented | `docs/requirements.md`, `docs/requirements_traceability.md`, `docs/risk_register.md`, `docs/project_plan.md` |
| Privacy mechanism | Implemented and tested | `fairvote/privacy/mechanisms/kary_rr.py`, `respondent/static/rr.js`, `tests/test_rr*.py` |
| Respondent server | Implemented and tested | `respondent/server.py`, `tests/test_respondent_server.py` |
| Linear MRP | Implemented canonical path | `fairvote/inference/mrp/linear.py`, `docs/mrp_canonical.md` |
| RR-aware Neural MRP | Implemented with diagnostics | `fairvote/inference/mrp/neural/` (facade: `fairvote/inference/mrp/rr_neural_mrp.py`), `docs/neural_rr_mrp_diagnostics.md` |
| Dashboard | Refactored and tested at pure-function level | `app/`, `tests/test_dashboard_modules.py` |
| Experiment pipeline | Modular and reproducible | `experiments/pipeline/`, `docs/experiment_protocol.md` |
| Final evidence | Pipeline ready; final heavy evidence run remains | `--preset final_evidence` |

## Remaining work before submission

1. Run the final-evidence preset with at least 30 trials, or more if runtime permits.
2. Check `failures.csv`; if failures occur, either fix the cause or document exclusions according to `docs/experiment_protocol.md`.
3. Use `summary_with_ci.csv`, `paired_comparisons.csv`, `ablations.csv` and `runtime_profile.csv` to build final report tables.
4. Expand the literature review in the final report using Harvard references.
5. Prepare a short dashboard/respondent-app demonstration script for the showcase video.
6. Prepare viva answers on privacy limitations, why RR-aware Neural MRP may not always help, and why synthetic results do not prove real-election accuracy.

## Suggested timeline from current repository state

| Period | Focus | Output |
|---|---|---|
| Week 1 | Final evidence dry run and full test run | Confirm install, tests, smoke and medium runs. |
| Week 2 | Final-evidence run | Archive complete final-evidence output folder. |
| Week 3 | Result interpretation | Draft tables/figures and write evaluation discussion. |
| Week 4 | Report integration | Convert docs into final report sections; add citations and limitations. |
| Week 5 | Video and viva preparation | Prepare demo path, viva question bank and evidence references. |

## Dependencies between deliverables

```text
Problem definition + requirements
  -> privacy mechanism + respondent app
  -> canonical estimators
  -> synthetic scenarios + experiment pipeline
  -> final evidence outputs
  -> final report evaluation and viva defence
```

The final report should not claim that RR-aware Neural MRP is superior until the final evidence run supports that claim. If the final evidence is mixed or negative, the research contribution remains valid as a controlled benchmark showing when additional neural complexity is or is not justified.

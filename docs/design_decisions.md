# Design Decisions

This document records the main engineering and research design decisions. Each entry states the context, the decision, alternatives considered, consequences, and evidence/tests.

## DD1: Use browser-side k-ary Randomized Response as the Local Differential Privacy mechanism

**Context.** The project needs a privacy mechanism that perturbs each respondent's answer before the server receives it.

**Decision.** Use k-ary Randomized Response in the browser. The client sends `perturbed_answer`, not the selected raw answer. The canonical Python implementation is kept in `fairvote/privacy/mechanisms/kary_rr.py`, while the browser implementation remains in `respondent/static/rr.js` for client-side privacy.

**Alternatives considered.**

- Store raw answers and apply central differential privacy later. Rejected because it requires trusting the server with raw answers.
- Use a more complex local-DP mechanism. Deferred because RR is simple, auditable, and appropriate for a final-year research prototype.

**Consequences.** RR introduces noise and worsens utility at low epsilon. The project therefore needs RR-aware estimators and a privacy/utility evaluation.

**Evidence/tests.** `tests/test_rr.py`, `tests/test_rr_properties.py`, `tests/test_rr_statistical.py`, `tests/test_respondent_rr_js.py`, `tests/test_browser_respondent_privacy.py`.

## DD2: Keep one canonical Python RR channel

**Context.** Earlier versions risked duplicated RR formulas in estimators, experiments and dashboard code.

**Decision.** All Python code imports RR parameters, transition matrices, privatisation and debiasing from `fairvote/privacy/mechanisms/kary_rr.py`.

**Alternatives considered.** Allow each estimator to implement its own RR matrix. Rejected because drift between formulas would damage assessment confidence and reproducibility.

**Consequences.** Mathematical changes to the RR mechanism are centralised. Browser JS remains separate only because privacy must occur before network submission.

**Evidence/tests.** `tests/test_rr.py`, `tests/test_rr_properties.py`, `tests/test_respondent_rr_js.py`.

## DD3: Harden the respondent server even though it is a prototype

**Context.** A malicious or modified client could try to send raw answers despite the browser-side RR design.

**Decision.** The server recursively rejects forbidden raw-answer keys, validates demographics against `poll_config.json`, limits request size, protects `/api/responses` with a bearer token, keeps `/api/results` aggregate-only, reduces timestamp precision by default, and restricts CORS to configured origins.

**Alternatives considered.** Trust the official browser client. Rejected because final-year viva scrutiny should include adversarial client behaviour.

**Consequences.** The server boundary is defensible as a research prototype, but it is still not production election infrastructure.

**Evidence/tests.** `tests/test_respondent_server.py`, `tests/test_integration_privacy_dashboard_experiment.py`, `docs/privacy_boundary.md`.

## DD4: Use regularised multinomial regression plus poststratification as the canonical RR-aware linear poststratification/MRP-style baseline

**Context.** The project needs a strong, interpretable baseline against the neural model.

**Decision.** Implement one canonical linear RR-aware estimator in `fairvote/inference/mrp/linear.py`. It is honestly documented as regularised multinomial regression through the RR channel plus poststratification, not full Bayesian hierarchical MRP.

**Alternatives considered.** Full Bayesian hierarchical MRP. Deferred because it would require heavier inference machinery and would distract from the RR/neural comparison. Multiple RR-aware linear poststratification/MRP implementations. Rejected because duplicated paths are hard to audit.

**Consequences.** The baseline is practical and reproducible. Its limitations are explicit.

**Evidence/tests.** `tests/test_mrp_canonical.py`, `docs/mrp_canonical.md`, `docs/requirements_traceability.md`.

## DD5: Train RR-aware Neural MRP through the RR observation channel

**Context.** A neural model that trains directly on privatized reports as if they were true labels would be a weak and misleading AI component.

**Decision.** `RRNeuralMRPModel` predicts latent true-vote probabilities and trains on the marginal likelihood of reported privatized labels through the known RR transition matrix.

**Alternatives considered.** Train a classifier directly on reported labels. Kept only as an ablation (`neural_naive_reported_mrp`) because it is methodologically weaker. Require true labels for training. Rejected because real respondent data does not contain true labels.

**Consequences.** The neural component has a principled technical role. It still must be compared against simpler baselines because extra flexibility can overfit.

**Evidence/tests.** `tests/test_rr_neural_mrp.py`, neural diagnostic columns in `raw_trials.csv`, `docs/neural_rr_mrp_diagnostics.md`.

## DD6: Design synthetic scenarios around a specific research question

**Context.** A broad implementation alone does not answer whether RR-aware Neural MRP is useful.

**Decision.** Add scenarios where neural should have a reason to help (`nonlinear_interaction`, `education_urbanicity_interaction`, `sparse_minority_curve`, `privacy_noise_sparse`) and scenarios where linear methods should be sufficient (`simple_linear`, `no_bias`).

**Alternatives considered.** Only generic no-bias/nonresponse scenarios. Rejected because they cannot test the value of nonlinear modelling.

**Consequences.** The project can support a conditional research conclusion: RR-aware Neural MRP helps only under some assumptions, if the evidence says so.

**Evidence/tests.** `experiments/pipeline/scenarios.py`, `tests/test_experiment_pipeline.py`, `docs/experiment_protocol.md`, `docs/research_contribution.md`.

## DD7: Use oracle baselines and ablations

**Context.** Without oracle or ablation methods, it is hard to know whether an estimator is failing because of privacy noise, sampling bias, misreporting, or model class.

**Decision.** Include synthetic-only oracle baselines and ablations such as no-poststratification, naive neural training, canonical RR-aware linear poststratification/MRP, and RR-aware Neural MRP.

**Alternatives considered.** Compare only raw, RR debiasing, linear and neural. Rejected because it gives weaker interpretation of failure modes.

**Consequences.** The output can support publishable-style interpretation, but oracle methods must never be presented as available on real data.

**Evidence/tests.** `experiments/pipeline/methods/`, `tests/test_experiment_pipeline.py`, `paired_comparisons.csv`, `ablations.csv`.

## DD8: Split dashboard and experiment code into testable modules

**Context.** Monolithic files are hard to audit and difficult to test.

**Decision.** Keep `app/streamlit_app.py` as a thin UI router and move parsing, inference orchestration, metrics, plotting and exports into `app/parsing`, `app/services`, and `app/plotting`. Move experiment orchestration into `experiments/pipeline` modules.

**Alternatives considered.** Leave the dashboard and experiment script monolithic. Rejected because it weakens engineering-quality marks and makes failures harder to isolate.

**Consequences.** Core logic can be tested without Streamlit or long experiment runs.

**Evidence/tests.** `tests/test_dashboard_modules.py`, `tests/test_experiment_pipeline.py`, `docs/dashboard_architecture.md`, `docs/experiment_pipeline_architecture.md`.

## DD9: Treat smoke tests and final evidence separately

**Context.** Fast smoke tests are useful during development but can be misleading if presented as final evidence.

**Decision.** Provide `smoke_test`, `medium_evidence`, and `final_evidence` presets. Final evidence uses repeated trials and non-minimal training settings; smoke outputs are explicitly labelled non-final.

**Alternatives considered.** Use one small default run for all purposes. Rejected because one-trial minimal evidence is not rigorous.

**Consequences.** Final evidence is more computationally expensive but more defensible.

**Evidence/tests.** `experiments/pipeline/presets.py`, `tests/test_experiment_pipeline.py`, `docs/experiment_protocol.md`.

## DD10: Keep documentation canonical and link rather than repeat

**Context.** Long repeated explanations across README files can become inconsistent.

**Decision.** Use focused canonical documents: `privacy_boundary.md`, `experiment_protocol.md`, `test_plan.md`, `requirements_traceability.md`, `design_decisions.md`, and `evidence_interpretation.md`. The root README is a gateway, not the full report.

**Alternatives considered.** Keep large repeated evidence and privacy sections in multiple files. Rejected because repetition creates drift and makes assessment scrutiny harder.

**Consequences.** Documentation is easier to maintain and audit.

**Evidence/tests.** Documentation map in `docs/README.md`; this file.

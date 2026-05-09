# Final Consistency Audit

This document records the final repository audit performed before packaging the submission.

## Terminology check

The assessment documents use these terms consistently:

| Canonical term | Repository meaning |
|---|---|
| Local Differential Privacy | The privacy model in which answer randomisation happens before the response leaves the user's browser. |
| Randomized Response | The k-ary answer-value randomisation mechanism used by the respondent client. |
| RR-aware linear poststratification/MRP | The canonical regularised multinomial regression fitted through the RR observation channel and then post-stratified. It is MRP-style, not a full hierarchical Bayesian sampler. |
| RR-aware Neural MRP | The PyTorch estimator that predicts latent true-choice probabilities but trains on privatized reported labels through the known RR channel. |
| Misreport-aware model | A model that composes a behavioural true-to-stated misreport channel with the Randomized Response observation channel. |

## Stale-reference cleanup

Historical constrained evidence folders were removed from `experiments/outputs/` to avoid presenting obsolete one-trial evidence as final submission evidence. The current canonical submitted evidence run is:

```text
evidence/final/2026-05-06_004647_mrp_vs_baselines/
```

The full `final_evidence` preset remains available for heavier reruns, but the submitted evidence should be described as a CPU-sized custom final-style run. The current outputs are documented in `EXAMINER_RUN_GUIDE.md`, `docs/reproducing_experiments.md`, `docs/experiment_protocol.md`, `docs/evidence_interpretation.md`, and `paper/generated/`.

## Canonical source-of-truth documents

| Topic | Source of truth |
|---|---|
| Problem definition/background | `docs/problem_definition_and_background.md` |
| Formal requirements | `docs/requirements.md` |
| Requirements-to-code/test/evidence mapping | `docs/requirements_traceability.md` |
| Privacy boundary | `docs/privacy_boundary.md` |
| Experiment protocol | `docs/experiment_protocol.md` |
| Evidence interpretation | `docs/evidence_interpretation.md` |
| Testing plan | `docs/test_plan.md` |
| Project plan and risk register | `docs/project_plan.md`, `docs/risk_register.md` |
| Design decisions | `docs/design_decisions.md` |
| Canonical MRP implementation | `docs/mrp_canonical.md` |
| Neural diagnostics | `docs/neural_rr_mrp_diagnostics.md` |

Other documents link to these sources rather than duplicating long explanations.

## Claim-support audit

| Major claim | Support |
|---|---|
| Responses are privatized client-side with Randomized Response. | `respondent/static/rr.js`, `respondent/static/app.js`, `fairvote/privacy/mechanisms/kary_rr.py`, `tests/test_respondent_client_privacy.py`, `tests/test_respondent_rr_js.py`, `tests/test_browser_respondent_privacy.py`. |
| Server rejects raw answer fields. | `respondent/server.py`, `tests/test_respondent_server.py`, `docs/privacy_boundary.md`. |
| `/api/results` is aggregate-only and `/api/responses` is protected. | `respondent/server.py`, `tests/test_respondent_server.py`, `docs/api_reference.md`. |
| Python RR implementation is canonical. | `fairvote/privacy/mechanisms/kary_rr.py`, `tests/test_rr.py`, `tests/test_rr_properties.py`, `tests/test_rr_statistical.py`. |
| RR-aware linear poststratification/MRP has one canonical path. | `fairvote/inference/mrp/linear.py`, `design.py`, `poststratify.py`, `tests/test_mrp_canonical.py`, `docs/mrp_canonical.md`. |
| RR-aware Neural MRP trains through reported labels, not true labels. | `fairvote/inference/mrp/neural/` (facade: `fairvote/inference/mrp/rr_neural_mrp.py`), `tests/test_rr_neural_mrp.py`, `docs/ai_component.md`, `docs/neural_rr_mrp_diagnostics.md`. |
| Experiment evidence is reproducible from config. | `experiments/pipeline/`, `tests/test_experiment_pipeline.py`, `docs/experiment_protocol.md`, `docs/reproducing_experiments.md`. |
| Neural superiority is not assumed. | `docs/research_contribution.md`, `docs/evidence_interpretation.md`, final-evidence paired comparison outputs when generated. |
| The system is not production election software. | `README.md`, `EXAMINER_RUN_GUIDE.md`, `docs/privacy_boundary.md`, `docs/problem_definition_and_background.md`. |

## Obsolete or compatibility paths retained

The following are retained deliberately but are not the primary review path:

- `experiments.evaluate_neural_mrp` — legacy neural-only helper retained for older tests and ad-hoc checks. Use `experiments.mrp_vs_baselines --preset final_evidence` for final evidence.
- `fairvote/inference/mrp/model.py` and `fairvote/inference/mrp/rr_mrp_fit.py` — compatibility wrappers around the canonical MRP implementation.
- `experiments/legacy/add_uncertainty_summaries.py` — compatibility/reporting helper for older per-trial output folders; the current final-evidence pipeline writes uncertainty outputs directly.

## Remaining honest limitations

- The repository ships a CPU-sized final-style evidence run rather than the full heavy `final_evidence` preset; report claims must state the reduced grid explicitly.
- Browser-level Playwright testing is optional because it requires installing a browser runtime.
- Coverage threshold is intentionally realistic for a research prototype with optional UI/neural/heavy experiment paths.
- Synthetic data supports controlled benchmarking, not real-election accuracy claims.

# FairVote-AI Documentation

FairVote-AI is a final-year research prototype for privacy-preserving polling under Local Differential Privacy. The documentation is organised so an examiner can inspect the problem, design, privacy boundary, test evidence, experiment protocol, and result interpretation without relying on repeated explanations across multiple files.

## Quick start

### Installation

FairVote-AI supports Python 3.14. From the project root:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Windows PowerShell:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Narrower installs:

```bash
pip install -e .                         # core library only
pip install -e ".[dashboard]"            # Streamlit dashboard
pip install -e ".[respondent]"           # Flask respondent app
pip install -e ".[neural]"               # PyTorch RR-aware Neural MRP
pip install -e ".[experiments]"          # experiment tooling
pip install -e ".[experiments,neural]"   # neural experiments
pip install -e ".[all]"                  # all runtime extras, no dev tools
```

Legacy aliases are preserved: `streamlit` for `dashboard`, `server` for `respondent`, and `ai` for `neural`.

### Common commands

```bash
python -m pytest -q
streamlit run app/streamlit_app.py
python respondent/server.py --port 5001
python -m experiments.mrp_vs_baselines --preset smoke_test
python -m experiments.mrp_vs_baselines --preset final_evidence
```

For the Streamlit dashboard **Upload & Estimate** tab, use a poll CSV from `fixtures/synthetic_with_truth/`. Those fixtures contain `reported_choice` plus evaluation-only `true_choice`. If the dashboard asks for a population CSV, use `app/data/population.csv`.

`smoke_test` is a fast sanity check only. `final_evidence` is the preferred repeated-trial evidence preset.

## Canonical documentation map

| File | Purpose |
|---|---|
| `problem_definition_and_background.md` | Canonical problem definition, academic background, aims, objectives, research questions, scope and references. |
| `requirements.md` | Formal functional, non-functional, privacy, evaluation, usability and reproducibility requirements. |
| `requirements_traceability.md` | Maps requirements to code, tests, evidence outputs, and marking-scheme relevance. |
| `project_plan.md` | Milestones, deliverables, dependencies, current status and remaining work. |
| `risk_register.md` | Five-column risk register with likelihood, severity and mitigation actions. |
| `experiment_protocol.md` | Canonical protocol for research questions, hypotheses, scenarios, baselines, metrics, seeds, success criteria and failure rules. |
| `test_plan.md` | Canonical unit/property/statistical/integration/browser/experiment-regression test plan and coverage gaps. |
| `privacy_boundary.md` | Canonical explanation of client-side privatisation, server storage/rejection rules, endpoint boundaries and limitations. |
| `reproducibility_status.md` | Current Python compatibility, evidence provenance and local verification limitations. |
| `design_decisions.md` | Decision records covering major engineering and research choices. |
| `evidence_interpretation.md` | How to interpret L1 error, subgroup error, calibration, runtime, neural-vs-linear deltas and synthetic-data limitations. |
| `research_contribution.md` | Research framing for when RR-aware Neural MRP should or should not improve over linear baselines. |
| `mrp_canonical.md` | Canonical RR-aware linear poststratification/MRP-style implementation, naming, validation and diagnostics. |
| `neural_rr_mrp_diagnostics.md` | RR-aware Neural MRP objective, validation NLL, early stopping, calibration and metadata. |
| `ai_component.md` | High-level explanation of what the AI component is and is not. |
| `architecture.md` | Whole-system architecture and module map. |
| `experiment_pipeline_architecture.md` | Modular experiment pipeline and method registry. |
| `dashboard_architecture.md` | Refactored dashboard module structure. |
| `api_reference.md` | Respondent server API and environment variables. |
| `reproducing_experiments.md` | Practical commands for regenerating evidence. |
| `ethics_and_privacy.md` | Broader ethics, privacy and sustainability discussion; links to `privacy_boundary.md`. |
| `dashboard_manual_test.md` | Manual dashboard verification checklist. |
| `respondent_manual_test.md` | Manual respondent app verification checklist. |
| `respondent_privacy_testing.md` | Additional manual respondent privacy checks. |
| `consistency_audit.md` | Final terminology, stale-reference and claim-support audit. |

## Canonical explanations to avoid repetition

Use these files as the source of truth:

- Problem definition and academic background: `problem_definition_and_background.md`
- Privacy boundary: `privacy_boundary.md`
- Experiment design: `experiment_protocol.md`
- Test strategy: `test_plan.md`
- Evidence interpretation: `evidence_interpretation.md`
- Formal requirements: `requirements.md`
- Requirements-to-evidence mapping: `requirements_traceability.md`
- Project plan: `project_plan.md`
- Risk register: `risk_register.md`
- Major design decisions: `design_decisions.md`
- Final consistency audit: `consistency_audit.md`

Other documents should link to these rather than re-stating the same material.

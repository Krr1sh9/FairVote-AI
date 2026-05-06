# FairVote-AI

FairVote-AI is a final-year research prototype for **privacy-preserving polling under Local Differential Privacy**. It combines browser-side k-ary Randomized Response, RR-aware statistical estimation, RR-aware Neural MRP, controlled synthetic experiments, and a Streamlit analyst dashboard.

The project is not production election software. Its purpose is to study privacy/utility/subgroup-error trade-offs and to answer a focused research question:

> When does RR-aware Neural MRP improve over simpler RR-aware linear/poststratification baselines under Local Differential Privacy and sampling bias?

For the project problem statement, academic background, objectives and research questions, see [`docs/problem_definition_and_background.md`](docs/problem_definition_and_background.md). For the assessment-focused run guide, see [`EXAMINER_RUN_GUIDE.md`](EXAMINER_RUN_GUIDE.md). For the full documentation map, see [`docs/README.md`](docs/README.md).

## System overview

```text
Respondent browser
  -> k-ary Randomized Response applied client-side
  -> Flask server stores perturbed answers + validated demographics
  -> dashboard / experiment pipeline estimates aggregate and subgroup outcomes
  -> RR debiasing, RR-aware linear poststratification/MRP, RR-aware Neural MRP and ablations are compared
```

The privacy mechanism is **Randomized Response**, not AI. The AI component is the **RR-aware Neural MRP** model in `fairvote/inference/mrp/neural/` (facade: `fairvote/inference/mrp/rr_neural_mrp.py`). It predicts latent true-vote probabilities from features and trains on privatized reported labels through the known RR observation channel. It does not require true labels in real-data training mode.

## Installation

FairVote-AI supports **Python 3.14**. Use a clean Python 3.14 virtual environment from the project root.

macOS/Linux:

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

`pip install -e ".[dev]"` installs the full local verification environment: tests, coverage, Ruff, mypy, dashboard dependencies, respondent-server dependencies, experiment dependencies and PyTorch.

Narrower installs are available:

```bash
pip install -e .                         # core library only
pip install -e ".[dashboard]"            # Streamlit dashboard
pip install -e ".[respondent]"           # Flask respondent app
pip install -e ".[neural]"               # PyTorch RR-aware Neural MRP
pip install -e ".[experiments]"          # experiment tooling
pip install -e ".[experiments,neural]"   # neural experiment scripts
pip install -e ".[all]"                  # all runtime extras, no dev tools
```

Optional browser privacy tests require:

```bash
pip install -e ".[browser]"
python -m playwright install chromium
```

## Quick commands

Run tests:

```bash
python -m pytest -q
```

Run coverage:

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

The coverage gate is risk-based: core privacy/MRP/respondent/experiment-pipeline/dashboard-service code is measured, while UI entrypoints, plotting/report scripts and compatibility wrappers are checked through smoke, compile, browser/manual and evidence-validation paths.

Run the respondent app:

```bash
python respondent/server.py --port 5001
```

Run the dashboard:

```bash
streamlit run app/streamlit_app.py
```

For the **Upload & Estimate** tab, use one of the truth-labelled synthetic poll CSV files from:

```text
fixtures/synthetic_with_truth/
```

These files are intended for dashboard evaluation/demo use because they contain both `reported_choice` and evaluation-only `true_choice`. If the dashboard asks for a population CSV for post-stratification, use `app/data/population.csv`. Do not use `true_choice` as a real respondent field; real respondent exports should contain only `perturbed_answer` plus validated demographics.

Run a fast experiment smoke test:

```bash
python -m experiments.mrp_vs_baselines --preset smoke_test
```

Run the preferred final-evidence preset:

```bash
python -m experiments.mrp_vs_baselines --preset final_evidence
```

Increase trial count if runtime allows:

```bash
python -m experiments.mrp_vs_baselines --preset final_evidence --trials 50
```

## Repository structure

```text
fairvote/
  privacy/                 # canonical RR channel and privacy estimators
  inference/mrp/           # canonical RR-aware linear poststratification/MRP and RR-aware Neural MRP
  metrics/                 # aggregate and subgroup metrics

respondent/
  server.py                # Flask respondent API
  static/rr.js             # browser-side Randomized Response
  static/app.js            # respondent-page client logic

app/
  streamlit_app.py         # thin dashboard entrypoint
  parsing/                 # CSV/JSONL parsing
  services/                # inference, metrics, exports, scenario helpers
  plotting/                # chart helpers
  ui/                      # Streamlit page modules

experiments/
  mrp_vs_baselines.py      # CLI wrapper
  pipeline/                # modular experiment pipeline, presets, method registry

tests/                     # unit, property, statistical, integration, browser and regression tests

docs/                      # assessment technical documentation
```

## Canonical technical documents

| Topic | Document |
|---|---|
| Problem definition and academic background | [`docs/problem_definition_and_background.md`](docs/problem_definition_and_background.md) |
| Formal requirements | [`docs/requirements.md`](docs/requirements.md) |
| Requirements-to-evidence mapping | [`docs/requirements_traceability.md`](docs/requirements_traceability.md) |
| Project plan | [`docs/project_plan.md`](docs/project_plan.md) |
| Risk register | [`docs/risk_register.md`](docs/risk_register.md) |
| Experiment protocol | [`docs/experiment_protocol.md`](docs/experiment_protocol.md) |
| Test plan and coverage gaps | [`docs/test_plan.md`](docs/test_plan.md) |
| Privacy boundary | [`docs/privacy_boundary.md`](docs/privacy_boundary.md) |
| Design decisions | [`docs/design_decisions.md`](docs/design_decisions.md) |
| Evidence interpretation | [`docs/evidence_interpretation.md`](docs/evidence_interpretation.md) |
| Research contribution | [`docs/research_contribution.md`](docs/research_contribution.md) |
| Canonical RR-aware linear poststratification/MRP | [`docs/mrp_canonical.md`](docs/mrp_canonical.md) |
| RR-aware Neural MRP diagnostics | [`docs/neural_rr_mrp_diagnostics.md`](docs/neural_rr_mrp_diagnostics.md) |
| Final consistency audit | [`docs/consistency_audit.md`](docs/consistency_audit.md) |

The root README is intentionally a gateway. Detailed privacy, testing, experiment and evidence explanations are kept in the canonical docs above to reduce repetition and avoid inconsistent claims.

## Evidence outputs

Final-evidence runs write:

```text
raw_trials.csv
summary_with_ci.csv
paired_comparisons.csv
ablations.csv
runtime_profile.csv
failures.csv
config.json
manifest.json
README.md
plots/
```

Use `summary_with_ci.csv` for report tables, `paired_comparisons.csv` for neural-minus-linear claims, `ablations.csv` for method-choice interpretation, and `runtime_profile.csv` for practical runtime discussion. See [`docs/evidence_interpretation.md`](docs/evidence_interpretation.md).

## Privacy claim

The defensible privacy claim is narrow:

- the answer value is privatized in the browser using k-ary Randomized Response;
- the server is designed to reject raw-answer fields, including nested malicious fields;
- demographics are validated but not locally privatized;
- `/api/results` is aggregate-only;
- individual-level `/api/responses` requires an analyst bearer token;
- this does not solve identity, metadata, deployment, authentication or production election-security problems.

See [`docs/privacy_boundary.md`](docs/privacy_boundary.md) for the canonical statement.

## Testing claim

The test suite includes unit tests, property-based tests, statistical RR tests, respondent integration tests, dashboard pure-function tests, experiment-regression tests and an optional Playwright browser privacy test. The repository-wide coverage threshold is intentionally realistic because heavy UI/neural/final-evidence paths are not all run in fast CI. See [`docs/test_plan.md`](docs/test_plan.md).

## Important limitations

- Synthetic experiments do not prove real-election accuracy.
- RR-aware Neural MRP is not assumed to be better than RR-aware linear poststratification/MRP; it is evaluated conditionally.
- Local Differential Privacy protects answer values, not demographics, metadata, IP logs or identity.
- Fairness metrics audit subgroup error but do not guarantee fairness.
- Smoke-test outputs must not be presented as final evidence.

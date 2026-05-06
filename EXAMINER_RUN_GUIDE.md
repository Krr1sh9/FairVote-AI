# FairVote-AI Examiner Run Guide

This guide gives the clean review path for checking the submission. It avoids historical development commands and points to the canonical documentation sources.

## 1. Read these first

1. [`docs/problem_definition_and_background.md`](docs/problem_definition_and_background.md) — title, aim, objectives, research questions, academic background, scope and non-goals.
2. [`docs/requirements_traceability.md`](docs/requirements_traceability.md) — maps requirements to code, tests, evidence outputs and marking-scheme relevance.
3. [`docs/privacy_boundary.md`](docs/privacy_boundary.md) — exact client/server privacy boundary and limitations.
4. [`docs/experiment_protocol.md`](docs/experiment_protocol.md) — final-evidence protocol, scenarios, baselines, metrics and success criteria.
5. [`docs/evidence_interpretation.md`](docs/evidence_interpretation.md) — how to interpret generated results without overclaiming.

FairVote-AI is a final-year research prototype, not production election software. The defensible contribution is:

> A reproducible research prototype and benchmark for comparing RR-aware polling estimators under Local Differential Privacy and sampling/misreporting bias.

## 2. Terminology used in the repository

| Term | Meaning in this project |
|---|---|
| Local Differential Privacy | Privacy model where each respondent's answer is randomized before it leaves their browser. |
| Randomized Response | The k-ary mechanism used to privatize answer values client-side. |
| RR-aware linear poststratification/MRP | Canonical regularised multinomial regression fitted through the RR observation channel, then post-stratified. It is MRP-style, not a full hierarchical Bayesian sampler. |
| RR-aware Neural MRP | PyTorch model that predicts latent true-vote probabilities but trains on privatized reported labels through the known RR channel. |
| Misreport-aware model | Baseline/extension that adds a behavioural true-to-stated misreport channel before Randomized Response. |

## 3. Install

Use **Python 3.14** from the repository root.

### Windows PowerShell

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

### macOS/Linux

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

`.[dev]` installs the standard examiner environment: tests, coverage, Ruff, mypy, dashboard dependencies, respondent-server dependencies, experiment tooling and PyTorch. Optional browser tests additionally require:

```bash
pip install -e ".[browser]"
python -m playwright install chromium
```

## 4. Run tests

The standard test-suite command is the same on Windows, macOS and Linux:

```bash
python -m pytest -q
```

On Windows PowerShell this command should be entered exactly as one line:

```powershell
python -m pytest -q
```

Optional browser and slow tests are intentionally skipped unless their environment variables are set. A result such as `174 passed, 2 skipped` is therefore a successful standard-suite run, not a failure.

### Coverage command — Windows PowerShell

Use PowerShell backticks for line continuation. The backtick must be the final character on the line, with no trailing spaces.

```powershell
python -m pytest -q `
  --cov=fairvote `
  --cov=experiments `
  --cov=respondent `
  --cov=app `
  --cov-report="term-missing:skip-covered" `
  --cov-report=html `
  --cov-report=xml `
  --cov-fail-under=60
```

If line continuation is awkward, use this single-line PowerShell version instead:

```powershell
python -m pytest -q --cov=fairvote --cov=experiments --cov=respondent --cov=app --cov-report="term-missing:skip-covered" --cov-report=html --cov-report=xml --cov-fail-under=60
```

### Coverage command — macOS/Linux shell

Use backslashes only in Bash/zsh/sh shells:

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

### Optional browser privacy test

Windows PowerShell:

```powershell
pip install -e ".[browser]"
python -m playwright install chromium
$env:FV_RUN_BROWSER="1"
python -m pytest tests/test_browser_respondent_privacy.py -q
Remove-Item Env:FV_RUN_BROWSER -ErrorAction SilentlyContinue
```

macOS/Linux:

```bash
pip install -e ".[browser]"
python -m playwright install chromium
FV_RUN_BROWSER=1 python -m pytest tests/test_browser_respondent_privacy.py -q
```

### Optional slow experiment test

Windows PowerShell:

```powershell
$env:FV_RUN_SLOW="1"
python -m pytest tests/test_mrp_vs_baselines.py -q
Remove-Item Env:FV_RUN_SLOW -ErrorAction SilentlyContinue
```

macOS/Linux:

```bash
FV_RUN_SLOW=1 python -m pytest tests/test_mrp_vs_baselines.py -q
```

See [`docs/test_plan.md`](docs/test_plan.md) for test layers, coverage gaps and known untested risks.

## 5. Run the respondent app

```bash
python respondent/server.py --port 5001
```

Open `http://127.0.0.1:5001` and submit a response. The server writes to:

```text
respondent/data/responses.jsonl
```

A valid record contains `perturbed_answer` plus validated demographics. It must not contain raw-answer fields such as `true_answer`, `true_choice`, `selected_answer`, `selectedOption`, `raw_vote` or `raw_answer`.

Useful endpoint check:

```bash
curl http://127.0.0.1:5001/api/results
```

`/api/results` is aggregate-only. `/api/responses` is protected and requires an analyst token:

```bash
FAIRVOTE_ANALYST_TOKEN=dev-token python respondent/server.py --port 5001
curl -H "Authorization: Bearer dev-token" http://127.0.0.1:5001/api/responses
```

Do not expose the Flask development server publicly.

## 6. Run the dashboard

```bash
streamlit run app/streamlit_app.py
```

In the dashboard **Upload & Estimate** tab, use one of the truth-labelled synthetic poll CSV files in:

```text
fixtures/synthetic_with_truth/
```

These fixture CSVs are the intended review/demo inputs for the upload path because they contain `reported_choice` and evaluation-only `true_choice`. If the dashboard asks for a population CSV for post-stratification, use `app/data/population.csv`. Real respondent JSONL should contain only `perturbed_answer` plus validated demographics and should not contain `true_choice`.

Dashboard implementation entry points:

- `app/streamlit_app.py` — thin Streamlit page router.
- `app/parsing/` — CSV/JSONL parsing.
- `app/services/` — inference, metrics, exports and scenario helpers.
- `app/plotting/` — plotting helpers.
- `app/ui/` — page modules.

## 7. Run a smoke experiment

Fast sanity check only:

```bash
python -m experiments.mrp_vs_baselines --preset smoke_test
```

This intentionally uses minimal training settings and one trial. Do not use it for final quantitative claims.

## 8. Reproduce final evidence

Preferred final-evidence command:

```bash
python -m experiments.mrp_vs_baselines --preset final_evidence
```

For stronger evidence if runtime allows:

```bash
python -m experiments.mrp_vs_baselines --preset final_evidence --trials 50
python -m experiments.mrp_vs_baselines --preset final_evidence --trials 100
```

The final-evidence preset uses non-minimal training settings, 30 trials by default, epsilons `0.2,0.5,1.0,2.0`, sample sizes `500,1000,2500`, and scenarios `simple_linear,nonresponse,nonlinear_interaction,shy_fixed`.

Each run writes a timestamped folder under `experiments/outputs/` containing:

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

For backwards compatibility it may also write `results_trials.csv` and `summary.csv`. Use the newer final-evidence files for report tables and viva answers.

## 9. Where to inspect code

| Concern | Primary files |
|---|---|
| Canonical Randomized Response channel | `fairvote/privacy/mechanisms/kary_rr.py`; browser reference `respondent/static/rr.js` |
| Respondent privacy boundary | `respondent/server.py`; `respondent/static/app.js`; `respondent/index.html` |
| RR-aware linear poststratification/MRP | `fairvote/inference/mrp/linear.py`; `design.py`; `poststratify.py` |
| RR-aware Neural MRP | `fairvote/inference/mrp/neural/` (facade: `fairvote/inference/mrp/rr_neural_mrp.py`); `docs/neural_rr_mrp_diagnostics.md` |
| Misreport-aware model | `fairvote/inference/mrp/misreport_rr.py`; `learned_misreport_rr.py` |
| Experiment pipeline | `experiments/pipeline/`; CLI wrapper `experiments/mrp_vs_baselines.py` |
| Dashboard | `app/streamlit_app.py`; `app/parsing/`; `app/services/`; `app/ui/` |
| Tests | `tests/`; see `docs/test_plan.md` |

## 10. Limitations to keep in mind

- Local Differential Privacy protects answer values, not identity, IP address, traffic metadata or demographic uniqueness.
- Demographics are validated but not randomized.
- The project audits subgroup error; it does not guarantee fairness.
- RR-aware Neural MRP is not assumed to beat the linear baseline. The final-evidence question is conditional: when, if ever, does it improve enough to justify the extra complexity?
- Synthetic experiments support controlled benchmarking, not real-election forecasting claims.
- Smoke-test outputs are not final evidence.

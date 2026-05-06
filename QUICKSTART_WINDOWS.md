# FairVote-AI Windows Quickstart

This guide is for Windows users running the current submission from PowerShell.

FairVote-AI is a final-year research prototype for privacy-preserving polling under Local Differential Privacy. It is not production election software and does not provide secure voter authentication, full anonymity or guaranteed fairness.

## 1. Requirements

Install:

1. Python 3.14.
2. Node.js LTS if you want to run the JavaScript Randomized Response tests.

Check versions:

```powershell
python --version
node --version
```

If `python` is not available, try:

```powershell
py --version
```

## 2. Open PowerShell in the project folder

In File Explorer, open the extracted `FairVote-AI` folder, click the address bar, type `powershell`, and press Enter.

You should see files such as `README.md`, `pyproject.toml`, `fairvote`, `respondent`, `app`, `experiments` and `tests`.

## 3. Create and activate a virtual environment

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If script activation is blocked:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

If `py -3.14` is unavailable but `python --version` shows Python 3.14.x:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 4. Install

Full examiner/development install:

```powershell
pip install -e ".[dev]"
```

Narrower installs:

```powershell
pip install -e .
pip install -e ".[dashboard]"
pip install -e ".[respondent]"
pip install -e ".[experiments,neural]"
```

## 5. Run tests

Standard suite:

```powershell
python -m pytest -q
```

Optional browser and slow tests are intentionally skipped unless their environment variables are set. A result such as `168 passed, 2 skipped` means the standard suite passed and the opt-in tests were skipped.

Coverage command for Windows PowerShell (same 60% gate as CI):

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

If PowerShell line continuation causes trouble, use the same command on one line:

```powershell
python -m pytest -q --cov=fairvote --cov=experiments --cov=respondent --cov=app --cov-report="term-missing:skip-covered" --cov-report=html --cov-report=xml --cov-fail-under=60
```

Targeted privacy/server checks:

```powershell
python -m pytest tests/test_respondent_server.py -q
python -m pytest tests/test_respondent_client_privacy.py -q
python -m pytest tests/test_respondent_rr_js.py -q
```

Targeted neural checks:

```powershell
python -m pytest tests/test_rr_neural_mrp.py -q
```

Experiment-regression checks:

```powershell
python -m pytest tests/test_experiment_pipeline.py -q
```

Optional Playwright browser test:

```powershell
pip install -e ".[browser]"
python -m playwright install chromium
$env:FV_RUN_BROWSER="1"
python -m pytest tests/test_browser_respondent_privacy.py -q
Remove-Item Env:FV_RUN_BROWSER -ErrorAction SilentlyContinue
```

Optional slow experiment test:

```powershell
$env:FV_RUN_SLOW="1"
python -m pytest tests/test_mrp_vs_baselines.py -q
Remove-Item Env:FV_RUN_SLOW -ErrorAction SilentlyContinue
```

## 6. Run the respondent app

```powershell
python respondent/server.py --port 5001
```

Open:

```text
http://127.0.0.1:5001
```

Submit a response. The server stores privatized responses in:

```text
respondent/data/responses.jsonl
```

A real respondent record should contain `perturbed_answer` and validated demographics. It should not contain `true_answer`, `true_choice`, `selected_answer`, `selectedOption`, `raw_vote` or `raw_answer`.

To check the aggregate endpoint:

```powershell
curl.exe http://127.0.0.1:5001/api/results
```

`/api/responses` requires an analyst bearer token. See `docs/privacy_boundary.md`.

## 7. Run the dashboard

```powershell
streamlit run app/streamlit_app.py
```

In the dashboard **Upload & Estimate** tab, use one of the synthetic poll CSV files from `fixtures/synthetic_with_truth/`, or upload `respondent/data/responses.jsonl` after using the respondent app.

For the fixture CSV files in `fixtures/synthetic_with_truth/`:

- `reported_choice` is the reported privatized label;
- `true_choice` is evaluation-only and should be selected only when the dashboard asks for a ground-truth/evaluation column;
- `region` and `age_band` are demographic feature columns;
- `epsilon` and `k` describe the Randomized Response setting used to generate the file;
- if the dashboard asks for a population CSV for post-stratification, use `app/data/population.csv`;
- real respondent exports should not contain true labels.

## 8. Run experiments

Smoke test only:

```powershell
python -m experiments.mrp_vs_baselines --preset smoke_test
```

Final-evidence preset:

```powershell
python -m experiments.mrp_vs_baselines --preset final_evidence
```

Increase repeated trials if runtime allows:

```powershell
python -m experiments.mrp_vs_baselines --preset final_evidence --trials 50
```

Outputs are written to `experiments/outputs/<timestamp>/` and include `raw_trials.csv`, `summary_with_ci.csv`, `paired_comparisons.csv`, `ablations.csv`, `runtime_profile.csv`, `config.json`, `manifest.json` and a run README.

## 9. Common troubleshooting

### PowerShell says scripts are disabled

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### JavaScript RR tests fail because Node is missing

Install Node.js LTS, then reopen PowerShell:

```powershell
winget install OpenJS.NodeJS.LTS
node --version
```

### PyTorch install fails

Install PyTorch using the official instructions for your machine, then rerun:

```powershell
pip install -e ".[dev]"
```

### The respondent page blocks repeated manual submissions

The app uses localStorage as casual duplicate prevention. Clear localStorage keys beginning with:

```text
fairvote.submitted.
```

This is not secure authentication.

### Dashboard cannot find columns

Check whether you uploaded a synthetic CSV or respondent JSONL. For real respondent JSONL, use `perturbed_answer` as the reported-answer column.

## 10. What to read next

- `EXAMINER_RUN_GUIDE.md`
- `docs/problem_definition_and_background.md`
- `docs/privacy_boundary.md`
- `docs/experiment_protocol.md`
- `docs/evidence_interpretation.md`
- `docs/test_plan.md`

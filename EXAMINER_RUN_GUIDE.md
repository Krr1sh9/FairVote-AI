# FairVote-AI Examiner Run Guide

This is a short practical guide for running and checking FairVote-AI from a fresh project folder.

## What this project does

FairVote-AI is an **AI-assisted privacy-preserving polling framework**. The respondent web app applies browser-side k-ary Randomized Response before submission, so the server stores privatized reported answers rather than true votes. The analysis layer compares RR debiasing, linear RR-aware MRP, misreport-aware MRP, and RR-aware Neural MRP for aggregate and subgroup inference from privatized reports. Randomized Response is the privacy mechanism; RR-aware Neural MRP is the AI component.

FairVote-AI is a final-year research prototype. It is **not** production election software and does **not** provide secure voter authentication.

---

## 1. Install on Windows PowerShell

Open PowerShell in the extracted `FairVote-AI` folder. You can do this by opening the folder in File Explorer, clicking the address bar, typing `powershell`, and pressing Enter.

Use Python 3.14.4 for final local verification. Create and activate a virtual environment:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
```

If activation is blocked, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Install the full project extras:

```powershell
python -m pip install --upgrade pip
pip install -e ".[dev,ai,streamlit,respondent]"
```

For JavaScript Randomized Response tests, install Node.js if needed:

```powershell
winget install OpenJS.NodeJS.LTS
```

Close and reopen PowerShell after installing Node.js, then check:

```powershell
node --version
```

---

## 2. Install on macOS / Linux

Open a terminal in the extracted `FairVote-AI` folder.

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python --version
python -m pip install --upgrade pip
pip install -e ".[dev,ai,streamlit,respondent]"
```

The intended final verification version is Python 3.14.4. In GitHub Actions, the workflow targets Python 3.14; local final verification should use Python 3.14.4 exactly if available.

---

## 3. Run the respondent web app

```bash
python respondent/server.py --port 5001
```

Open:

```text
http://127.0.0.1:5001
```

Useful checks:

1. Submit one response.
2. Refresh the page and confirm the browser shows an already-submitted message.
3. Check `respondent/data/responses.jsonl` and confirm it stores `perturbed_answer`, not true/raw answer fields.
4. Remember: the localStorage duplicate guard is casual duplicate prevention only, not secure one-person-one-vote.

To allow access from another device on the same Wi-Fi network:

```bash
python respondent/server.py --host 0.0.0.0 --port 5001
```

Do not expose the Flask development server publicly.

---

## 4. Run the dashboard

```bash
streamlit run app/streamlit_app.py
```

Use the synthetic demo files in:

```text
app/data/
```

Recommended demo files:

```text
app/data/population.csv
app/data/poll_no_bias_eps1_n5000_20260126T222552.csv
app/data/poll_nonresponse_eps1_n5000_20260126T222832.csv
app/data/poll_shy_privacy_helps_eps0.5_n5000_20260126T222748.csv
```

In the dashboard:

- use `reported_choice` for synthetic CSV reported-answer data;
- use `true_choice` only for synthetic evaluation, not real polling mode;
- use `perturbed_answer` for respondent JSONL uploads;
- compare RR debiasing, linear RR-aware MRP, neural RR-aware MRP, and misreport-aware MRP where available.

---

## 5. Run tests

Full standard test suite:

```bash
pytest -q
```

Targeted neural model tests:

```bash
pytest tests/test_rr_neural_mrp.py -q
```

Targeted respondent tests:

```bash
pytest tests/test_respondent_server.py -q
pytest tests/test_respondent_client_privacy.py -q
pytest tests/test_respondent_rr_js.py -q
```

The JavaScript RR tests require Node.js. On Windows:

```powershell
winget install OpenJS.NodeJS.LTS
```

Dashboard syntax/import-style check:

```bash
pytest tests/test_streamlit_app_syntax.py -q
```

---

## 6. Run slow tests

The slow MRP integration test is intentionally local-only and is not required for quick CI runs.

PowerShell:

```powershell
$env:FV_RUN_SLOW="1"
pytest tests/test_mrp_vs_baselines.py -q
Remove-Item Env:FV_RUN_SLOW
```

macOS / Linux:

```bash
FV_RUN_SLOW=1 pytest tests/test_mrp_vs_baselines.py -q
```

---

## 7. Run a small experiment

```bash
python -m experiments.evaluate_neural_mrp --preset small
```

This writes a timestamped output folder under:

```text
experiments/outputs/
```

Expected files include:

```text
config.json
results_trials.csv
summary.csv
neural_comparison.csv
method_rankings.csv
neural_verdict.csv
```

---

## 8. Final evidence outputs

The included final evidence pack is here:

```text
experiments/outputs/final_neural_evidence/
```

It contains:

```text
README.md
config.json
results_trials.csv
results_trials.jsonl
summary.csv
summary.jsonl
neural_comparison.csv
neural_comparison.jsonl
method_rankings.csv
neural_verdict.csv
neural_verdict.json
plots/
```

Important: this evidence pack is **computationally constrained final-style evidence**, not the exhaustive full preset. It uses one trial per condition and reduced MRP/neural training steps. It should be used as constrained evidence, not as proof that neural RR-MRP is generally superior.

---

## 9. Limitations to know before judging

- Randomized Response is the privacy mechanism, not AI.
- RR-aware Neural MRP is the AI component.
- The server rejects `true_answer` and related true/raw answer fields.
- Real respondent data should not contain `true_choice`; that column is synthetic/evaluation-only.
- Local Differential Privacy protects submitted answer values, not full anonymity.
- Demographics, metadata, small groups, and deployment logs still create residual privacy risks.
- localStorage duplicate prevention is casual only; it is not secure authentication.
- FairVote-AI audits subgroup error; it does not guarantee fairness.
- The project is not production election software.
- The included final evidence shows mixed results: neural RR-MRP is privacy-compatible and valid, but it does not consistently outperform linear RR-aware MRP in the constrained evidence pack.

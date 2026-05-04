# FairVote-AI Windows Quickstart

This guide is for Windows users who have just downloaded and extracted the FairVote-AI project and are using **PowerShell**.

FairVote-AI is a university research prototype for privacy-preserving polling. It is **not production election software**, does **not** provide secure one-person-one-vote authentication, and does **not** guarantee anonymity. The respondent app uses Local Differential Privacy to protect the submitted answer value, and the analysis tools estimate aggregate/subgroup results from privatized reports.

---

## 0. What you need installed

Install these before starting:

1. **Python 3.14.4**
   - Download Python 3.14.4 from <https://www.python.org/downloads/windows/>.
   - During installation, tick **Add Python to PATH** if offered.
   - This project is targeted at Python 3.14.4 for final submission; other Python versions should be treated as untested here.
2. **Node.js LTS** for JavaScript Randomized Response tests
   - Download from <https://nodejs.org/>
   - Or install with PowerShell using `winget`:

```powershell
winget install OpenJS.NodeJS.LTS
```

After installing, close and reopen PowerShell, then check:

```powershell
python --version
node --version
npm --version
```

If `python --version` does not work, try:

```powershell
py --version
```

---

## 1. Open PowerShell in the FairVote-AI project folder

1. Find the folder you extracted, for example a folder named `FairVote-AI`.
2. Open the folder in File Explorer.
3. Click the address bar at the top.
4. Type:

```text
powershell
```

5. Press **Enter**.

PowerShell should open directly in the project folder.

Check you are in the correct folder:

```powershell
Get-ChildItem
```

You should see files/folders such as:

```text
README.md
pyproject.toml
fairvote
respondent
app
experiments
tests
```

---

## 2. Create a virtual environment

Run:

```powershell
py -3.14 -m venv .venv
```

If the `py` launcher does not work, try:

```powershell
python -m venv .venv
```

This creates a local virtual environment folder called `.venv`.

---

## 3. Fix PowerShell execution policy if activation is blocked

Try activating the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell says scripts are disabled, run this command:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Type:

```text
Y
```

then press **Enter**.

Now activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

Your prompt should now start with something like:

```text
(.venv)
```

Important: this Linux/macOS command is **not** for Windows PowerShell:

```bash
source .venv/bin/activate
```

On Windows PowerShell, use:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

## 4. Install FairVote-AI with all development extras

Make sure the virtual environment is active, then run:

```powershell
python -m pip install --upgrade pip
pip install -e ".[dev,ai,streamlit,respondent]"
```

This installs:

- core FairVote-AI code;
- test tools;
- PyTorch for the neural MRP model;
- Streamlit for the dashboard;
- Flask/Gunicorn for the respondent app.

If PyTorch installation fails on your machine, install PyTorch separately using the official selector at <https://pytorch.org/get-started/locally/>, then rerun:

```powershell
pip install -e ".[dev,ai,streamlit,respondent]"
```

---

## 5. Run the Python test suite

Run:

```powershell
pytest -q
```

Run the respondent/browser privacy tests:

```powershell
pytest tests/test_respondent_client_privacy.py -q
pytest tests/test_respondent_rr_js.py -q
pytest tests/test_respondent_server.py -q
```

Run neural model tests:

```powershell
pytest tests/test_rr_neural_mrp.py -q
```

Run experiment tests:

```powershell
pytest tests/test_evaluate_neural_mrp.py -q
pytest tests/test_mrp_vs_baselines.py -q
```

Run the slow MRP integration test before final submission:

```powershell
$env:FV_RUN_SLOW = "1"
pytest tests/test_mrp_vs_baselines.py -q
Remove-Item Env:FV_RUN_SLOW
```

---

## 6. Install Node.js for JavaScript RR tests

The browser-side Randomized Response tests use Node.js.

Check Node is installed:

```powershell
node --version
```

If this fails, install Node.js LTS:

```powershell
winget install OpenJS.NodeJS.LTS
```

Close and reopen PowerShell, reactivate the virtual environment, then rerun:

```powershell
pytest tests/test_respondent_rr_js.py -q
```

These tests verify that `respondent/static/rr.js` applies k-ary Randomized Response and that low epsilon produces frequent flips.

---

## 7. Run the respondent app

Start the respondent server:

```powershell
python respondent/server.py --port 5001
```

Open this in your browser:

```text
http://127.0.0.1:5001
```

Submit one response.

The server writes privatized responses to:

```text
respondent/data/responses.jsonl
```

This file should contain `perturbed_answer` and demographics. It should **not** contain `true_answer`.

---

## 8. Test the duplicate submission block

The respondent app uses `localStorage` to prevent accidental repeat submissions from the same browser.

Test it:

1. Open `http://127.0.0.1:5001`.
2. Submit a response.
3. Refresh the page.
4. The app should show that this browser has already submitted.

This is **only casual duplicate prevention**. It is not secure voter authentication and it is not one-person-one-vote enforcement. A user can clear localStorage, use private browsing, use a different browser, or use another device.

To reset the guard for manual testing:

1. Open browser Developer Tools.
2. Go to **Application** or **Storage**.
3. Find **Local Storage** for `http://127.0.0.1:5001`.
4. Delete keys starting with:

```text
fairvote.submitted.
```

Then refresh the page.

---

## 9. Test audit mode

Audit mode is disabled by default. To enable it, open:

```text
http://127.0.0.1:5001/?audit=1
```

After submitting, the page shows a local audit panel with:

- selected answer index;
- perturbed answer index sent to the server;
- epsilon;
- number of categories `k`;
- whether the answer flipped.

This is for local debugging only. The selected/true answer is **not** sent to the server.

To manually check Randomized Response more clearly, temporarily edit:

```text
respondent/poll_config.json
```

Set:

```json
"epsilon": 0.01
```

Restart the respondent server and open:

```text
http://127.0.0.1:5001/?audit=1
```

With low epsilon, flips should happen frequently. Restore the original epsilon afterwards.

---

## 10. Test that the server rejects `true_answer`

Keep the respondent server running in one PowerShell window.

Open a second PowerShell window in the project folder and run:

```powershell
'{"true_answer":1,"perturbed_answer":2,"demographics":{"age_group":"18-29"}}' | Set-Content bad_payload.json

curl.exe -i -X POST "http://127.0.0.1:5001/api/respond" -H "Content-Type: application/json" --data-binary "@bad_payload.json"

Remove-Item bad_payload.json
```

Expected result:

```text
HTTP/1.1 400 BAD REQUEST
```

The response should say that `true_answer` must not be sent to the server.

This confirms the server-side privacy safeguard is still active.

---

## 11. Run the Streamlit dashboard

Open a new PowerShell window in the project folder.

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Run:

```powershell
streamlit run app/streamlit_app.py
```

Streamlit should open a browser page automatically. If not, open the URL shown in the terminal, usually:

```text
http://localhost:8501
```

---

## 12. Upload synthetic CSVs in the dashboard

Synthetic example files are in:

```text
app/data/
```

Example files include names such as:

```text
poll_no_bias_eps1_n5000_...
poll_nonresponse_eps1_n5000_...
poll_shy_privacy_helps_eps0.5_n5000_...
```

These are synthetic demonstration files. Columns such as `true_choice` or `stated_choice` are included only for simulation/evaluation. Real respondent exports should not contain true votes.

In the dashboard:

1. Upload one synthetic CSV from `app/data/`.
2. Select the reported answer column, usually `reported_choice`.
3. Select demographic columns.
4. Choose an inference method such as RR debiasing, Linear RR-aware MRP, or Neural RR-aware MRP.
5. Run the estimate.

---

## 13. Upload respondent JSONL in the dashboard

After using the respondent app, responses are saved to:

```text
respondent/data/responses.jsonl
```

In the dashboard:

1. Upload `respondent/data/responses.jsonl`.
2. Select the reported answer field, usually `perturbed_answer`.
3. Select demographic fields if available.
4. Run RR debiasing or an MRP method.

Real respondent JSONL should not contain `true_answer`.

---

## 14. Run the small neural experiment

Run:

```powershell
python -m experiments.evaluate_neural_mrp --preset small
```

This checks that the experiment pipeline works and creates output files under:

```text
experiments/outputs/
```

This is a smoke test. It is not final dissertation evidence.

The included final evidence pack is in:

```text
experiments/outputs/final_neural_evidence/
```

It is computationally constrained and labelled as such.

---

## 15. Common troubleshooting

### `source .venv/bin/activate` does not work

That command is for Linux/macOS shells, not Windows PowerShell.

Use:

```powershell
.\.venv\Scripts\Activate.ps1
```

### PowerShell says scripts are disabled

Run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

### `py -3.14` does not work

First check which Python launchers Windows can see:

```powershell
py -0p
python --version
```

If Python 3.14.4 is missing, install it from <https://www.python.org/downloads/windows/> and tick **Add Python to PATH**. Then close and reopen PowerShell and run:

```powershell
py -3.14 -m venv .venv
```

If `python --version` shows Python 3.14.4, this fallback is also acceptable:

```powershell
python -m venv .venv
```

### `pip install` fails while installing PyTorch

Install PyTorch using the official instructions for your machine:

```text
https://pytorch.org/get-started/locally/
```

Then rerun:

```powershell
pip install -e ".[dev,ai,streamlit,respondent]"
```

### JavaScript RR tests are skipped

Install Node.js LTS, close and reopen PowerShell, then check:

```powershell
node --version
```

Then run:

```powershell
pytest tests/test_respondent_rr_js.py -q
```

### Respondent app does not reflect JavaScript changes

The service worker is configured to avoid stale `rr.js` during development. If your browser still seems stale:

1. Open Developer Tools.
2. Go to **Application**.
3. Go to **Service Workers**.
4. Click **Unregister** for the local FairVote service worker.
5. Clear site data for `127.0.0.1:5001`.
6. Refresh the page.

### Duplicate submission block prevents more manual tests

This is expected after one successful submission. Clear localStorage keys starting with:

```text
fairvote.submitted.
```

This duplicate block is not secure authentication. It is only a casual same-browser repeat-submission guard.

### Dashboard cannot find columns

Check whether you uploaded:

- a synthetic CSV from `app/data/`, or
- a respondent JSONL from `respondent/data/responses.jsonl`.

For synthetic CSVs, true labels such as `true_choice` are evaluation-only. For real respondent JSONL, use the reported field such as `perturbed_answer`.

### Flask or Streamlit command not found

Make sure you installed the full extras:

```powershell
pip install -e ".[dev,ai,streamlit,respondent]"
```

Make sure your virtual environment is active:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

## Final reminder

FairVote-AI is a research prototype. It is not production election software. It does not provide secure voter authentication, full anonymity, or guaranteed fairness. It demonstrates local answer-value privacy with Randomized Response and evaluates RR-aware statistical and neural inference methods for aggregate and subgroup estimates.

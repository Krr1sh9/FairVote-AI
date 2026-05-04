# Respondent Web App Manual Verification Checklist

This checklist is for manually verifying the FairVote-AI respondent web app before a demo, viva, or submission. It is beginner-safe and examiner-safe.

The respondent app is a **privacy-preserving polling demo**, not production election software. The localStorage duplicate-submission guard is only casual duplicate prevention. It is **not** secure voter authentication and must not be described as one-person-one-vote protection.

## What this checklist verifies

- The respondent server starts.
- The browser applies Randomized Response before submission.
- The server stores only the perturbed/randomized answer.
- The server rejects any request that includes `true_answer` or other true/raw answer fields such as `true_choice`, `selected_answer`, or `raw_vote`.
- Refreshing the page after a successful submission shows the already-submitted message.
- Audit mode can be used locally to check whether Randomized Response flipped an answer.
- Low epsilon produces frequent flips.
- The app can be tested from another device on the same Wi-Fi if needed.

## 0. Start from the project root

Open PowerShell or a terminal in the extracted `FairVote-AI` project folder.

You should see files and folders such as:

```text
README.md
pyproject.toml
respondent/
app/
fairvote/
experiments/
tests/
```

Activate your virtual environment if you are using one.

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install the respondent dependency if needed:

```powershell
pip install -e ".[respondent]"
```

For the full project environment:

```powershell
pip install -e ".[dev,ai,streamlit,respondent]"
```

## 1. Start the respondent server

From the project root, run:

```powershell
python respondent/server.py --port 5001
```

Expected result:

- The Flask development server starts.
- It listens on `http://127.0.0.1:5001`.
- Keep this PowerShell window open while testing.

If port `5001` is already in use, either close the existing process or use another port, for example:

```powershell
python respondent/server.py --port 5002
```

## 2. Open the respondent page

In your browser, open:

```text
http://127.0.0.1:5001
```

Expected result:

- The poll question loads.
- The answer options are visible.
- The privacy explanation/audit summary is visible.
- No debug/audit submission details are shown by default.

## 3. Submit one response

1. Select one answer option.
2. Fill in any demographic fields if shown.
3. Click **Submit response (with privacy)**.

Expected result:

- The form disappears after successful submission.
- A result screen appears.
- The displayed submitted option may match the selected answer or may differ, depending on Randomized Response.

Important: matching the selected answer once does **not** mean Randomized Response is broken. At normal epsilon, the mechanism often keeps the original answer. Use audit mode and low epsilon testing below to verify flips.

## 4. Refresh and confirm duplicate-submission block

Refresh the browser page.

Expected result:

```text
Already submitted from this browser
```

or equivalent wording explaining that this browser has already submitted the poll.

This is implemented using `localStorage` in the browser.

Important limitation:

- This blocks accidental repeat submissions from the same browser.
- It is **not** secure authentication.
- It can be bypassed by clearing localStorage, using private browsing, another browser, another device, or editing browser storage.
- Do not claim this is real one-person-one-vote protection.

## 5. Inspect `respondent/data/responses.jsonl`

After one submission, inspect:

```text
respondent/data/responses.jsonl
```

PowerShell command:

```powershell
Get-Content respondent\data\responses.jsonl
```

Expected stored record format:

```json
{"perturbed_answer":2,"demographics":{"region":"London"},"timestamp":"..."}
```

The exact value may differ.

Confirm that the stored record contains:

```text
perturbed_answer
demographics
timestamp
```

Confirm that the stored record does **not** contain:

```text
true_answer
true_choice
selected_answer
selectedOption
raw_vote
```

You can search for forbidden fields with PowerShell:

```powershell
Select-String -Path respondent\data\responses.jsonl -Pattern "true_answer|true_choice|selected_answer|selectedOption|raw_vote"
```

Expected result:

- No matches.

## 6. Test server rejection of `true_answer`

Keep the respondent server running.

In a second PowerShell window, run this PowerShell-safe command:

```powershell
'{"true_answer":1,"perturbed_answer":2,"demographics":{"age_group":"18-29"}}' | Set-Content bad_payload.json

curl.exe -i -X POST "http://127.0.0.1:5001/api/respond" -H "Content-Type: application/json" --data-binary "@bad_payload.json"

Remove-Item bad_payload.json
```

Expected result:

- HTTP status `400 Bad Request`.
- JSON error saying `true_answer` must not be sent to the server.

This confirms the server-side safeguard is still active.

## 7. Open audit mode

Audit mode is disabled by default. To enable it for local testing, open:

```text
http://127.0.0.1:5001/?audit=1
```

Expected result:

- A debug/audit panel appears after submission.
- It can show:
  - selected answer index;
  - perturbed answer index;
  - epsilon;
  - number of categories `k`;
  - whether the answer flipped.

Important privacy note:

- Audit mode is for local manual testing only.
- It does **not** send `true_answer` to the server.
- It does **not** store the true answer in `responses.jsonl`.
- Do not use audit mode as evidence of production election readiness.

## 8. Clear localStorage correctly for repeated manual tests

Because the duplicate guard blocks repeat submissions from the same browser, clear the localStorage marker before repeated manual tests.

### Option A: Browser DevTools console

1. Open the respondent page.
2. Press `F12` to open Developer Tools.
3. Go to the **Console** tab.
4. Run:

```javascript
Object.keys(localStorage)
  .filter(k => k.startsWith("fairvote.submitted."))
  .forEach(k => localStorage.removeItem(k));
location.reload();
```

### Option B: Browser storage UI

In Chrome or Edge:

1. Press `F12`.
2. Go to **Application**.
3. Open **Local Storage**.
4. Select `http://127.0.0.1:5001`.
5. Delete keys beginning with:

```text
fairvote.submitted.
```

Then refresh the page.

Do this only for testing. In the real demo, the duplicate-submission message is expected after one submission.

## 9. Temporarily set epsilon to `0.01` and verify frequent flips

This is the clearest manual test that Randomized Response is genuinely applied.

1. Stop the respondent server with `Ctrl+C`.
2. Open:

```text
respondent/poll_config.json
```

3. Temporarily set:

```json
"epsilon": 0.01
```

4. Save the file.
5. Restart the server:

```powershell
python respondent/server.py --port 5001
```

6. Open audit mode:

```text
http://127.0.0.1:5001/?audit=1
```

7. Submit a response.
8. Clear localStorage between repeated attempts using the command in Section 8.
9. Repeat several times.

Expected result:

- With `epsilon = 0.01`, the selected answer should flip frequently.
- With the default five options, the keep probability is about 20.2%, so flips should occur roughly 79.8% of the time.

Do not expect every single attempt to flip. Randomness means occasional kept answers are normal.

## 10. Restore epsilon

After testing, restore `respondent/poll_config.json` to its normal epsilon value.

Check the current intended value in your project. If unsure, use the version from your submitted repository.

Then restart the server:

```powershell
python respondent/server.py --port 5001
```

## 11. Test same-Wi-Fi access with `--host 0.0.0.0` if needed

For a phone or another laptop on the same Wi-Fi, run:

```powershell
python respondent/server.py --host 0.0.0.0 --port 5001
```

Find your computer's local IPv4 address:

```powershell
ipconfig
```

Look for an address like:

```text
192.168.x.x
```

On the other device, open:

```text
http://YOUR_LOCAL_IP:5001
```

Example:

```text
http://192.168.1.25:5001
```

Important safety notes:

- Use this only on a trusted local network for demonstration.
- This is not production deployment.
- Do not expose the Flask development server to the public internet.
- Same-Wi-Fi access does not add authentication or secure one-person-one-vote protection.

## 12. Clean up after manual testing

If you created manual test responses, either remove the test file before final submission or clearly treat it as local test data.

To delete local test responses in PowerShell:

```powershell
Remove-Item respondent\data\responses.jsonl
```

The submitted repository should not include real respondent data.

Keep:

```text
respondent/data/.gitkeep
```

Do not submit:

```text
respondent/data/responses.jsonl
```

## What this checklist proves

This checklist verifies that:

- the respondent server runs;
- the browser submits a privatized reported answer;
- refresh-based duplicate submission is blocked by localStorage;
- no true answer is stored in `responses.jsonl`;
- the server rejects `true_answer` and other true/raw answer-field payloads;
- audit mode can locally show selected/perturbed indices;
- low epsilon produces frequent Randomized Response flips;
- same-Wi-Fi demo access is possible if needed.

## What this checklist does not prove

This checklist does **not** prove:

- secure voter authentication;
- one-person-one-vote enforcement;
- production election readiness;
- full anonymity;
- real-world election accuracy.

FairVote-AI is a final-year project and research prototype for privacy-preserving polling and RR-aware inference, not production election infrastructure.

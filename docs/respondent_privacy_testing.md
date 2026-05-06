# Respondent app privacy and Randomized Response testing

This note explains how to manually verify the respondent web app privacy behaviour.

## What the respondent app should do

The respondent browser applies k-ary Randomized Response before sending a response.
The server should receive only:

```json
{
  "perturbed_answer": 2,
  "demographics": {}
}
```

The browser must not send `true_answer` or other true/raw answer fields such as
`true_choice`, `selected_answer`, `selectedOption`, `raw_vote`, or `raw_answer`, and the server rejects any request that contains them anywhere in the JSON payload, including nested objects/lists.

## Duplicate-submission guard

After a successful submission, the browser stores a localStorage marker for the
current poll. On refresh, the page shows that this browser has already submitted
and does not show the voting form again.

This is only casual duplicate prevention. It is useful for demos and accidental
refreshes, but it is not secure one-person-one-vote authentication. A user can
clear localStorage, use private browsing, or use another device/browser. Do not
describe this as real election voter authentication.

The localStorage marker stores metadata only, such as a timestamp and poll hash.
It does not store the selected true answer.

## Debug/audit mode

Debug/audit mode is disabled by default. To enable it for manual testing, open:

```text
http://127.0.0.1:5001/?audit=1
```

After submission, the page displays local-only audit values:

- selected answer index,
- perturbed answer index sent to the server,
- epsilon,
- number of options `k`,
- whether the answer flipped.

These values are for local testing only. The selected answer index is not sent to
the server and is not stored.

## Manual test: verify Randomized Response with epsilon = 0.01

1. Temporarily edit `respondent/poll_config.json`:

   ```json
   "epsilon": 0.01
   ```

2. Start the respondent server:

   ```bash
   python respondent/server.py --port 5001
   ```

3. Open the app in a fresh browser profile or clear localStorage:

   ```text
   http://127.0.0.1:5001/?audit=1
   ```

4. Submit a response and observe the debug/audit panel.

At `epsilon = 0.01`, with the default five answer options, the expected probability
of keeping the selected answer is roughly:

```text
exp(0.01) / (exp(0.01) + 5 - 1) ≈ 0.202
```

So the answer should flip roughly 79.8% of the time. It will not flip every time,
because Randomized Response is random.

To repeat the manual test in the same browser, clear site data/localStorage or
use a private browsing window. This is expected because the app blocks accidental
repeat submission using a localStorage guard.

## Manual test: server rejects raw-answer fields

Run this while the server is running.

Windows PowerShell-safe method:

```powershell
'{"true_answer":1,"perturbed_answer":2,"demographics":{"age_group":"18-29"}}' | Set-Content bad_payload.json

curl.exe -i -X POST "http://127.0.0.1:5001/api/respond" -H "Content-Type: application/json" --data-binary "@bad_payload.json"

Remove-Item bad_payload.json
```

macOS/Linux bash method:

```bash
curl -i -X POST "http://127.0.0.1:5001/api/respond" \
  -H "Content-Type: application/json" \
  --data-binary '{"true_answer":1,"perturbed_answer":2,"demographics":{"age_group":"18-29"}}'
```

Expected result: HTTP 400 with an error message saying the true/raw answer must not be sent to the server. Repeat the same check with a nested payload such as `{"perturbed_answer":2,"metadata":{"raw_answer":1}}`; this should also be rejected before storage.

## Manual test: analyst export requires a bearer token

By default, `/api/responses` is disabled because it exposes one record per respondent. To test authorised export locally:

```bash
export FAIRVOTE_ANALYST_TOKEN="dev-secret"
python respondent/server.py --port 5001
curl -i http://127.0.0.1:5001/api/responses
curl -i -H "Authorization: Bearer dev-secret" http://127.0.0.1:5001/api/responses
```

The first request should return `401`; the second should return stored perturbed records. Use `/api/results` for aggregate-only output when you do not need individual-level exports.

## Service worker cache note

The service worker intentionally avoids caching `index.html`, `/api/config`, `/static/rr.js`, `/static/app.js`, and `/static/styles.css` with a cache-first strategy. This prevents stale JavaScript from
hiding changes to the Randomized Response implementation during development and
assessment.

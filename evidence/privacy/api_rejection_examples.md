# API raw-answer rejection evidence

This artefact documents server-side rejection cases for `POST /api/respond`. The implementation is split across `respondent/privacy.py`, `respondent/app_factory.py`, and `respondent/storage.py`; `respondent/server.py` remains the CLI/compatibility entrypoint. The executable tests are in `tests/test_respondent_server.py` and `tests/test_respondent_client_privacy.py`.

The server recursively scans request JSON before storage. Any key whose normalised name matches a forbidden raw-answer field is rejected. Rejection happens before the append-only JSONL store is written, so rejected payloads are not stored.

## Example 1: top-level raw answer

Request body:

```json
{
  "true_answer": 1,
  "perturbed_answer": 2,
  "demographics": {
    "age_group": "18-29"
  }
}
```

Expected response:

```text
HTTP 400
REJECTED: true/raw answer fields are not accepted by the server. Forbidden field path(s): $.true_answer. The client must randomise before submission.
```

Expected storage effect: not stored.

## Example 2: nested raw answer

Request body:

```json
{
  "perturbed_answer": 2,
  "demographics": {
    "age_group": "18-29"
  },
  "metadata": {
    "raw_answer": 1
  }
}
```

Expected response:

```text
HTTP 400
REJECTED: true/raw answer fields are not accepted by the server. Forbidden field path(s): $.metadata.raw_answer. The client must randomise before submission.
```

Expected storage effect: not stored.

## Example 3: spelling/case/separator variant

Request body:

```json
{
  "perturbed_answer": 2,
  "demographics": {},
  "Selected Answer": 1
}
```

Expected response:

```text
HTTP 400
REJECTED: true/raw answer fields are not accepted by the server. Forbidden field path(s): $.Selected Answer. The client must randomise before submission.
```

Expected storage effect: not stored.

## Valid aggregate path

For normal analysis, use `GET /api/results`. It returns aggregate counts only and does not expose individual records.

## Privileged individual export path

`GET /api/responses` exposes one perturbed record per respondent and is deliberately protected. It requires an analyst bearer token or can be fully disabled through `FAIRVOTE_DISABLE_INDIVIDUAL_EXPORT=1`.

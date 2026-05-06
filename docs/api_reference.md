# Respondent Server API Reference

Base URL: `http://localhost:5001`

This API is for the respondent collection server. It is intentionally simple: the server stores privatized reported answers, not true answers. The AI inference models run later in the analyst dashboard or experiment scripts; they do not require changes to the collection API.

## Endpoints

### `GET /`

Serves the respondent web client (`index.html`).

### `GET /api/config`

Returns the poll configuration.

**Response:**

```json
{
  "question": "Which party do you intend to vote for?",
  "options": ["Labour", "Conservative", "Reform", "LibDem", "Green"],
  "epsilon": 1.0,
  "demographic_fields": [
    {"name": "age_group", "label": "Age Group", "options": ["18-24", "25-34", "35-44"]},
    {"name": "region", "label": "Region", "options": ["London", "South East"]}
  ]
}
```

### `POST /api/respond`

Submits a privatized response. The client must apply k-ary Randomized Response before calling this endpoint.

**Request body:**

```json
{
  "perturbed_answer": 2,
  "demographics": {
    "age_group": "25-34",
    "region": "London"
  }
}
```

**Success response (201):**

```json
{
  "status": "ok",
  "stored_answer": 2
}
```

**Error responses:**

- `400` if `true_answer` or another true/raw answer field is present anywhere in the JSON payload, including nested objects/lists. Forbidden keys are `true_answer`, `true_choice`, `selected_answer`, `selectedOption`, `raw_vote`, and `raw_answer`.
- `400` if `perturbed_answer` is missing, non-integer, or out of range `[0, k-1]`.
- `400` if `demographics` contains an unknown field, a value not listed in `poll_config.json`, too many fields, or overly long keys/values.
- `413` if the request body exceeds the Flask request-size limit.

### `GET /api/results`

Returns aggregate counts from collected privatized responses.

**Response:**

```json
{
  "total": 150,
  "counts": [25, 40, 30, 20, 35],
  "epsilon": 1.0,
  "k": 5,
  "options": ["Labour", "Conservative", "Reform", "LibDem", "Green"]
}
```

These are counts of reported answers, not counts of true answers.

### `GET /api/responses`

Returns all stored responses as a JSON array for analyst export. This is individual-level data and is disabled unless `FAIRVOTE_ANALYST_TOKEN` is configured. Callers must send `Authorization: Bearer <token>`. Prefer `/api/results` for normal aggregate-only analysis.

**Response:**

```json
[
  {
    "perturbed_answer": 2,
    "demographics": {"age_group": "25-34", "region": "London"},
    "timestamp": "2026-03-29T18:00:00Z"
  }
]
```

## Configuration

Edit `respondent/poll_config.json`:

```json
{
  "question": "Which party do you intend to vote for?",
  "options": ["Labour", "Conservative", "Reform", "LibDem", "Green"],
  "epsilon": 1.0,
  "demographic_fields": [
    {
      "name": "age_group",
      "label": "Age Group",
      "options": ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    }
  ]
}
```

## Server environment variables

| Variable | Purpose | Default |
|---|---|---|
| `FAIRVOTE_ANALYST_TOKEN` | Enables and protects `GET /api/responses`; requests must use `Authorization: Bearer <token>`. | unset, endpoint disabled |
| `FAIRVOTE_ALLOWED_ORIGINS` | Comma-separated explicit CORS origins for API access. Wildcard `*` is rejected. | unset, CORS disabled |
| `FAIRVOTE_MAX_CONTENT_BYTES` | Maximum JSON request body size. | `8192` |
| `FAIRVOTE_TIMESTAMP_PRECISION` | Stored timestamp precision: `none`, `day`, `hour`, `minute`, `second`, or `iso`. | `minute` |
| `FAIRVOTE_MAX_DEMOGRAPHIC_FIELDS` | Maximum number of demographic fields accepted/stored. | `8` |

## Privacy boundary

The collection API enforces the privacy boundary used by the rest of the project:

1. The browser applies k-ary Randomized Response using `crypto.getRandomValues()`.
2. Only the privatized reported answer is transmitted.
3. The server recursively rejects requests containing raw-answer keys anywhere in the JSON payload: `true_answer`, `true_choice`, `selected_answer`, `selectedOption`, `raw_vote`, or `raw_answer`.
4. Demographic labels are accepted only if the field name and value appear in `poll_config.json`; unknown or overly long fields are rejected.
5. Stored data contains privatized responses, validated demographics, and at most a reduced-precision timestamp. `/api/results` is aggregate-only; `/api/responses` requires an analyst bearer token.
6. RR debiasing, RR-aware linear poststratification/MRP, misreport-aware model variants, and RR-aware Neural MRP all operate downstream on the exported reported answers.

The RR-aware Neural MRP model trains on `perturbed_answer`/reported labels plus features. It does not require a new endpoint for true labels. Synthetic files may contain true labels for evaluation, but the respondent API is deliberately designed so real polling mode does not submit or store them.


## Inference and experiment outputs

The respondent API does not expose true votes and does not run inference models directly. Inference happens later in the dashboard or experiment pipeline using privatized reported answers.

The canonical final-evidence command is:

```bash
python -m experiments.mrp_vs_baselines --preset final_evidence
```

Generated run folders under `experiments/outputs/<timestamp>/` contain `raw_trials.csv`, `summary_with_ci.csv`, `paired_comparisons.csv`, `ablations.csv`, `runtime_profile.csv`, `config.json`, `manifest.json` and a run README. Synthetic `true_choice` values may appear in experiment CSVs for evaluation only; they are not accepted by the respondent API in real polling mode.

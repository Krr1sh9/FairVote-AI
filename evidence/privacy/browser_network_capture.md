# Browser-network privacy evidence

This artefact is the committed readable proof point for the browser-side privacy boundary. It is paired with the executable browser test in `tests/test_browser_respondent_privacy.py`.

## What is tested

The Playwright test intercepts the browser `POST /api/respond` request after a user selects an option and submits the form. The test captures the actual JSON body sent by the page and asserts that the network payload contains only:

```json
{
  "perturbed_answer": 0,
  "demographics": {
    "region": "North"
  }
}
```

The integer value is stochastic because k-ary Randomized Response is applied locally in the browser before submission. The important boundary is the schema, not the specific sampled value.

## Raw-answer fields checked as not present

The captured request body must not contain any of the following raw/true-answer keys:

- `true_answer`
- `true_choice`
- `selected_answer`
- `selectedOption`
- `raw_vote`
- `raw_answer`

The test asserts both that these keys are absent from the top-level object and that their string names are not present anywhere in the serialised payload.

## Source of evidence

- Browser implementation: `respondent/static/app.js`
- Randomized Response implementation: `respondent/static/rr.js`
- Executable test: `tests/test_browser_respondent_privacy.py`

## Reproduction command

```bash
pip install -r requirements.lock.txt
pip install -e . --no-deps
python -m playwright install chromium
FV_RUN_BROWSER=1 python -m pytest tests/test_browser_respondent_privacy.py -q
```

## Report-safe interpretation

This evidence supports the narrow claim that the submitted browser client sends a locally perturbed answer value rather than the selected answer value. It does not prove anonymity of IP address, device metadata, traffic timing, or demographic combinations.

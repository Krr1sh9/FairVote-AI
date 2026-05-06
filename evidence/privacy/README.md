# Privacy evidence artefacts

These files are committed to make the privacy boundary verifiable without relying only on prose.

| File | Purpose |
|---|---|
| `browser_network_capture.md` | Documents the Playwright network-capture assertion that the browser sends `perturbed_answer` and not selected/raw-answer fields. |
| `api_rejection_examples.md` | Documents top-level, nested and normalised raw-answer rejection examples. |
| `privacy_report_example.json` | Example rare-cell/export-risk report from the privacy-report boundary. |

Verify the committed artefacts structurally with:

```bash
python -m scripts.verify_privacy_evidence
```

Run executable privacy tests with:

```bash
python -m pytest tests/test_respondent_client_privacy.py tests/test_respondent_server.py -q
FV_RUN_BROWSER=1 python -m pytest tests/test_browser_respondent_privacy.py -q
```

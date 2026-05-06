"""Optional browser-level respondent privacy test using Playwright.

Run with:
    FV_RUN_BROWSER=1 python -m pytest tests/test_browser_respondent_privacy.py -q

Install browser support with:
    pip install -e .[browser]
    python -m playwright install chromium
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.browser

if os.environ.get("FV_RUN_BROWSER") != "1":
    pytest.skip("set FV_RUN_BROWSER=1 to run optional browser privacy tests", allow_module_level=True)

pytest.importorskip("playwright.sync_api")
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
RESPONDENT_DIR = ROOT / "respondent"


def _asset(path: str) -> str:
    return (RESPONDENT_DIR / path).read_text(encoding="utf-8")


def test_browser_submission_sends_perturbed_answer_not_true_answer() -> None:
    captured_payloads: list[dict] = []

    def route_handler(route):
        url = route.request.url
        if url.rstrip("/") == "http://fairvote.test":
            route.fulfill(status=200, content_type="text/html", body=_asset("index.html"))
        elif "/static/rr.js" in url:
            route.fulfill(status=200, content_type="application/javascript", body=_asset("static/rr.js"))
        elif "/static/app.js" in url:
            route.fulfill(status=200, content_type="application/javascript", body=_asset("static/app.js"))
        elif "/static/styles.css" in url:
            route.fulfill(status=200, content_type="text/css", body=_asset("static/styles.css"))
        elif "/static/sw.js" in url:
            route.fulfill(status=200, content_type="application/javascript", body=_asset("static/sw.js"))
        elif "/static/manifest.json" in url:
            route.fulfill(status=200, content_type="application/manifest+json", body=_asset("static/manifest.json"))
        elif "/api/config" in url:
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(
                    {
                        "question": "Browser privacy test?",
                        "options": ["Alpha", "Beta", "Gamma"],
                        "epsilon": 1.0,
                        "demographic_fields": [
                            {"name": "region", "label": "Region", "options": ["North", "South"], "required": False}
                        ],
                    }
                ),
            )
        elif "/api/respond" in url:
            payload = json.loads(route.request.post_data or "{}")
            captured_payloads.append(payload)
            route.fulfill(status=201, content_type="application/json", body=json.dumps({"status": "ok"}))
        else:
            route.fulfill(status=404, body="not found")

    with sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as exc:  # pragma: no cover - depends on local browser install
            pytest.skip(f"Chromium browser is not installed for Playwright: {exc}")
        try:
            page = browser.new_page()
            page.route("**/*", route_handler)
            page.goto("http://fairvote.test/")
            page.locator(".option-btn").nth(0).click()
            page.locator("#demo-region").select_option("North")
            page.locator("#submit-btn").click()
            try:
                page.locator("#result-state").wait_for(state="visible", timeout=5000)
            except PlaywrightTimeoutError as exc:  # pragma: no cover - browser diagnostic
                raise AssertionError("respondent page did not show submitted state") from exc
        finally:
            browser.close()

    assert len(captured_payloads) == 1
    payload = captured_payloads[0]
    assert set(payload) == {"perturbed_answer", "demographics"}
    assert payload["demographics"] == {"region": "North"}
    assert isinstance(payload["perturbed_answer"], int)
    assert 0 <= payload["perturbed_answer"] < 3
    forbidden = {"true_answer", "true_choice", "selected_answer", "selectedOption", "raw_vote", "raw_answer"}
    assert forbidden.isdisjoint(payload.keys())
    assert forbidden.isdisjoint(json.dumps(payload))

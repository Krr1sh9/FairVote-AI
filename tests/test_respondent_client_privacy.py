"""Static privacy checks for the respondent web client.

These tests guard the browser-side privacy boundary. They do not replace browser
E2E tests, but they catch accidental regressions such as adding true_answer to
the JSON payload or removing the local duplicate-submission guard. The selected
raw answer may exist locally in the browser while RR is applied, but it must not
be included in the submitted payload.
"""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "respondent" / "index.html"
APP_JS = ROOT / "respondent" / "static" / "app.js"
SW = ROOT / "respondent" / "static" / "sw.js"

FORBIDDEN = ["true_answer", "true_choice", "selected_answer", "selectedOption", "raw_vote", "raw_answer"]


def _index_html() -> str:
    return INDEX.read_text(encoding="utf-8")


def _app_js() -> str:
    return APP_JS.read_text(encoding="utf-8")


def test_index_uses_external_client_assets_for_strict_csp() -> None:
    html = _index_html()

    assert "https://fonts.googleapis.com" not in html
    assert "<style" not in html
    assert "<script>" not in html
    assert "style=" not in html
    assert 'href="/static/styles.css' in html
    assert 'src="/static/rr.js' in html
    assert 'src="/static/app.js' in html


def test_client_payload_does_not_include_true_answer() -> None:
    # Parse the JavaScript payload literal from app.js and assert that no
    # raw-answer key appears. This catches regressions where a developer
    # accidentally adds the selected option to the POST body.
    js = _app_js()

    payload_start = js.index("const payload = {")
    payload_end = js.index("};", payload_start)
    payload_block = js[payload_start:payload_end]

    assert "perturbed_answer" in payload_block
    assert "demographics" in payload_block
    for key in FORBIDDEN:
        assert key not in payload_block


def test_duplicate_guard_uses_localstorage_without_storing_true_answer() -> None:
    # The localStorage duplicate guard should persist only metadata (poll hash,
    # submission timestamp), never any representation of the true or perturbed
    # answer. Leaking either would undermine the privacy guarantee.
    js = _app_js()

    assert "SUBMISSION_STORAGE_PREFIX" in js
    assert "localStorage.setItem" in js
    assert "client-localStorage-casual-duplicate-prevention" in js
    assert "showAlreadySubmitted" in js

    setitem_start = js.index("localStorage.setItem")
    setitem_end = js.index("));", setitem_start)
    stored_block = js[setitem_start:setitem_end]

    assert "submitted_at" in stored_block
    assert "poll_hash" in stored_block
    for key in FORBIDDEN:
        assert key not in stored_block
    assert "perturbedAnswer" not in stored_block


def test_debug_audit_mode_is_removed_from_production_client() -> None:
    html = _index_html()
    js = _app_js()

    assert "Debug mode disabled" in html
    assert "?audit=1" not in html
    assert "fairvote.auditMode" not in js
    assert "AUDIT_MODE" not in js
    assert 'get("audit") === "1"' not in js
    assert "debug-selected" not in html
    assert "debug-perturbed" not in html
    assert "debugSelected" not in js
    assert "debugPerturbed" not in js

    submit_start = js.index("const payload = {")
    submit_end = js.index("};", submit_start)
    payload_block = js[submit_start:submit_end]
    assert "debugSelected" not in payload_block
    assert "debugPerturbed" not in payload_block


def test_service_worker_avoids_stale_privacy_client_assets() -> None:
    # The service worker must use a network-first strategy for rr.js/app.js/CSS
    # so that updated privacy-mechanism and payload code reaches the browser.
    sw = SW.read_text(encoding="utf-8")

    assert "fairvote-cache-v4-no-stale-client-assets" in sw
    assert "url.pathname === '/static/rr.js'" in sw
    assert "url.pathname === '/static/app.js'" in sw
    assert "url.pathname === '/static/styles.css'" in sw
    assert "Do not cache response submissions" in sw

    static_cache_block = sw.split("STATIC_CACHE_URLS", 1)[1].split("];", 1)[0]
    assert "'/static/rr.js'" not in static_cache_block
    assert "'/static/app.js'" not in static_cache_block
    assert "'/static/styles.css'" not in static_cache_block

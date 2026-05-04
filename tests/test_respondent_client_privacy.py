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
SW = ROOT / "respondent" / "static" / "sw.js"


def _index_html() -> str:
    return INDEX.read_text(encoding="utf-8")


def test_client_payload_does_not_include_true_answer() -> None:
    # Parse the JavaScript payload literal from index.html and assert that no
    # raw-answer key appears.  This catches regressions where a developer
    # accidentally adds the selected option to the POST body.
    html = _index_html()

    payload_start = html.index("const payload = {")
    payload_end = html.index("};", payload_start)
    payload_block = html[payload_start:payload_end]

    assert "perturbed_answer" in payload_block
    assert "demographics" in payload_block
    assert "true_answer" not in payload_block
    assert "true_choice" not in payload_block
    assert "selected_answer" not in payload_block
    assert "selectedOption" not in payload_block
    assert "raw_vote" not in payload_block


def test_duplicate_guard_uses_localstorage_without_storing_true_answer() -> None:
    # The localStorage duplicate guard should persist only metadata (poll hash,
    # submission timestamp), never any representation of the true or perturbed
    # answer.  Leaking either would undermine the privacy guarantee.
    html = _index_html()

    assert "SUBMISSION_STORAGE_PREFIX" in html
    assert "localStorage.setItem" in html
    assert "client-localStorage-casual-duplicate-prevention" in html
    assert "showAlreadySubmitted" in html

    setitem_start = html.index("localStorage.setItem")
    setitem_end = html.index("));", setitem_start)
    stored_block = html[setitem_start:setitem_end]

    assert "submitted_at" in stored_block
    assert "poll_hash" in stored_block
    assert "true_answer" not in stored_block
    assert "true_choice" not in stored_block
    assert "selected_answer" not in stored_block
    assert "selectedOption" not in stored_block
    assert "raw_vote" not in stored_block
    assert "perturbedAnswer" not in stored_block


def test_debug_audit_mode_is_disabled_by_default_and_local_only() -> None:
    html = _index_html()

    assert "Debug/audit mode" in html
    assert "?audit=1" in html
    assert "AUDIT_MODE" in html
    assert 'get("audit") === "1"' in html
    assert "sent to the server" in html

    submit_start = html.index("const payload = {")
    submit_end = html.index("};", submit_start)
    payload_block = html[submit_start:submit_end]
    assert "debugSelected" not in payload_block
    assert "debugPerturbed" not in payload_block


def test_service_worker_avoids_stale_rr_javascript() -> None:
    # The service worker must use a network-first strategy for rr.js so that
    # updated RR parameters reach the browser immediately.  If rr.js appeared
    # in the static cache list, the browser could serve an outdated privacy
    # mechanism without the user realising.
    sw = SW.read_text(encoding="utf-8")

    assert "fairvote-cache-v3-no-stale-rr" in sw
    assert "url.pathname === '/static/rr.js'" in sw
    assert "Do not cache response submissions" in sw
    assert "'/static/rr.js'" not in sw.split("STATIC_CACHE_URLS", 1)[1].split("];", 1)[0]

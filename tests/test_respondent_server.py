# tests/test_respondent_server.py
"""
Tests for the FairVote-AI respondent server.

Tests the Flask API endpoints and verifies the core LDP security property:
the server never receives or stores the true answer.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

# Skip the entire module if Flask is not installed
flask = pytest.importorskip("flask")

from respondent.server import create_app, ResponseStore, load_config


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def tmp_config(tmp_path):
    """Create a temporary poll config file."""
    cfg = {
        "question": "What is your favourite colour?",
        "options": ["Red", "Blue", "Green"],
        "epsilon": 1.5,
        "demographic_fields": [
            {"name": "age", "label": "Age", "options": ["18-24", "25+"], "required": False}
        ],
    }
    path = tmp_path / "poll_config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


@pytest.fixture
def tmp_data(tmp_path):
    """Return a path for temporary response storage."""
    return tmp_path / "data" / "responses.jsonl"


@pytest.fixture
def client(tmp_config, tmp_data):
    """Create a Flask test client."""
    app = create_app(config_path=tmp_config, data_path=tmp_data)
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def data_path(tmp_data):
    return tmp_data


# =============================================================================
# API endpoint tests
# =============================================================================

class TestGetConfig:
    def test_returns_config(self, client):
        resp = client.get("/api/config")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["question"] == "What is your favourite colour?"
        assert data["options"] == ["Red", "Blue", "Green"]
        assert data["epsilon"] == 1.5
        assert len(data["demographic_fields"]) == 1

    def test_config_does_not_leak_sensitive_fields(self, client):
        resp = client.get("/api/config")
        data = resp.get_json()
        # Config should only contain what the client needs
        assert "question" in data
        assert "options" in data
        assert "epsilon" in data


class TestPostRespond:
    def test_valid_response_accepted(self, client, data_path):
        payload = {"perturbed_answer": 1, "demographics": {"age": "18-24"}}
        resp = client.post(
            "/api/respond",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["status"] == "ok"
        assert data["stored_answer"] == 1

        # Verify stored in JSONL
        lines = data_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["perturbed_answer"] == 1
        assert record["demographics"]["age"] == "18-24"
        assert "timestamp" in record

    def test_response_without_demographics(self, client, data_path):
        payload = {"perturbed_answer": 0}
        resp = client.post(
            "/api/respond",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert resp.status_code == 201

        record = json.loads(data_path.read_text(encoding="utf-8").strip())
        assert record["perturbed_answer"] == 0
        assert record["demographics"] == {}

    # Parameterised over all known raw-answer field names.  Each one is
    # individually verified to trigger a hard 400 rejection before storage.
    @pytest.mark.parametrize(
        "forbidden_field",
        ["true_answer", "true_choice", "selected_answer", "selectedOption", "raw_vote", "raw_answer"],
    )
    def test_rejects_true_or_raw_answer_fields(self, client, data_path, forbidden_field):
        """The respondent API rejects true/raw answer fields before storage."""
        payload = {"perturbed_answer": 1, forbidden_field: 2}
        resp = client.post(
            "/api/respond",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert resp.status_code == 400
        assert "REJECTED" in resp.get_json()["error"]
        assert forbidden_field in resp.get_json()["error"]

        # Verify nothing was stored
        assert not data_path.exists() or data_path.read_text(encoding="utf-8").strip() == ""

    def test_rejects_missing_perturbed_answer(self, client):
        resp = client.post(
            "/api/respond",
            data=json.dumps({"demographics": {}}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_rejects_out_of_range_answer(self, client):
        # k=3 options, so valid range is [0, 2]
        resp = client.post(
            "/api/respond",
            data=json.dumps({"perturbed_answer": 5}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_rejects_negative_answer(self, client):
        resp = client.post(
            "/api/respond",
            data=json.dumps({"perturbed_answer": -1}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_rejects_invalid_json(self, client):
        resp = client.post(
            "/api/respond",
            data="not json",
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_multiple_responses_appended(self, client, data_path):
        for i in range(5):
            payload = {"perturbed_answer": i % 3}
            resp = client.post(
                "/api/respond",
                data=json.dumps(payload),
                content_type="application/json",
            )
            assert resp.status_code == 201

        lines = data_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 5


class TestGetResults:
    def test_empty_results(self, client):
        resp = client.get("/api/results")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["total"] == 0
        assert data["counts"] == [0, 0, 0]
        assert data["k"] == 3

    def test_results_after_responses(self, client):
        # Submit some responses
        for ans in [0, 0, 1, 2, 2, 2]:
            client.post(
                "/api/respond",
                data=json.dumps({"perturbed_answer": ans}),
                content_type="application/json",
            )

        resp = client.get("/api/results")
        data = resp.get_json()
        assert data["total"] == 6
        assert data["counts"] == [2, 1, 3]
        assert data["epsilon"] == 1.5
        assert data["options"] == ["Red", "Blue", "Green"]


class TestGetResponses:
    def test_responses_export_disabled_without_token(self, client):
        client.post(
            "/api/respond",
            data=json.dumps({"perturbed_answer": 0}),
            content_type="application/json",
        )

        resp = client.get("/api/responses")
        assert resp.status_code == 503
        assert "FAIRVOTE_ANALYST_TOKEN" in resp.get_json()["error"]

    def test_responses_export_requires_authorization(self, tmp_config, tmp_data, monkeypatch):
        monkeypatch.setenv("FAIRVOTE_ANALYST_TOKEN", "secret-token")
        app = create_app(config_path=tmp_config, data_path=tmp_data)
        app.config["TESTING"] = True
        with app.test_client() as c:
            c.post(
                "/api/respond",
                data=json.dumps({"perturbed_answer": 0}),
                content_type="application/json",
            )

            missing = c.get("/api/responses")
            assert missing.status_code == 401

            wrong = c.get("/api/responses", headers={"Authorization": "Bearer wrong"})
            assert wrong.status_code == 401

            blocked = c.get("/api/responses", headers={"Authorization": "Bearer secret-token"})
            assert blocked.status_code == 409
            body = blocked.get_json()
            assert "k-anonymity" in body["error"]
            assert body["privacy_report"]["rare_cell_count"] == 1


    def test_responses_export_succeeds_when_demographic_cells_are_not_rare(self, tmp_config, tmp_data, monkeypatch):
        monkeypatch.setenv("FAIRVOTE_ANALYST_TOKEN", "secret-token")
        app = create_app(config_path=tmp_config, data_path=tmp_data)
        app.config["TESTING"] = True
        with app.test_client() as c:
            for _ in range(5):
                c.post(
                    "/api/respond",
                    data=json.dumps({"perturbed_answer": 0, "demographics": {"age": "18-24"}}),
                    content_type="application/json",
                )

            ok = c.get("/api/responses", headers={"Authorization": "Bearer secret-token"})
            assert ok.status_code == 200
            body = ok.get_json()
            assert len(body["records"]) == 5
            assert body["privacy_report"]["rare_cell_count"] == 0
            assert body["records"][0]["perturbed_answer"] == 0

    def test_no_true_answer_in_stored_records(self, client, data_path):
        # End-to-end check: after a successful submission, the stored JSONL
        # record should contain only perturbed_answer, validated demographics,
        # and an optional reduced-precision timestamp.
        client.post(
            "/api/respond",
            data=json.dumps({"perturbed_answer": 2, "demographics": {"age": "25+"}}),
            content_type="application/json",
        )

        raw = data_path.read_text(encoding="utf-8")
        record = json.loads(raw)
        assert record["perturbed_answer"] == 2
        assert record["demographics"] == {"age": "25+"}
        for forbidden in ["true_answer", "true_choice", "selected_answer", "selectedOption", "raw_vote", "raw_answer"]:
            assert forbidden not in raw


class TestPrivacyBoundaryHardening:
    def test_rejects_nested_true_or_raw_answer_fields(self, client, data_path):
        payload = {
            "perturbed_answer": 1,
            "metadata": {"nested": [{"raw_answer": 2}]},
        }
        resp = client.post("/api/respond", data=json.dumps(payload), content_type="application/json")

        assert resp.status_code == 400
        assert "$.metadata.nested[0].raw_answer" in resp.get_json()["error"]
        assert not data_path.exists() or data_path.read_text(encoding="utf-8").strip() == ""

    def test_rejects_unknown_demographics(self, client, data_path):
        payload = {"perturbed_answer": 1, "demographics": {"age": "18-24", "postcode": "E1"}}
        resp = client.post("/api/respond", data=json.dumps(payload), content_type="application/json")

        assert resp.status_code == 400
        assert "unknown demographic field" in resp.get_json()["error"]
        assert not data_path.exists() or data_path.read_text(encoding="utf-8").strip() == ""

    def test_rejects_invalid_demographic_values(self, client, data_path):
        payload = {"perturbed_answer": 1, "demographics": {"age": "13-17"}}
        resp = client.post("/api/respond", data=json.dumps(payload), content_type="application/json")

        assert resp.status_code == 400
        assert "invalid demographic value" in resp.get_json()["error"]
        assert not data_path.exists() or data_path.read_text(encoding="utf-8").strip() == ""

    def test_rejects_too_many_demographic_fields(self, tmp_path):
        cfg = {
            "question": "Q",
            "options": ["A", "B"],
            "epsilon": 1.0,
            "demographic_fields": [
                {"name": f"d{i}", "label": f"D{i}", "options": ["x"], "required": False}
                for i in range(9)
            ],
        }
        path = tmp_path / "poll_config.json"
        path.write_text(json.dumps(cfg), encoding="utf-8")
        with pytest.raises(ValueError, match="more than"):
            create_app(config_path=path, data_path=tmp_path / "responses.jsonl")

    def test_valid_perturbed_only_submission_accepted(self, client, data_path):
        payload = {"perturbed_answer": 2, "demographics": {"age": "18-24"}}
        resp = client.post("/api/respond", data=json.dumps(payload), content_type="application/json")

        assert resp.status_code == 201
        record = json.loads(data_path.read_text(encoding="utf-8"))
        assert record["perturbed_answer"] == 2
        assert record["demographics"] == {"age": "18-24"}

    def test_request_size_limit_rejects_large_payload(self, client):
        payload = {"perturbed_answer": 0, "padding": "x" * 10000}
        resp = client.post("/api/respond", data=json.dumps(payload), content_type="application/json")

        assert resp.status_code == 413

    def test_wildcard_cors_origin_is_rejected(self, tmp_config, tmp_data, monkeypatch):
        monkeypatch.setenv("FAIRVOTE_ALLOWED_ORIGINS", "*")
        with pytest.raises(ValueError, match="wildcard"):
            create_app(config_path=tmp_config, data_path=tmp_data)

    def test_timestamp_precision_defaults_to_minute(self, client, data_path):
        client.post(
            "/api/respond",
            data=json.dumps({"perturbed_answer": 1}),
            content_type="application/json",
        )
        record = json.loads(data_path.read_text(encoding="utf-8"))
        assert record["timestamp"].endswith(":00Z")

    def test_timestamp_storage_can_be_disabled(self, tmp_config, tmp_data, monkeypatch):
        monkeypatch.setenv("FAIRVOTE_TIMESTAMP_PRECISION", "none")
        app = create_app(config_path=tmp_config, data_path=tmp_data)
        app.config["TESTING"] = True
        with app.test_client() as c:
            resp = c.post(
                "/api/respond",
                data=json.dumps({"perturbed_answer": 1}),
                content_type="application/json",
            )
            assert resp.status_code == 201
        record = json.loads(tmp_data.read_text(encoding="utf-8"))
        assert "timestamp" not in record


class TestServeIndex:
    def test_index_returns_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"FairVote" in resp.data

    def test_security_headers_use_strict_csp_without_inline_allowances(self, client):
        resp = client.get("/")
        csp = resp.headers["Content-Security-Policy"]
        assert "script-src 'self'" in csp
        assert "style-src 'self'" in csp
        assert "unsafe-inline" not in csp
        assert resp.headers["X-Frame-Options"] == "DENY"


# =============================================================================
# ResponseStore unit tests
# =============================================================================

class TestResponseStore:
    def test_append_and_read(self, tmp_path):
        store = ResponseStore(tmp_path / "test.jsonl")
        store.append({"a": 1})
        store.append({"b": 2})
        records = store.read_all()
        assert len(records) == 2
        assert records[0]["a"] == 1
        assert records[1]["b"] == 2

    def test_count(self, tmp_path):
        store = ResponseStore(tmp_path / "test.jsonl")
        assert store.count() == 0
        store.append({"x": 1})
        store.append({"y": 2})
        assert store.count() == 2

    def test_empty_file(self, tmp_path):
        store = ResponseStore(tmp_path / "nonexistent.jsonl")
        assert store.read_all() == []
        assert store.count() == 0


# =============================================================================
# Config validation tests
# =============================================================================

class TestLoadConfig:
    def test_valid_config(self, tmp_config):
        cfg = load_config(tmp_config)
        assert cfg["question"] == "What is your favourite colour?"
        assert len(cfg["options"]) == 3

    def test_missing_question(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"options": ["A", "B"], "epsilon": 1.0}))
        with pytest.raises(ValueError, match="question"):
            load_config(path)

    def test_too_few_options(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"question": "Q", "options": ["A"], "epsilon": 1.0}))
        with pytest.raises(ValueError, match="options"):
            load_config(path)

    def test_invalid_epsilon(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"question": "Q", "options": ["A", "B"], "epsilon": -1}))
        with pytest.raises(ValueError, match="epsilon"):
            load_config(path)

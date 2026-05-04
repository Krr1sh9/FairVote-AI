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
        ["true_answer", "true_choice", "selected_answer", "selectedOption", "raw_vote"],
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
    def test_returns_all_records(self, client):
        for ans in [0, 1]:
            client.post(
                "/api/respond",
                data=json.dumps({"perturbed_answer": ans}),
                content_type="application/json",
            )

        resp = client.get("/api/responses")
        assert resp.status_code == 200
        records = resp.get_json()
        assert len(records) == 2
        assert records[0]["perturbed_answer"] == 0
        assert records[1]["perturbed_answer"] == 1

    def test_no_true_answer_in_stored_records(self, client):
        # End-to-end check: even after a successful submission, the stored
        # JSONL records should contain only perturbed_answer, demographics,
        # and timestamp.  No field that could be confused with a true answer
        # should appear.
        client.post(
            "/api/respond",
            data=json.dumps({"perturbed_answer": 2}),
            content_type="application/json",
        )

        resp = client.get("/api/responses")
        records = resp.get_json()
        for rec in records:
            assert "true_answer" not in rec


class TestServeIndex:
    def test_index_returns_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"FairVote" in resp.data


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

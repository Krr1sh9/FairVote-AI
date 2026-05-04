# respondent/server.py
"""
FairVote-AI Respondent Server

A minimal Flask API that:
  1. Serves the respondent web client (index.html)
  2. Provides poll configuration via GET /api/config
  3. Accepts perturbed responses via POST /api/respond
  4. Returns aggregate counts via GET /api/results (analyst-only)

The respondent protocol sends only the perturbed (RR-randomised) answer.
Raw answer fields are rejected before storage.

Run:
    pip install -e ".[dev,ai,streamlit,respondent]"
    python respondent/server.py
    # Then open http://localhost:5001 in any browser
"""
from __future__ import annotations

import argparse
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from flask import Flask, jsonify, request, send_from_directory, abort
except ImportError:
    raise ImportError(
        "Flask is required for the respondent server.\n"
        'Install it with:  pip install -e ".[dev,ai,streamlit,respondent]"'
    )


# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = BASE_DIR / "poll_config.json"
DEFAULT_DATA_PATH = BASE_DIR / "data" / "responses.jsonl"


def load_config(path: Path) -> dict:
    """Load and validate the public poll configuration.

    The option list defines category indices used by the client-side RR code;
    it is not a place where respondent answers are stored.
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Validate required fields
    if "question" not in cfg or not cfg["question"]:
        raise ValueError("poll_config.json must have a 'question' field.")
    if "options" not in cfg or len(cfg["options"]) < 2:
        raise ValueError("poll_config.json must have >= 2 'options'.")
    if "epsilon" not in cfg or float(cfg["epsilon"]) <= 0:
        raise ValueError("poll_config.json must have a positive 'epsilon'.")

    return cfg


# =============================================================================
# Storage  (append-only JSONL file, thread-safe)
# =============================================================================

class ResponseStore:
    """Thread-safe append-only storage for perturbed responses.

    The store deliberately writes one JSON object per line so records can be
    appended without rewriting the file. Each record has already passed the
    server-side raw-answer rejection checks.
    """

    def __init__(self, path: Path):
        self._path = Path(path)
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict) -> None:
        """Append a single JSON record to the JSONL file."""
        line = json.dumps(record, separators=(",", ":")) + "\n"
        with self._lock:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line)

    def read_all(self) -> List[dict]:
        """Read all stored records."""
        if not self._path.exists():
            return []
        records = []
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def count(self) -> int:
        """Count total stored responses (without loading all into memory)."""
        if not self._path.exists():
            return 0
        n = 0
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    n += 1
        return n

    @property
    def path(self) -> Path:
        return self._path


# =============================================================================
# Flask app factory
# =============================================================================

def create_app(
    config_path: Optional[Path] = None,
    data_path: Optional[Path] = None,
) -> Flask:
    """Create and configure the Flask application."""

    config_path = Path(config_path or DEFAULT_CONFIG_PATH)
    data_path = Path(data_path or DEFAULT_DATA_PATH)

    poll_config = load_config(config_path)
    store = ResponseStore(data_path)
    # k is the number of poll options and determines the valid range [0, k-1]
    # for perturbed_answer values accepted by the API.
    k = len(poll_config["options"])

    app = Flask(
        __name__,
        static_folder=str(BASE_DIR / "static"),
        static_url_path="/static",
    )

    # ── Serve index.html ──
    @app.route("/")
    def index():
        return send_from_directory(str(BASE_DIR), "index.html")

    # ── GET /api/config ──
    @app.route("/api/config")
    def api_config():
        """Return poll configuration (question, options, epsilon, demographics)."""
        return jsonify({
            "question": poll_config["question"],
            "options": poll_config["options"],
            "epsilon": float(poll_config["epsilon"]),
            "demographic_fields": poll_config.get("demographic_fields", []),
        })

    # ── POST /api/respond ──
    @app.route("/api/respond", methods=["POST"])
    def api_respond():
        """Accept a perturbed response from the respondent client.

        Expected JSON body:
        {
            "perturbed_answer": int,       # in [0, k-1]
            "demographics": { ... }        # optional dict
        }

        The respondent API accepts only the perturbed answer and demographics.
        If a true/raw answer field such as 'true_answer' or 'true_choice' is
        present, the request is rejected before anything is stored.
        """
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({"error": "Invalid JSON body"}), 400

        # Privacy boundary: reject payloads that contain raw answer fields before
        # anything is written to disk. The respondent protocol sends only
        # perturbed_answer plus demographics.
        # These aliases are rejected to prevent accidental uploads of raw votes
        # from synthetic/evaluation CSVs or manual testing tools.
        forbidden_true_fields = {
            "true_answer",
            "true_choice",
            "selected_answer",
            "selectedOption",
            "raw_vote",
        }
        leaked_fields = sorted(forbidden_true_fields.intersection(data))
        if leaked_fields:
            # A hard rejection is preferable to silently dropping fields, because
            # it exposes client-side regressions during manual testing.
            return jsonify({
                "error": "REJECTED: true/raw answer fields are not accepted by the server. "
                         f"Forbidden field(s): {', '.join(leaked_fields)}. "
                         "The client must randomise before submission."
            }), 400

        # Validate perturbed_answer
        perturbed = data.get("perturbed_answer")
        if perturbed is None:
            return jsonify({"error": "Missing 'perturbed_answer' field"}), 400
        try:
            perturbed = int(perturbed)
        except (TypeError, ValueError):
            return jsonify({"error": "'perturbed_answer' must be an integer"}), 400
        if perturbed < 0 or perturbed >= k:
            return jsonify({"error": f"'perturbed_answer' must be in [0, {k-1}]"}), 400

        # Demographics are optional and intentionally free-form; the server
        # does not validate individual field values.
        demographics = data.get("demographics", {})
        if not isinstance(demographics, dict):
            demographics = {}

        # Store only the perturbed category, optional demographics, and a server
        # timestamp. There is deliberately no raw-answer field in the record.
        record = {
            "perturbed_answer": perturbed,
            "demographics": demographics,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        store.append(record)

        return jsonify({"status": "ok", "stored_answer": perturbed}), 201

    # ── GET /api/results ──
    @app.route("/api/results")
    def api_results():
        """Return aggregate counts from collected perturbed responses.

        This endpoint is intended for analysts.  It returns:
        - total: number of responses
        - counts: per-option count of perturbed answers
        - epsilon: the privacy parameter used
        - options: the option labels
        """
        records = store.read_all()
        # Aggregate the perturbed answers into per-option counts.  These are
        # the privatized counts an analyst would debias with the known epsilon.
        counts = [0] * k
        for rec in records:
            ans = rec.get("perturbed_answer", -1)
            if 0 <= ans < k:
                counts[ans] += 1

        return jsonify({
            "total": len(records),
            "counts": counts,
            "epsilon": float(poll_config["epsilon"]),
            "k": k,
            "options": poll_config["options"],
        })

    # ── GET /api/responses (full data export for analyst) ──
    @app.route("/api/responses")
    def api_responses():
        """Return all stored responses as a JSON array (for analyst import)."""
        return jsonify(store.read_all())

    return app


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="FairVote-AI Respondent Server — serves the privacy-preserving poll client."
    )
    parser.add_argument(
        "--config", type=str, default=str(DEFAULT_CONFIG_PATH),
        help="Path to poll_config.json"
    )
    parser.add_argument(
        "--data", type=str, default=str(DEFAULT_DATA_PATH),
        help="Path to responses.jsonl storage file"
    )
    parser.add_argument(
        "--port", type=int, default=5001,
        help="Port to listen on (default: 5001)"
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable Flask debug mode"
    )
    args = parser.parse_args()

    app = create_app(
        config_path=Path(args.config),
        data_path=Path(args.data),
    )

    print(f"\n  FairVote Respondent Server")
    print(f"  Poll: {args.config}")
    print(f"  Data: {args.data}")
    print(f"  URL:  http://{args.host}:{args.port}/\n")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

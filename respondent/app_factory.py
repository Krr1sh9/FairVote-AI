"""Flask app factory for the respondent collection server."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from flask import Flask, jsonify, request, send_from_directory
    from werkzeug.exceptions import RequestEntityTooLarge
except ImportError as exc:  # pragma: no cover - exercised only without Flask installed
    raise ImportError(
        'Flask is required for the respondent server.\nInstall it with:  pip install -e ".[respondent]"'
    ) from exc

try:  # pragma: no cover - optional dependency branch
    from flask_cors import CORS

    HAS_CORS = True
except ImportError:  # pragma: no cover - optional dependency branch
    CORS = None  # type: ignore[assignment]
    HAS_CORS = False

from .config import (
    BASE_DIR,
    DEFAULT_CONFIG_PATH,
    DEFAULT_DATA_PATH,
    DEFAULT_K_ANONYMITY,
    DEFAULT_MAX_CONTENT_LENGTH,
    load_config,
)
from .privacy import (
    PayloadRejected,
    allowed_cors_origins,
    authorised_for_response_export,
    exportable_individual_records,
    find_forbidden_fields,
    privacy_report,
    timestamp_for_storage,
    validate_demographics,
)
from .storage import ResponseStore


def _security_headers(response: Any) -> Any:
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "0"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=(), payment=()"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self'; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "font-src 'self'; "
        "object-src 'none'; "
        "base-uri 'none'; "
        "frame-ancestors 'none'; "
        "manifest-src 'self'"
    )
    return response


def create_app(config_path: Path | None = None, data_path: Path | None = None) -> Flask:
    """Create and configure the Flask respondent application."""
    config_path = Path(config_path or DEFAULT_CONFIG_PATH)
    data_path = Path(data_path or DEFAULT_DATA_PATH)

    poll_config = load_config(config_path)
    store = ResponseStore(data_path)
    k = len(poll_config["options"])

    app = Flask(__name__, static_folder=str(BASE_DIR / "static"), static_url_path="/static")
    app.config["MAX_CONTENT_LENGTH"] = DEFAULT_MAX_CONTENT_LENGTH
    app.config["POLL_CONFIG"] = poll_config
    app.config["RESPONSE_STORE"] = store

    allowed_origins = allowed_cors_origins()
    if HAS_CORS and allowed_origins:
        CORS(app, resources={r"/api/*": {"origins": allowed_origins}})

    @app.errorhandler(RequestEntityTooLarge)
    def request_too_large(_exc: RequestEntityTooLarge):
        return jsonify({"error": f"Request body too large; limit is {app.config['MAX_CONTENT_LENGTH']} bytes"}), 413

    @app.after_request
    def add_security_headers(response):
        return _security_headers(response)

    @app.route("/")
    def index():
        return send_from_directory(str(BASE_DIR), "index.html")

    @app.route("/api/config")
    def api_config():
        return jsonify(
            {
                "question": poll_config["question"],
                "options": poll_config["options"],
                "epsilon": float(poll_config["epsilon"]),
                "demographic_fields": poll_config.get("demographic_fields", []),
            }
        )

    @app.route("/api/respond", methods=["POST"])
    def api_respond():
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({"error": "Invalid JSON body"}), 400
        if not isinstance(data, dict):
            return jsonify({"error": "JSON body must be an object"}), 400

        leaked_fields = find_forbidden_fields(data)
        if leaked_fields:
            return (
                jsonify(
                    {
                        "error": "REJECTED: true/raw answer fields are not accepted by the server. "
                        f"Forbidden field path(s): {', '.join(leaked_fields)}. "
                        "The client must randomise before submission."
                    }
                ),
                400,
            )

        perturbed = data.get("perturbed_answer")
        if perturbed is None:
            return jsonify({"error": "Missing 'perturbed_answer' field"}), 400
        try:
            if isinstance(perturbed, bool):
                raise ValueError
            perturbed = int(perturbed)
        except (TypeError, ValueError) as _exc:
            return jsonify({"error": "'perturbed_answer' must be an integer"}), 400
        if perturbed < 0 or perturbed >= k:
            return jsonify({"error": f"'perturbed_answer' must be in [0, {k - 1}]"}), 400

        try:
            demographics = validate_demographics(data.get("demographics", {}), poll_config)
        except PayloadRejected as exc:
            return jsonify({"error": f"Invalid demographics: {exc}"}), 400

        record: dict[str, Any] = {"perturbed_answer": perturbed, "demographics": demographics}
        timestamp = timestamp_for_storage()
        if timestamp is not None:
            record["timestamp"] = timestamp
        store.append(record)

        return jsonify({"status": "ok", "stored_answer": perturbed}), 201

    @app.route("/api/results")
    def api_results():
        records = store.read_all()
        counts = [0] * k
        for rec in records:
            ans = rec.get("perturbed_answer", -1)
            if isinstance(ans, int) and 0 <= ans < k:
                counts[ans] += 1

        return jsonify(
            {
                "total": len(records),
                "counts": counts,
                "epsilon": float(poll_config["epsilon"]),
                "k": k,
                "options": poll_config["options"],
            }
        )

    @app.route("/api/responses")
    def api_responses():
        ok, message = authorised_for_response_export(request.headers.get("Authorization", ""))
        if not ok:
            status = 503 if "disabled" in message.lower() else 401
            return jsonify({"error": message}), status
        records = store.read_all()
        export_records, report, block_message = exportable_individual_records(records, k_anonymity=DEFAULT_K_ANONYMITY)
        if block_message:
            return jsonify({"error": block_message, "privacy_report": report}), 409
        return jsonify({"records": export_records, "privacy_report": report})

    @app.route("/api/privacy-report")
    def api_privacy_report():
        return jsonify(privacy_report(store.read_all(), k_anonymity=DEFAULT_K_ANONYMITY))

    return app

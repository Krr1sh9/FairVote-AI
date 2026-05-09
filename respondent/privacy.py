"""Privacy-boundary validation and individual-export controls."""

from __future__ import annotations

import hashlib
import hmac
import os
import re
from datetime import UTC, datetime
from typing import Any

from .config import (
    DEFAULT_K_ANONYMITY,
    DEFAULT_TIMESTAMP_PRECISION,
    MAX_DEMOGRAPHIC_FIELDS,
    MAX_DEMOGRAPHIC_KEY_LENGTH,
    MAX_DEMOGRAPHIC_VALUE_LENGTH,
)

FORBIDDEN_RAW_ANSWER_KEYS = frozenset(
    {
        "true_answer",
        "true_choice",
        "selected_answer",
        "selectedOption",
        "selected_option",
        "raw_vote",
        "raw_answer",
        "raw_choice",
        "actual_answer",
        "actual_choice",
        "unperturbed_answer",
        "unrandomized_answer",
        "original_answer",
        "chosen_answer",
    }
)
FORBIDDEN_RAW_ANSWER_KEY_NORMALISED = frozenset(
    re.sub(r"[^a-z0-9]", "", key.lower()) for key in FORBIDDEN_RAW_ANSWER_KEYS
)


class PayloadRejected(ValueError):
    """Raised when a request violates the respondent privacy boundary."""


def find_forbidden_fields(payload: Any, *, path: str = "$") -> list[str]:
    """Return paths of forbidden raw-answer keys anywhere inside a JSON value."""
    hits: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            key_str = str(key)
            child_path = f"{path}.{key_str}"
            key_norm = re.sub(r"[^a-z0-9]", "", key_str.lower())
            if key_str in FORBIDDEN_RAW_ANSWER_KEYS or key_norm in FORBIDDEN_RAW_ANSWER_KEY_NORMALISED:
                hits.append(child_path)
            hits.extend(find_forbidden_fields(value, path=child_path))
    elif isinstance(payload, list):
        for idx, value in enumerate(payload):
            hits.extend(find_forbidden_fields(value, path=f"{path}[{idx}]"))
    return hits


def demographic_schema(poll_config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {field["name"]: field for field in poll_config.get("demographic_fields", [])}


def validate_demographics(demographics: Any, poll_config: dict[str, Any]) -> dict[str, str]:
    """Validate respondent demographics against poll_config.json and return a sanitized dict."""
    if demographics is None:
        demographics = {}
    if not isinstance(demographics, dict):
        raise PayloadRejected("'demographics' must be an object")
    if len(demographics) > MAX_DEMOGRAPHIC_FIELDS:
        raise PayloadRejected(f"too many demographic fields; maximum is {MAX_DEMOGRAPHIC_FIELDS}")

    schema = demographic_schema(poll_config)
    sanitized: dict[str, str] = {}
    for key, value in demographics.items():
        if not isinstance(key, str):
            raise PayloadRejected("demographic field names must be strings")
        if len(key) > MAX_DEMOGRAPHIC_KEY_LENGTH:
            raise PayloadRejected(f"demographic field name exceeds maximum length {MAX_DEMOGRAPHIC_KEY_LENGTH}")
        if key not in schema:
            raise PayloadRejected(f"unknown demographic field: {key}")
        if not isinstance(value, str):
            raise PayloadRejected(f"demographic value for {key!r} must be a string")
        if len(value) > MAX_DEMOGRAPHIC_VALUE_LENGTH:
            raise PayloadRejected(
                f"demographic value for {key!r} exceeds maximum length {MAX_DEMOGRAPHIC_VALUE_LENGTH}"
            )
        if value == "":
            continue
        if value not in schema[key]["options"]:
            raise PayloadRejected(f"invalid demographic value for {key!r}: {value}")
        sanitized[key] = value

    missing_required = [name for name, field in schema.items() if field.get("required") and name not in sanitized]
    if missing_required:
        raise PayloadRejected(f"missing required demographic field(s): {', '.join(missing_required)}")
    return sanitized


def timestamp_for_storage() -> str | None:
    """Return a reduced-precision timestamp, or None when disabled."""
    precision = os.getenv("FAIRVOTE_TIMESTAMP_PRECISION", DEFAULT_TIMESTAMP_PRECISION).lower()
    now = datetime.now(UTC)
    if precision in {"none", "off", "false", "0"}:
        return None
    if precision == "day":
        return now.date().isoformat()
    if precision == "hour":
        return now.replace(minute=0, second=0, microsecond=0).isoformat().replace("+00:00", "Z")
    if precision == "minute":
        return now.replace(second=0, microsecond=0).isoformat().replace("+00:00", "Z")
    if precision == "second":
        return now.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    if precision == "iso":
        return now.isoformat().replace("+00:00", "Z")
    raise ValueError("FAIRVOTE_TIMESTAMP_PRECISION must be none, day, hour, minute, second, or iso")


def allowed_cors_origins() -> list[str]:
    raw = os.getenv("FAIRVOTE_ALLOWED_ORIGINS", "").strip()
    if not raw:
        return []
    origins = [part.strip() for part in raw.split(",") if part.strip()]
    if "*" in origins:
        raise ValueError("FAIRVOTE_ALLOWED_ORIGINS must list explicit origins; wildcard '*' is not allowed")
    return origins


def configured_export_enabled() -> bool:
    return bool(os.getenv("FAIRVOTE_ANALYST_TOKEN") or os.getenv("FAIRVOTE_ANALYST_TOKEN_SHA256"))


def authorised_for_response_export(auth_header: str) -> tuple[bool, str]:
    """Validate analyst export credentials without requiring plaintext token storage."""
    if os.getenv("FAIRVOTE_DISABLE_INDIVIDUAL_EXPORT", "").lower() in {"1", "true", "yes", "on"}:
        return False, "Full response export is disabled by FAIRVOTE_DISABLE_INDIVIDUAL_EXPORT."
    configured_plain = [part.strip() for part in os.getenv("FAIRVOTE_ANALYST_TOKEN", "").split(",") if part.strip()]
    configured_hashes = [
        part.strip().lower() for part in os.getenv("FAIRVOTE_ANALYST_TOKEN_SHA256", "").split(",") if part.strip()
    ]
    if not configured_plain and not configured_hashes:
        return (
            False,
            "Full response export is disabled because FAIRVOTE_ANALYST_TOKEN or FAIRVOTE_ANALYST_TOKEN_SHA256 is not set.",
        )
    prefix = "Bearer "
    if not auth_header.startswith(prefix):
        return False, "Missing Authorization: Bearer token header."
    supplied = auth_header[len(prefix) :].strip()
    if not supplied:
        return False, "Missing bearer token."
    for token in configured_plain:
        if hmac.compare_digest(supplied, token):
            return True, "ok"
    supplied_hash = hashlib.sha256(supplied.encode("utf-8")).hexdigest()
    for token_hash in configured_hashes:
        if hmac.compare_digest(supplied_hash, token_hash):
            return True, "ok"
    return False, "Invalid analyst token."


def demographic_cell_key(record: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    demographics = record.get("demographics", {})
    if not isinstance(demographics, dict):
        demographics = {}
    return tuple(sorted((str(k), str(v)) for k, v in demographics.items()))


def privacy_report(records: list[dict[str, Any]], *, k_anonymity: int = DEFAULT_K_ANONYMITY) -> dict[str, Any]:
    """Summarise individual-record re-identification risk before export."""
    cell_counts: dict[tuple[tuple[str, str], ...], int] = {}
    for rec in records:
        key = demographic_cell_key(rec)
        cell_counts[key] = cell_counts.get(key, 0) + 1
    rare = [count for count in cell_counts.values() if count < int(k_anonymity)]
    return {
        "total_records": len(records),
        "demographic_cells": len(cell_counts),
        "k_anonymity_threshold": int(k_anonymity),
        "rare_cell_count": len(rare),
        "minimum_cell_size": min(cell_counts.values()) if cell_counts else 0,
        "individual_export_enabled": configured_export_enabled(),
        "timestamp_precision": os.getenv("FAIRVOTE_TIMESTAMP_PRECISION", DEFAULT_TIMESTAMP_PRECISION).lower(),
    }


def rare_cell_export_allowed() -> bool:
    return os.getenv("FAIRVOTE_ALLOW_RISKY_INDIVIDUAL_EXPORT", "").lower() in {"1", "true", "yes", "on"}


def exportable_individual_records(
    records: list[dict[str, Any]], *, k_anonymity: int = DEFAULT_K_ANONYMITY
) -> tuple[list[dict[str, Any]], dict[str, Any], str | None]:
    """Return individual records only when the privacy report passes k-anonymity.

    The endpoint is for controlled research audits, not normal analysis.  Normal
    analysis should use aggregate endpoints.  To avoid accidental rare-cell
    disclosure, the default policy refuses exports when any demographic cell has
    fewer than ``k_anonymity`` records.  A development-only override is available
    through FAIRVOTE_ALLOW_RISKY_INDIVIDUAL_EXPORT=1 and is surfaced in the
    returned report.
    """
    report = privacy_report(records, k_anonymity=k_anonymity)
    if report["rare_cell_count"] and not rare_cell_export_allowed():
        return (
            [],
            report,
            "Individual export blocked: demographic rare cells fail k-anonymity. Use /api/results or aggregate exports.",
        )
    clean: list[dict[str, Any]] = []
    for rec in records:
        leaked = find_forbidden_fields(rec)
        if leaked:
            raise PayloadRejected(f"Refusing to export raw-answer field(s): {', '.join(leaked)}")
        clean.append(dict(rec))
    report = dict(report)
    report["risky_export_override"] = rare_cell_export_allowed()
    return clean, report, None

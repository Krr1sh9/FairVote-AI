"""Configuration validation for the respondent collection server."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = BASE_DIR / "poll_config.json"
DEFAULT_DATA_PATH = BASE_DIR / "data" / "responses.jsonl"

MAX_DEMOGRAPHIC_FIELDS = int(os.getenv("FAIRVOTE_MAX_DEMOGRAPHIC_FIELDS", "8"))
MAX_DEMOGRAPHIC_KEY_LENGTH = int(os.getenv("FAIRVOTE_MAX_DEMOGRAPHIC_KEY_LENGTH", "64"))
MAX_DEMOGRAPHIC_VALUE_LENGTH = int(os.getenv("FAIRVOTE_MAX_DEMOGRAPHIC_VALUE_LENGTH", "128"))
DEFAULT_MAX_CONTENT_LENGTH = int(os.getenv("FAIRVOTE_MAX_CONTENT_BYTES", "8192"))
DEFAULT_TIMESTAMP_PRECISION = os.getenv("FAIRVOTE_TIMESTAMP_PRECISION", "minute").lower()
DEFAULT_K_ANONYMITY = int(os.getenv("FAIRVOTE_EXPORT_K_ANONYMITY", "5"))


def _require_string(value: Any, *, field_name: str, max_len: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    if len(value) > max_len:
        raise ValueError(f"{field_name} exceeds maximum length {max_len}")
    return value


def normalise_demographic_fields(raw_fields: Any) -> list[dict[str, Any]]:
    """Validate poll-config demographic fields and return a normalised list."""
    if raw_fields is None:
        return []
    if not isinstance(raw_fields, list):
        raise ValueError("poll_config.json field 'demographic_fields' must be a list")
    if len(raw_fields) > MAX_DEMOGRAPHIC_FIELDS:
        raise ValueError(f"poll_config.json defines more than {MAX_DEMOGRAPHIC_FIELDS} demographic fields")

    seen: set[str] = set()
    normalised: list[dict[str, Any]] = []
    for idx, field in enumerate(raw_fields):
        if not isinstance(field, dict):
            raise ValueError(f"demographic_fields[{idx}] must be an object")
        name = _require_string(
            field.get("name"), field_name=f"demographic_fields[{idx}].name", max_len=MAX_DEMOGRAPHIC_KEY_LENGTH
        )
        if name in seen:
            raise ValueError(f"duplicate demographic field name: {name}")
        seen.add(name)

        label = field.get("label", name)
        if not isinstance(label, str) or not label:
            raise ValueError(f"demographic_fields[{idx}].label must be a non-empty string")
        if len(label) > MAX_DEMOGRAPHIC_VALUE_LENGTH:
            raise ValueError(f"demographic_fields[{idx}].label exceeds maximum length {MAX_DEMOGRAPHIC_VALUE_LENGTH}")

        options = field.get("options", [])
        if not isinstance(options, list) or not options:
            raise ValueError(f"demographic_fields[{idx}].options must be a non-empty list")
        clean_options: list[str] = []
        for opt_idx, opt in enumerate(options):
            clean_opt = _require_string(
                opt,
                field_name=f"demographic_fields[{idx}].options[{opt_idx}]",
                max_len=MAX_DEMOGRAPHIC_VALUE_LENGTH,
            )
            if clean_opt not in clean_options:
                clean_options.append(clean_opt)
        if not clean_options:
            raise ValueError(f"demographic_fields[{idx}].options must contain at least one option")

        normalised.append(
            {
                "name": name,
                "label": label,
                "options": clean_options,
                "required": bool(field.get("required", False)),
            }
        )
    return normalised


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate the public poll configuration."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if not isinstance(cfg, dict):
        raise ValueError("poll_config.json must contain a JSON object")
    if "question" not in cfg or not isinstance(cfg["question"], str) or not cfg["question"].strip():
        raise ValueError("poll_config.json must have a non-empty 'question' field.")
    if "options" not in cfg or not isinstance(cfg["options"], list) or len(cfg["options"]) < 2:
        raise ValueError("poll_config.json must have >= 2 'options'.")
    if any(not isinstance(opt, str) or not opt for opt in cfg["options"]):
        raise ValueError("poll_config.json 'options' must be non-empty strings.")
    if len(set(cfg["options"])) != len(cfg["options"]):
        raise ValueError("poll_config.json 'options' must be unique.")
    if "epsilon" not in cfg or float(cfg["epsilon"]) <= 0:
        raise ValueError("poll_config.json must have a positive 'epsilon'.")

    cfg = dict(cfg)
    cfg["demographic_fields"] = normalise_demographic_fields(cfg.get("demographic_fields", []))
    return cfg

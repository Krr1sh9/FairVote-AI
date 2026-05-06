"""Parsing and lightweight validation helpers for dashboard uploads.

These functions intentionally do not import Streamlit, so they can be tested
without launching the dashboard.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Optional, Sequence


@dataclass(frozen=True)
class JsonlParseReport:
    """Line-level JSONL parse audit for evidence/reproducibility runs."""

    rows: list[dict[str, str]]
    invalid_lines: list[int]
    non_object_lines: list[int]

    @property
    def rejected_count(self) -> int:
        return len(self.invalid_lines) + len(self.non_object_lines)


def decode_upload_bytes(raw: bytes) -> str:
    """Decode uploaded text using UTF-8 with Latin-1 fallback."""

    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def read_csv_bytes(raw: bytes) -> list[dict[str, str]]:
    """Read CSV bytes into string-valued row dictionaries."""

    text = decode_upload_bytes(raw)
    reader = csv.DictReader(io.StringIO(text))
    return [{str(k): (v if v is not None else "") for k, v in r.items()} for r in reader]


def read_jsonl_bytes(raw: bytes, *, strict: bool = True) -> list[dict[str, str]]:
    """Read respondent JSONL and flatten demographics for dashboard use.

    Strict parsing is now the default because silent row loss undermines final
    evidence.  Exploratory dashboard sessions may pass ``strict=False`` to keep
    partial valid rows, but reproducibility/evidence commands should not.
    """

    return read_jsonl_bytes_with_report(raw, strict=strict).rows


def read_jsonl_bytes_with_report(raw: bytes, *, strict: bool = True) -> JsonlParseReport:
    """Read JSONL and return accepted rows plus rejected line numbers."""

    text = decode_upload_bytes(raw)
    rows: list[dict[str, str]] = []
    invalid_lines: list[int] = []
    non_object_lines: list[int] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            invalid_lines.append(line_no)
            if strict:
                raise ValueError(f"invalid JSONL on line {line_no}: {exc}") from exc
            continue
        if not isinstance(record, dict):
            non_object_lines.append(line_no)
            if strict:
                raise ValueError(f"JSONL line {line_no} is not an object")
            continue
        flat: dict[str, str] = {}
        for k, v in record.items():
            if k == "demographics" and isinstance(v, dict):
                for dk, dv in v.items():
                    flat[str(dk)] = str(dv)
            else:
                flat[str(k)] = str(v)
        rows.append(flat)
    return JsonlParseReport(rows=rows, invalid_lines=invalid_lines, non_object_lines=non_object_lines)


def read_uploaded_csv(uploaded_file: Any) -> list[dict[str, str]]:
    """Read a Streamlit uploaded CSV-like object."""

    return read_csv_bytes(uploaded_file.getvalue())


def read_uploaded_jsonl(uploaded_file: Any) -> list[dict[str, str]]:
    """Read a Streamlit uploaded JSONL-like object."""

    return read_jsonl_bytes(uploaded_file.getvalue())


def columns(rows: list[dict[str, str]]) -> list[str]:
    if not rows:
        return []
    return list(rows[0].keys())


def find_best_col(cols: list[str], candidates: Sequence[str]) -> int:
    """Return index of the first exact/substr candidate match, or 0."""

    lower = [c.lower() for c in cols]
    for cand in candidates:
        cand_l = cand.lower()
        for i, c in enumerate(lower):
            if c == cand_l:
                return i
    for cand in candidates:
        cand_l = cand.lower()
        for i, c in enumerate(lower):
            if cand_l in c:
                return i
    return 0


def valid_multiselect_defaults(defaults: Sequence[str], options: Sequence[str]) -> list[str]:
    """Return only defaults accepted by Streamlit for the current options."""

    option_set = set(options)
    return [value for value in defaults if value in option_set]


def load_poll_option_labels(root: Path) -> list[str]:
    """Load respondent poll option labels for display-only dashboard labelling."""

    config_path = root / "respondent" / "poll_config.json"
    if not config_path.exists():
        return []

    try:
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception:
        return []

    options = config.get("options")
    if not isinstance(options, list):
        return []

    labels = [str(opt).strip() for opt in options]
    return [lab for lab in labels if lab]


def category_index_from_value(value: Any) -> Optional[int]:
    """Parse a dashboard category value as a non-negative integer index, if safe."""

    if value is None or isinstance(value, bool):
        return None

    try:
        import numpy as np

        if isinstance(value, np.bool_):
            return None
        integer_types = (int, np.integer)
        float_types = (float, np.floating)
    except Exception:
        integer_types = (int,)
        float_types = (float,)

    if isinstance(value, integer_types):
        idx = int(value)
        return idx if idx >= 0 else None

    if isinstance(value, float_types):
        import math

        if not math.isfinite(float(value)) or not float(value).is_integer():
            return None
        idx = int(value)
        return idx if idx >= 0 else None

    raw = str(value).strip()
    if raw == "":
        return None

    try:
        as_float = float(raw)
    except Exception:
        return None

    import math

    if not math.isfinite(as_float) or not as_float.is_integer():
        return None

    idx = int(as_float)
    return idx if idx >= 0 else None


def is_numeric_category_value(value: Any) -> bool:
    return category_index_from_value(value) is not None

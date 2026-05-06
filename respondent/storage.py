"""Append-only JSONL storage for respondent records."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from .privacy import PayloadRejected, find_forbidden_fields


class ResponseStore:
    """Thread-safe append-only storage for perturbed responses."""

    def __init__(self, path: Path):
        self._path = Path(path)
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict[str, Any]) -> None:
        """Append a single sanitized JSON record to the JSONL file."""
        leaked = find_forbidden_fields(record)
        if leaked:
            raise PayloadRejected(f"Refusing to store raw-answer field(s): {', '.join(leaked)}")
        line = json.dumps(record, separators=(",", ":")) + "\n"
        with self._lock, open(self._path, "a", encoding="utf-8") as f:
            f.write(line)

    def read_all(self) -> list[dict[str, Any]]:
        """Read all stored records."""
        if not self._path.exists():
            return []
        records: list[dict[str, Any]] = []
        with open(self._path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def count(self) -> int:
        """Count total stored responses without loading all records."""
        if not self._path.exists():
            return 0
        n = 0
        with open(self._path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    n += 1
        return n

    @property
    def path(self) -> Path:
        return self._path

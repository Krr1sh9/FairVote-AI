"""Small reusable view helpers for the upload page."""
from __future__ import annotations

from collections.abc import Sequence


def bullet_list(items: Sequence[str]) -> str:
    """Return Markdown bullets for compact Streamlit help text."""
    return "\n".join(f"- {item}" for item in items)

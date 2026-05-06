"""Thin Streamlit upload page wrapper.

The implementation lives in :mod:`app.controllers.upload_controller`; this file
is intentionally small so the UI package exposes page entry points rather than a
large controller/service monolith.
"""

from __future__ import annotations

from pathlib import Path

from app.controllers.upload_controller import render_upload_tab as _render_upload_tab


def render_upload_tab(root: Path) -> None:
    """Render the upload-and-estimate page."""
    _render_upload_tab(root)


__all__ = ["render_upload_tab"]

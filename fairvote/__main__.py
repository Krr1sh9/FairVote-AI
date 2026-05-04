"""Module entry point for ``python -m fairvote``.

This file delegates to the CLI so command-line behaviour stays in one place.
"""

from __future__ import annotations

from fairvote.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

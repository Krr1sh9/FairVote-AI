# fairvote/main.py
"""
FairVote-AI main entrypoint (wrapper around the CLI).

You can run this via:
  python -m fairvote --help
"""

from __future__ import annotations

from fairvote.cli import main as _cli_main


def main() -> int:
    """Delegate to the CLI implementation used by package entry points."""
    return int(_cli_main())


if __name__ == "__main__":
    raise SystemExit(main())

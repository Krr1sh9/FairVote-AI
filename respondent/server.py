"""FairVote-AI respondent collection server.

The privacy boundary is intentionally narrow: respondents submit only the
Randomized-Response-perturbed answer plus configured demographic labels.  The
server hard-rejects raw-answer keys recursively before storage.  Individual
record export is disabled unless an analyst token is configured and, by default,
rare demographic cells satisfy the k-anonymity guard.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    # Support the documented command `python respondent/server.py` as well as
    # package execution.  When this file is run directly, Python sets
    # sys.path[0] to the respondent/ directory, so relative imports have no
    # package context and absolute `respondent.*` imports cannot see the
    # repository root unless we add it explicitly.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from respondent.app_factory import create_app
    from respondent.config import DEFAULT_CONFIG_PATH, DEFAULT_DATA_PATH, load_config, normalise_demographic_fields
    from respondent.privacy import (
        FORBIDDEN_RAW_ANSWER_KEYS,
        PayloadRejected,
        find_forbidden_fields,
        privacy_report,
        validate_demographics,
    )
    from respondent.storage import ResponseStore
else:
    from .app_factory import create_app
    from .config import DEFAULT_CONFIG_PATH, DEFAULT_DATA_PATH, load_config, normalise_demographic_fields
    from .privacy import (
        FORBIDDEN_RAW_ANSWER_KEYS,
        PayloadRejected,
        find_forbidden_fields,
        privacy_report,
        validate_demographics,
    )
    from .storage import ResponseStore

__all__ = [
    "FORBIDDEN_RAW_ANSWER_KEYS",
    "PayloadRejected",
    "ResponseStore",
    "create_app",
    "find_forbidden_fields",
    "load_config",
    "normalise_demographic_fields",
    "privacy_report",
    "validate_demographics",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FairVote-AI Respondent Server — serves the privacy-preserving poll client."
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Path to poll_config.json")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH), help="Path to responses.jsonl storage file")
    parser.add_argument("--port", type=int, default=5001, help="Port to run server on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode (development only)")
    args = parser.parse_args()

    app = create_app(config_path=Path(args.config), data_path=Path(args.data))

    if args.debug:
        print("\n" + "=" * 70)
        print("WARNING: Flask debug mode is ENABLED.")
        print("This is suitable for LOCAL DEVELOPMENT ONLY.")
        print("DO NOT use debug=True in production.")
        print("=" * 70 + "\n")

    print("=" * 70)
    print("FairVote-AI Respondent Server")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Data:   {args.data}")
    print(f"URL:    http://{args.host}:{args.port}")
    print("RR protocol: browser sends perturbed_answer only; raw-answer keys rejected recursively")
    print("Individual /api/responses export requires Bearer auth and passes k-anonymity by default")
    print("Press Ctrl+C to stop")
    print("=" * 70)

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":  # pragma: no cover
    main()

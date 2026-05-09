from __future__ import annotations

import os
from pathlib import Path

import pytest


def repo_root() -> Path:
    # Heuristic: tests/ lives at <repo>/tests
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def project_root() -> Path:
    return repo_root()


def slow_enabled() -> bool:
    # Slow integration tests (full experiment runs) are skipped in CI to keep
    # the feedback loop fast.  Set FV_RUN_SLOW=1 locally for full coverage.
    return os.environ.get("FV_RUN_SLOW", "").strip().lower() in {"1", "true", "yes", "y"}


def pytest_collection_modifyitems(config, items):
    if slow_enabled():
        return
    skip_slow = pytest.mark.skip(reason="slow test (set FV_RUN_SLOW=1 to enable)")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)

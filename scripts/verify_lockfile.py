"""Verify that the dependency lockfile is exact-pinned and usable as an examiner artefact.

This is deliberately a structural verifier, not an online resolver. It checks that
committed lock entries are exact pins and that every direct dependency declared
in pyproject.toml appears in the lock. A networked environment can regenerate the
lock with `make lock`.
"""
from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

PIN_RE = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s;#]+)(?:\s*;\s*.+)?$")
UNPINNED_TOKENS = (">=", "<=", "~=", "!=", "<", ">")


def _normalise_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirement_name(req: str) -> str:
    # Handles forms such as "pytest>=8,<9", "gunicorn>=21; marker", "foo[bar]>=1".
    req = req.split(";", 1)[0].strip()
    req = re.split(r"[<>=!~]", req, maxsplit=1)[0].strip()
    req = req.split("[", 1)[0].strip()
    return _normalise_name(req)


def parse_lock(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    errors: list[str] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("--"):
            continue
        match = PIN_RE.match(line)
        if not match:
            errors.append(f"{path}:{lineno}: not an exact 'name==version' pin: {raw}")
            continue
        name, version = match.groups()
        if any(token in version for token in UNPINNED_TOKENS):
            errors.append(f"{path}:{lineno}: version is not exact: {raw}")
        key = _normalise_name(name)
        if key in pins and pins[key] != version:
            errors.append(f"{path}:{lineno}: duplicate pin for {name!r} with conflicting versions")
        pins[key] = version
    if errors:
        raise SystemExit("\n".join(errors))
    return pins


def direct_dependency_names(pyproject: Path) -> set[str]:
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    project = data.get("project", {})
    names = {_requirement_name(req) for req in project.get("dependencies", [])}
    for reqs in project.get("optional-dependencies", {}).values():
        names.update(_requirement_name(req) for req in reqs)
    return {n for n in names if n}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", default="requirements.lock.txt", help="Lockfile path")
    parser.add_argument("--pyproject", default="pyproject.toml", help="pyproject.toml path")
    args = parser.parse_args(argv)

    lock = Path(args.lock)
    pyproject = Path(args.pyproject)
    if not lock.exists():
        raise SystemExit(f"Lockfile not found: {lock}")
    if not pyproject.exists():
        raise SystemExit(f"pyproject.toml not found: {pyproject}")

    pins = parse_lock(lock)
    missing = sorted(direct_dependency_names(pyproject) - set(pins))
    if missing:
        raise SystemExit("Direct dependencies missing from lockfile: " + ", ".join(missing))

    print(f"OK: {lock} contains {len(pins)} exact pins and covers all direct pyproject dependencies.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

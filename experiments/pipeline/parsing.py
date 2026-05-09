"""CLI parsing helpers for experiment scripts."""

from __future__ import annotations


def parse_list(s: str) -> list[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def parse_int_list(s: str) -> list[int]:
    vals = [int(x.strip()) for x in str(s).split(",") if x.strip()]
    if not vals:
        raise ValueError("Provide a comma-separated integer list, e.g. '500,1000,2500'.")
    if any(v <= 0 for v in vals):
        raise ValueError("All integer list values must be positive.")
    return vals


def parse_eps_list(s: str) -> list[float]:
    vals = [float(x.strip()) for x in str(s).split(",") if x.strip()]
    if not vals:
        raise ValueError("Provide --eps like '0.2,0.5,1.0'.")
    if any(e <= 0 for e in vals):
        raise ValueError("All epsilons must be > 0.")
    return vals


def parse_hidden_layers(s: str) -> tuple[int, ...]:
    raw = str(s).strip()
    if raw.lower() in {"", "none", "linear"}:
        return ()
    out: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        width = int(item)
        if width <= 0:
            raise ValueError("Neural hidden layer sizes must be positive integers.")
        out.append(width)
    if not out:
        raise ValueError("Provide --neural_hidden_layers like '32' or '64,32'.")
    return tuple(out)


def parse_multipliers(s: str) -> dict[str, float]:
    s = str(s).strip()
    if not s:
        return {}
    out: dict[str, float] = {}
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError("Use --multipliers like 'London=1.4,Wales=0.7'")
        name, val = item.split("=", 1)
        v = float(val.strip())
        if v <= 0:
            raise ValueError("Multipliers must be > 0.")
        out[name.strip()] = v
    return out

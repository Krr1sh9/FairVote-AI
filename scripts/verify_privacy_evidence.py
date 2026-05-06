"""Verify committed privacy-evidence artefacts exist and contain the expected proof points."""
from __future__ import annotations

from pathlib import Path

REQUIRED = {
    Path("evidence/privacy/browser_network_capture.md"): ["perturbed_answer", "selected_answer", "not present", "tests/test_browser_respondent_privacy.py"],
    Path("evidence/privacy/api_rejection_examples.md"): ["HTTP 400", "true_answer", "metadata.raw_answer", "not stored"],
    Path("evidence/privacy/privacy_report_example.json"): ["rare_cell_count", "timestamp_precision", "individual_export_enabled"],
}


def main() -> int:
    missing: list[str] = []
    for path, needles in REQUIRED.items():
        if not path.exists():
            missing.append(f"missing {path}")
            continue
        text = path.read_text(encoding="utf-8")
        for needle in needles:
            if needle not in text:
                missing.append(f"{path} does not contain required text: {needle!r}")
    if missing:
        raise SystemExit("Privacy evidence verification failed:\n- " + "\n- ".join(missing))
    print("OK: privacy evidence artefacts are present and readable.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

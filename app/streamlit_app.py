# app/streamlit_app.py
"""FairVote-AI Streamlit UI entrypoint.

The page-specific Streamlit code lives in ``app.ui``.  Pure dashboard logic
(parsing, estimation helpers, plotting, exports and run orchestration) lives in
``app.parsing``, ``app.services`` and ``app.plotting`` so it can be unit-tested
without launching Streamlit.

Run:
  pip install -e ".[dashboard]"
  # add ,[neural] if you want Neural RR-aware MRP in the dashboard
  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.ui.about import render_about_tab
from app.ui.recommendations import render_recommendations_tab
from app.ui.runs import render_runs_tab
from app.ui.scenario import render_scenario_tab
from app.ui.upload import render_upload_tab


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    st.set_page_config(page_title="FairVote-AI", layout="wide")
    root = project_root()
    outputs_dir = root / "experiments" / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    st.title("FairVote-AI")
    tabs = st.tabs(["Upload & Estimate", "Scenario Simulator", "Simulation & Runs", "Recommendations", "About"])

    with tabs[0]:
        render_upload_tab(root)
    with tabs[1]:
        render_scenario_tab(root)
    with tabs[2]:
        render_runs_tab(root, outputs_dir)
    with tabs[3]:
        render_recommendations_tab()
    with tabs[4]:
        render_about_tab()


if __name__ == "__main__":
    main()

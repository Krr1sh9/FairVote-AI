"""Experiment-run Streamlit page."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.services.runs import list_runs as _list_runs
from app.services.runs import read_csv_rows as _read_csv_rows
from app.services.runs import run_module as _run_module


def render_runs_tab(root: Path, outputs_dir: Path) -> None:
    st.subheader("Simulation & runs")
    st.write("Run the experiment pipeline and inspect outputs.")

    with st.sidebar:
        st.header("Simulation configuration")
        sim_trials = st.number_input("Trials", min_value=1, max_value=200, value=10, step=1, key="sim_trials")
        sim_eps = st.text_input("Eps list", value="0.2,0.5,1.0,2.0", key="sim_eps")
        sim_major_mass = st.number_input(
            "Major mass", min_value=0.0, max_value=1.0, value=0.02, step=0.01, key="sim_major_mass"
        )

    colA, colB = st.columns([1, 1])

    with colA:
        if st.button("Run mrp_vs_baselines", type="primary"):
            with st.spinner("Running mrp_vs_baselines..."):
                rc, out = _run_module(
                    "experiments.mrp_vs_baselines",
                    [
                        "--trials",
                        str(int(sim_trials)),
                        "--eps",
                        str(sim_eps),
                        "--major_mass",
                        str(float(sim_major_mass)),
                    ],
                    cwd=root,
                )
            st.text_area("Output", out, height=280)
            if rc == 0:
                st.success("Run completed.")
            else:
                st.error("Run failed. See output above.")

    with colB:
        st.write("Existing run folders:")
        runs = _list_runs(outputs_dir)
        choice = st.selectbox("Run folder", options=[p.name for p in runs] if runs else ["(none)"])
        if choice != "(none)":
            run_dir = outputs_dir / choice
            st.code(str(run_dir.as_posix()))
            summary_csv = run_dir / "summary.csv"
            if summary_csv.exists():
                st.dataframe(_read_csv_rows(summary_csv), use_container_width=True)
            else:
                st.info("No summary.csv found in this run folder.")

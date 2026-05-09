"""Optimisation/recommendation Streamlit page."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st


def render_recommendations_tab() -> None:
    st.subheader("Optimisation & Recommendations")
    st.markdown("Upload a `summary.csv` from an experiment run to find the optimal privacy-utility configuration that satisfies your requirements.")

    rec_csv = st.file_uploader("Upload summary.csv from an experiment run", type=["csv"], key="rec_csv")
    if rec_csv is None:
        return

    try:
        from fairvote.optimisation.recommend import Constraints, Objective, read_summary_csv, recommend_per_scenario

        raw_csv = rec_csv.getvalue()
        with tempfile.NamedTemporaryFile("wb", delete=False, suffix=".csv") as f:
            f.write(raw_csv)
            temp_path = Path(f.name)

        cands = read_summary_csv(temp_path)
        temp_path.unlink()

        if not cands:
            st.warning("No valid candidates found in CSV.")
            return

        st.success(f"Loaded {len(cands)} candidates across {len(set(c.scenario for c in cands))} scenarios.")

        colR1, colR2 = st.columns(2)
        with colR1:
            st.markdown("**Privacy Constraints**")
            eps_max = st.number_input("Max Epsilon (epsilon_max)", min_value=0.0, max_value=10.0, value=10.0, step=0.1)

            st.markdown("**Utility Constraints**")
            l1_max = st.number_input("Max Overall L1 Error", min_value=0.0, max_value=2.0, value=2.0, step=0.01)

        with colR2:
            st.markdown("**Fairness Constraints**")
            w_reg_l1 = st.number_input("Max Worst Region L1 (Major)", min_value=0.0, max_value=2.0, value=2.0, step=0.01)
            w_age_l1 = st.number_input("Max Worst Age L1 (Major)", min_value=0.0, max_value=2.0, value=2.0, step=0.01)

        obj_primary = st.selectbox("Optimize Objective (minimise)", ["mean_overall_l1", "mean_worst_region_l1_major", "mean_worst_age_l1_major", "mean_overall_mae", "epsilon"])

        if st.button("Generate Recommendations", type="primary"):
            cons = Constraints(
                epsilon_max=float(eps_max) if eps_max < 9.9 else None,
                overall_l1_max=float(l1_max) if l1_max < 1.99 else None,
                worst_region_l1_major_max=float(w_reg_l1) if w_reg_l1 < 1.99 else None,
                worst_age_l1_major_max=float(w_age_l1) if w_age_l1 < 1.99 else None,
            )
            objective = Objective(primary=obj_primary)

            recs = recommend_per_scenario(cands, constraints=cons, objective=objective)

            for rec in recs:
                st.markdown(f"### Scenario: `{rec.scenario}`")
                if rec.chosen:
                    st.success(f"**Recommended Method:** `{rec.chosen.method}` at **$\\epsilon={rec.chosen.epsilon}$**")
                    c_dict = {
                        "Metric": ["Overall L1 Error", "Worst Region L1 (Major)", "Worst Age L1 (Major)"],
                        "Value": [
                            f"{rec.chosen.mean_overall_l1:.4f} ± {rec.chosen.std_overall_l1:.4f}",
                            f"{rec.chosen.mean_worst_region_l1_major:.4f} ± {rec.chosen.std_worst_region_l1_major:.4f}",
                            f"{rec.chosen.mean_worst_age_l1_major:.4f} ± {rec.chosen.std_worst_age_l1_major:.4f}",
                        ],
                    }
                    st.table(c_dict)
                else:
                    st.error(f"No feasible configuration found. Reason: {rec.reason_if_none}")
                st.markdown(f"*(Feasible candidates: {rec.feasible_count} / {rec.total_count})*")
                st.divider()

    except Exception as e:
        st.error(f"Error parsing recommendations: {e}")

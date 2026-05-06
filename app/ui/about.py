"""About/demo-guidance Streamlit page."""

from __future__ import annotations

import streamlit as st


def render_about_tab() -> None:
    st.subheader("What to demo for marks")
    st.markdown(
        """
### Real-world demo (Upload & Estimate)
- Upload a poll dataset (RR-privatised responses)
- Show baseline RR-debias + group audit
- Upload population counts and show post-stratification
- Switch to RR-aware MRP and compare estimates (MRP may help under nonresponse / demographic skew, but is not guaranteed to improve results)
- Export the Results Bundle ZIP (plots + tables + markdown) and paste into your report

### Marks-bearing experiments
- Use `experiments.mrp_vs_baselines` for controlled trials across epsilons and bias scenarios
- Use the report table generator + recommendation scripts for the engineering decision story
"""
    )

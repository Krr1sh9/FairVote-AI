from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from fairvote.simulation.population import default_population
from fairvote.study import (
    METHOD_LABELS,
    METHODS,
    SUPPORTED_BIAS_LEVELS,
    SUPPORTED_EPSILONS,
    SUPPORTED_SAMPLE_SIZES,
    evaluate_poll,
)

RESEARCH_QUESTION = (
    "How do the Local Differential Privacy budget, polling sample size and demographic "
    "sampling bias affect the accuracy of estimates produced using k-ary Randomised "
    "Response, and when can simple cell-wise poststratification improve population-level "
    "estimates?"
)


def main() -> None:
    st.set_page_config(page_title="FairVote-AI", layout="centered")
    st.title("FairVote-AI")
    st.caption("Privacy-Preserving Polling with Randomised Response and Poststratification")
    st.write(f"Research question: {RESEARCH_QUESTION}")

    st.sidebar.header("Poll settings")
    epsilon = st.sidebar.select_slider(
        "Privacy budget (epsilon)",
        options=list(SUPPORTED_EPSILONS),
        value=1.0,
        help="Smaller epsilon gives stronger privacy and noisier reports.",
    )
    n_respondents = st.sidebar.select_slider(
        "Number of respondents",
        options=list(SUPPORTED_SAMPLE_SIZES),
        value=1000,
    )
    bias = st.sidebar.selectbox(
        "Sampling bias",
        options=list(SUPPORTED_BIAS_LEVELS),
        index=0,
        help="Controls how strongly younger demographic cells are over-sampled.",
    )
    seed = int(
        st.sidebar.number_input(
            "Random seed",
            min_value=0,
            max_value=10**6,
            value=0,
            step=1,
        )
    )

    population = default_population()
    result = evaluate_poll(
        population,
        n_respondents,
        epsilon,
        bias,
        np.random.default_rng(seed),
    )

    st.subheader("Selected settings")
    settings = pd.DataFrame(
        {
            "Setting": [
                "Epsilon",
                "Respondents",
                "Sampling bias",
                "Random seed",
            ],
            "Value": [
                f"{epsilon:g}",
                f"{n_respondents:,}",
                str(bias),
                str(seed),
            ],
        }
    )
    st.dataframe(settings, hide_index=True, width="stretch")

    st.subheader("Population truth and estimates")
    shares = pd.DataFrame(
        {
            "True population": result.truth,
            **{METHOD_LABELS[method]: result.estimates[method] for method in METHODS},
        },
        index=list(population.category_names),
    )
    st.bar_chart(shares)
    st.dataframe(shares.style.format("{:.3f}"), width="stretch")

    st.subheader("Error against the known truth")
    errors = pd.DataFrame(
        [
            {
                "Method": METHOD_LABELS[method],
                "L1 error": result.l1_errors[method],
                "Maximum absolute category error": result.max_abs_errors[method],
            }
            for method in METHODS
        ]
    ).set_index("Method")
    st.dataframe(errors.style.format("{:.4f}"), width="stretch")

    with st.expander("Technical diagnostics"):
        diagnostics = pd.DataFrame(
            {
                "Cell": population.cell_labels,
                "Sampled respondents": result.sample_cell_counts,
                "Population weight": population.weights,
                "Sample proportion": result.sample_cell_proportions,
            }
        )
        st.dataframe(
            diagnostics.style.format(
                {
                    "Population weight": "{:.3f}",
                    "Sample proportion": "{:.3f}",
                }
            ),
            hide_index=True,
            width="stretch",
        )
        st.write(f"Demographic imbalance (L1): {result.demographic_imbalance:.4f}")
        st.write(f"Empty-cell fallback count: {result.fallback_cells}")
        st.caption(
            f"The population contains {population.n_cells} cells: "
            f"{', '.join(population.cell_labels)}. Empty cells borrow the raw whole-sample "
            "RR correction before the final population estimate is projected."
        )

    with st.expander("Method summary"):
        st.write("Raw privatised reports are the uncorrected response frequencies.")
        st.write("Overall RR correction analytically inverts the Randomised Response channel.")
        st.write(
            "Poststratified RR estimate computes raw cell-wise RR inversions, weights "
            "them by known population shares, and projects only the final combined estimate."
        )


if __name__ == "__main__":
    main()

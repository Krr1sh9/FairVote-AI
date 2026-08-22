from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from fairvote.ai.features import (
    FEATURE_COLUMNS,
    extract_features,
    validate_selector_dataset,
)
from fairvote.ai.selector import (
    FittedSelector,
    export_tree_rules,
    recommend,
    train_selector,
)
from fairvote.simulation.population import default_population
from fairvote.study import (
    METHOD_LABELS,
    METHODS,
    SUPPORTED_BIAS_LEVELS,
    SUPPORTED_EPSILONS,
    SUPPORTED_SAMPLE_SIZES,
    StudyResult,
    evaluate_poll,
)

# The app reads the selector dataset from the default output location used by the recommender experiment pipeline.
AI_DATASET_PATH = Path("results") / "ai" / "selector_dataset.csv"
AI_GENERATION_COMMAND = "python -m experiments.train_selector"

# The recommender has only been evaluated within the fixed synthetic study.
AI_LIMITATION = (
    "The recommendation is learned from the fixed synthetic experiment grid. It has not been "
    "validated on real polling data or settings outside this simulation."
)

RESEARCH_QUESTION = (
    "How do the Local Differential Privacy budget, polling sample size and demographic "
    "sampling bias affect the accuracy of estimates produced using k-ary Randomised "
    "Response, and when can simple cell-wise poststratification improve population-level "
    "estimates?"
)


@st.cache_resource
def _load_selector(dataset_path: str, modified: float) -> FittedSelector:
    # modified is deliberately part of the function signature.
    # Passing the dataset modification time makes it part of Streamlit's cache key, so a different timestamp produces a different cache key.
    dataset = pd.read_csv(dataset_path)
    validate_selector_dataset(dataset)
    return train_selector(dataset)


def _render_recommendation(result: StudyResult, epsilon: float, n_respondents: int) -> None:
    st.subheader("AI-assisted estimator recommendation")

    # The statistical simulation remains usable even when the selector dataset has not been generated.
    if not AI_DATASET_PATH.is_file():
        st.write("The recommender dataset has not been generated yet, so no recommendation is available.")
        st.write("Create it with the following command and then reload this page.")
        st.code(AI_GENERATION_COMMAND, language="bash")
        return

    selector = _load_selector(str(AI_DATASET_PATH), AI_DATASET_PATH.stat().st_mtime)

    # Feature extraction uses the same feature definition as selector training.
    # Building the NumPy row in FEATURE_COLUMNS order keeps the prediction values aligned with the expected model feature order.
    features = extract_features(result, epsilon, n_respondents)
    feature_row = np.array([[features[column] for column in FEATURE_COLUMNS]], dtype=float)
    prediction = recommend(selector, feature_row)

    # The table shows the predicted L1 error for every estimator so the recommendation can be inspected directly.
    predicted = pd.DataFrame(
        [
            {
                "Method": METHOD_LABELS[method],
                "Predicted L1 error": prediction.predicted_errors[method],
            }
            for method in METHODS
        ]
    ).set_index("Method")
    st.dataframe(predicted.style.format("{:.4f}"), width="stretch")

    # An approximate tie is reported when the gap between the two lowest predicted errors is within the selector's tie threshold.
    if prediction.approximate_tie:
        tied = " and ".join(METHOD_LABELS[method] for method in prediction.tied_methods)
        st.write(f"Approximate tie between {tied}.")
        st.write(f"Lowest predicted error: {METHOD_LABELS[prediction.best_method]}.")
    else:
        st.write(f"Recommended estimator: {METHOD_LABELS[prediction.best_method]}.")

    st.write(f"Approximate-tie threshold: {prediction.tie_threshold:.4f}")
    st.caption(
        "Predictions use only the six observable inputs listed below. The sampling bias condition, "
        "the population truth and the measured errors are never given to the model."
    )
    st.caption(AI_LIMITATION)

    # The expander exposes the exact feature values used for the recommendation and the text rules for one fitted decision tree.
    with st.expander("Model inputs and decision rules"):
        inputs = pd.DataFrame(
            {
                "Feature": list(FEATURE_COLUMNS),
                "Value": [features[column] for column in FEATURE_COLUMNS],
            }
        )
        st.dataframe(inputs.style.format({"Value": "{:.4f}"}), hide_index=True, width="stretch")
        chosen = st.selectbox(
            "Decision tree to inspect",
            options=list(METHODS),
            format_func=lambda method: METHOD_LABELS[method],
        )
        st.code(export_tree_rules(selector, chosen), language="text")


def main() -> None:
    st.set_page_config(page_title="FairVote-AI", layout="centered")
    st.title("FairVote-AI")
    st.caption("Privacy-Preserving Polling with Randomised Response and Poststratification")
    st.write(f"Research question: {RESEARCH_QUESTION}")

    # These controls use the epsilon, sample-size and bias values supported by the shared study workflow.
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

    # The interactive seed controls the random generator for this single displayed synthetic poll.
    # The batch experiment initialises each generator from the repetition seed together with the epsilon, sample-size and bias indices.
    population = default_population()
    result = evaluate_poll(
        population,
        n_respondents,
        epsilon,
        bias,
        np.random.default_rng(seed),
    )

    # The selected settings are displayed so the shown output can be associated with the exact interactive configuration.
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

    # evaluate_poll returns the known synthetic population truth and all three estimator outputs for the same poll.
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

    # Error can be calculated directly because the synthetic population truth is fixed and known.
    # These measured errors are displayed for evaluation and are not passed to the recommender as input features.
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

    # The recommender derives its six model features from this StudyResult together with the selected epsilon and sample size.
    _render_recommendation(result, epsilon, n_respondents)

    # The diagnostics show the realised demographic composition and the number of cells that required fallback handling.
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
            "RR correction before clipping and renormalisation are applied to the "
            "final population estimate."
        )

    # The method summary uses the same estimator definitions as the shared study workflow.
    with st.expander("Method summary"):
        st.write("Raw privatised reports are the uncorrected response frequencies.")
        st.write("Overall RR correction analytically inverts the Randomised Response channel.")
        st.write(
            "Poststratified RR estimate computes unconstrained cell-wise RR inversions, weights "
            "them by known population shares, and applies clipping and renormalisation only to the "
            "final combined estimate."
        )


if __name__ == "__main__":
    main()

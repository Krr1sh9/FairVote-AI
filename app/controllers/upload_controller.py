"""Upload-and-estimate Streamlit page.

The UI remains here, while parsing, estimation, plotting, metrics and export
helpers are imported from testable modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st

from app.parsing.upload import (
    columns as _columns,
)
from app.parsing.upload import (
    find_best_col as _find_best_col,
)
from app.parsing.upload import (
    load_poll_option_labels as _load_poll_option_labels,
)
from app.parsing.upload import (
    read_uploaded_csv as _read_uploaded_csv,
)
from app.parsing.upload import (
    read_uploaded_jsonl as _read_uploaded_jsonl,
)
from app.parsing.upload import (
    valid_multiselect_defaults as _valid_multiselect_defaults,
)
from app.plotting.charts import (
    HAS_MPL as _HAS_MPL,
)
from app.plotting.charts import (
    plot_group_bars as _plot_group_bars,
)
from app.plotting.charts import (
    plot_overall_distributions as _plot_overall_distributions,
)
from app.services.category import (
    answer_like_columns as _answer_like_columns,
)
from app.services.category import (
    build_category_map,
    encode_categories,
    filter_valid,
    group_keys,
    poststratify_from_groups,
    read_population_weights,
)
from app.services.category import (
    category_display_map as _category_display_map,
)
from app.services.category import (
    display_labels_for_categories as _display_labels_for_categories,
)
from app.services.category import (
    format_group_key as _format_group_key,
)
from app.services.category import (
    normalised_mean_probability as _normalised_mean_probability,
)
from app.services.category import (
    parse_hidden_layers as _parse_hidden_layers,
)
from app.services.category import (
    poststratify_probabilities as _poststratify_probabilities,
)
from app.services.exports import (
    build_group_audit_csv,
    build_metadata_json,
    build_overall_estimates_csv,
    build_plot_zip,
    build_results_bundle,
    build_results_summary_markdown,
    now_string,
)
from app.services.inference import (
    _HAS_RR_MRP,
    DesignMatrix,
    MisreportRRMultinomialModel,
    MRPRRMultinomialModel,
    available_method_options,
    bootstrap_ci,
    estimate_distribution,
    resolve_estimation_method,
    shy_misreport_matrix,
)
from app.services.inference import (
    is_learned_mrp_method as _is_learned_mrp_method,
)
from app.services.inference import (
    load_neural_mrp_model as _load_neural_mrp_model,
)
from app.services.inference import (
    method_prefix as _method_prefix,
)
from app.services.inference import (
    method_short_label as _method_short_label,
)
from app.services.metrics import fmt as _fmt
from app.services.metrics import group_metric_summary as _group_metric_summary
from app.services.metrics import overall_metrics as _overall_metrics
from app.services.upload_analysis import candidate_truth_columns, validate_truth_column_policy


def render_upload_tab(root: Path) -> None:
    st.subheader("Upload a poll CSV and estimate under randomized response")

    if not _HAS_MPL:
        st.warning("matplotlib not installed. Install it to enable plots: pip install matplotlib")

    col1, col2 = st.columns([1, 1])
    with col1:
        poll_file = st.file_uploader("Poll CSV or JSONL (responses)", type=["csv", "jsonl"], key="poll_file")
    with col2:
        pop_file = st.file_uploader("Population CSV (optional, for post-stratification)", type=["csv"], key="pop_csv")

    poll_rows: list[dict[str, str]] = []
    pop_rows: list[dict[str, str]] = []

    is_jsonl_upload = False
    if poll_file is not None:
        try:
            if poll_file.name.endswith(".jsonl"):
                is_jsonl_upload = True
                poll_rows = _read_uploaded_jsonl(poll_file)
                st.success(f"Loaded poll JSONL: {len(poll_rows)} rows, {len(_columns(poll_rows))} columns")
            else:
                poll_rows = _read_uploaded_csv(poll_file)
                st.success(f"Loaded poll CSV: {len(poll_rows)} rows, {len(_columns(poll_rows))} columns")
        except Exception as e:
            st.error(f"Failed to read poll file: {e}")

    if pop_file is not None:
        try:
            pop_rows = _read_uploaded_csv(pop_file)
            st.success(f"Loaded population CSV: {len(pop_rows)} rows, {len(_columns(pop_rows))} columns")
        except Exception as e:
            st.error(f"Failed to read population CSV: {e}")

    if not poll_rows:
        st.info("Upload a poll CSV or JSONL to begin.")
    else:
        cols = _columns(poll_rows)

        # ---------- Sidebar config ----------
        with st.sidebar:
            st.header("Poll configuration")

            response_col = st.selectbox(
                "Reported choice column (privatised)",
                options=cols,
                index=_find_best_col(
                    cols, ["perturbed_answer", "reported_choice", "reported", "response", "vote", "choice"]
                ),
            )

            truth_like_cols = candidate_truth_columns(cols)
            synthetic_evaluation_mode = st.checkbox(
                "Synthetic evaluation mode: allow a true-choice column",
                value=False,
                help=(
                    "Enable only for generated/synthetic experiment CSVs. Real respondent exports should not "
                    "contain true choices; leaving this off prevents accidental misuse of private/raw labels."
                ),
            )
            if truth_like_cols and not synthetic_evaluation_mode:
                st.warning(
                    "A truth-like column was detected, but it is ignored because synthetic evaluation mode is off."
                )
            truth_col_raw = st.selectbox(
                "Optional: true choice column (synthetic evaluation only)",
                options=["(none)"] + cols,
                index=(1 + [c.lower() for c in cols].index("true_choice"))
                if synthetic_evaluation_mode and any(c.lower() == "true_choice" for c in cols)
                else 0,
                disabled=not synthetic_evaluation_mode,
            )
            try:
                truth_col = validate_truth_column_policy(
                    truth_col=None if truth_col_raw == "(none)" else truth_col_raw,
                    synthetic_evaluation_mode=bool(synthetic_evaluation_mode),
                )
            except ValueError as exc:
                st.error(str(exc))
                truth_col = None

            st.divider()

            st.subheader("Method")
            method = st.radio(
                "Estimation method",
                options=available_method_options(),
                index=0,
                help=(
                    "RR debiasing estimates aggregate shares directly. Learned MRP methods fit "
                    "P(true choice | demographics) while accounting for the RR observation process."
                ),
            )
            method, method_warning = resolve_estimation_method(method)
            if method_warning:
                st.warning(method_warning)
            if method == "Neural RR-aware MRP":
                st.warning(
                    "Neural MRP is a learned model. Compare it against RR debiasing and linear MRP; "
                    "it can overfit or underperform on small/noisy privatized samples."
                )

            st.divider()

            # epsilon default: if column exists, try first non-empty value
            eps_default = 1.0
            if any(c.lower() == "epsilon" for c in cols):
                for r in poll_rows:
                    v = str(r.get("epsilon", "")).strip()
                    if v:
                        try:
                            eps_default = float(v)
                            break
                        except Exception:
                            pass

            st.subheader("Privacy")
            epsilon = st.number_input("epsilon", min_value=0.01, max_value=10.0, value=float(eps_default), step=0.05)

            st.divider()

            st.subheader("Categories (k)")
            raw_labels = [r.get(response_col, "") for r in poll_rows]
            uniq_labels = sorted({str(v).strip() for v in raw_labels if str(v).strip() != ""})
            st.caption(f"Unique labels in reported column: {len(uniq_labels)}")
            if len(uniq_labels) > 50 and len(uniq_labels) > 0.2 * len(poll_rows):
                st.warning(
                    "Selected response column has very high cardinality. This often means you picked an ID column."
                )
            k_override_val = st.number_input("Override k (optional)", min_value=0, max_value=200, value=0, step=1)
            k_override = None if int(k_override_val) <= 0 else int(k_override_val)
            sidebar_cmap = build_category_map(raw_labels, k_override=k_override)
            sidebar_display_labels, _ = _display_labels_for_categories(
                sidebar_cmap,
                option_labels=_load_poll_option_labels(root),
                use_poll_config=True,
            )

            st.divider()

            st.subheader("Audit settings")
            group_options = [c for c in cols if c != response_col]
            default_groups = [c for c in ["region", "age_band"] if c in group_options]
            group_cols = st.multiselect(
                "Group columns for auditing (e.g., region, age_band)",
                options=group_options,
                default=_valid_multiselect_defaults(default_groups, group_options),
            )
            major_mass = st.number_input(
                "Major group mass threshold", min_value=0.0, max_value=1.0, value=0.02, step=0.01
            )
            n_boot = st.number_input(
                "Bootstrap resamples (baseline only; 0 disables)", min_value=0, max_value=5000, value=300, step=50
            )
            boot_seed = st.number_input("Bootstrap seed", min_value=0, max_value=10_000_000, value=123, step=1)

            st.divider()

            st.subheader("Post-stratification (optional)")
            if pop_rows:
                pop_cols = _columns(pop_rows)
                post_options = [c for c in pop_cols if c in cols]
                default_post = [c for c in group_cols if c in pop_cols]
                post_cols = st.multiselect(
                    "Post-strat key columns (must exist in BOTH poll and population CSV)",
                    options=post_options,
                    default=_valid_multiselect_defaults(default_post, post_options),
                )
                count_col = st.selectbox(
                    "Population count column",
                    options=pop_cols,
                    index=_find_best_col(pop_cols, ["count", "n", "pop", "population"]),
                )
            else:
                post_cols = []
                count_col = None

            if _is_learned_mrp_method(method):
                st.divider()
                st.subheader("Learned MRP settings")
                answer_like_cols = _answer_like_columns(response_col, truth_col)
                mrp_feature_options = [c for c in cols if c.strip().lower() not in answer_like_cols]
                mrp_default_cols = group_cols if group_cols else default_groups
                mrp_feature_cols = st.multiselect(
                    "Feature columns (categorical demographics)",
                    options=mrp_feature_options,
                    default=_valid_multiselect_defaults(mrp_default_cols, mrp_feature_options),
                    help=(
                        "Learned MRP predicts latent true choice from these features, then trains through "
                        "the randomized-response observation model using only reported labels."
                    ),
                )
                mrp_lr = st.number_input(
                    "Learning rate",
                    min_value=0.0001,
                    max_value=1.0,
                    value=0.02 if method == "Neural RR-aware MRP" else 0.05,
                    step=0.005,
                )
                mrp_steps = st.number_input(
                    "Training steps",
                    min_value=25,
                    max_value=20000,
                    value=500 if method == "Neural RR-aware MRP" else 2000,
                    step=25,
                )
                mrp_batch = st.number_input("Batch size", min_value=16, max_value=8192, value=512, step=64)
                mrp_seed = st.number_input("Model seed", min_value=0, max_value=10_000_000, value=0, step=1)

                if method == "Linear RR-aware MRP":
                    mrp_l2 = st.number_input("L2 regularization", min_value=0.0, max_value=100.0, value=1.0, step=0.5)

                if method == "Misreport-aware RR-MRP":
                    mrp_l2 = st.number_input("L2 regularization", min_value=0.0, max_value=100.0, value=1.0, step=0.5)
                    shy_category_options = list(range(len(sidebar_cmap.labels))) if sidebar_cmap.labels else [0]

                    def _format_shy_category(idx: int) -> str:
                        raw = sidebar_cmap.labels[idx] if 0 <= idx < len(sidebar_cmap.labels) else str(idx)
                        disp = sidebar_display_labels[idx] if 0 <= idx < len(sidebar_display_labels) else raw
                        return disp if disp == raw else f"{disp} ({raw})"

                    misreport_shy_category = st.selectbox(
                        "Shy/misreported category",
                        options=shy_category_options,
                        index=min(1, len(shy_category_options) - 1),
                        format_func=_format_shy_category,
                    )
                    misreport_honesty = st.slider(
                        "Honesty for that category before RR",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.8,
                        step=0.01,
                        help="Only this simple shy-voter misreport model is exposed in the dashboard.",
                    )

                if method == "Neural RR-aware MRP":
                    neural_size = st.selectbox(
                        "Neural model size",
                        options=["Small: 16", "Medium: 32,16", "Custom"],
                        index=0,
                        help="Keep this small unless you have enough respondents. Larger networks can overfit RR noise.",
                    )
                    if neural_size == "Small: 16":
                        neural_hidden_layers_text = "16"
                    elif neural_size == "Medium: 32,16":
                        neural_hidden_layers_text = "32,16"
                    else:
                        neural_hidden_layers_text = st.text_input(
                            "Hidden layer sizes", value="32,16", help="Comma-separated widths, e.g. 64,32"
                        )
                    neural_dropout = st.slider("Dropout", min_value=0.0, max_value=0.8, value=0.0, step=0.05)
                    neural_weight_decay = st.number_input(
                        "Weight decay", min_value=0.0, max_value=10.0, value=0.0001, step=0.0001, format="%.5f"
                    )

        # ---------- Build category map ----------
        raw_labels = [r.get(response_col, "") for r in poll_rows]
        cmap = build_category_map(raw_labels, k_override=k_override)
        k = len(cmap.labels)
        option_labels = _load_poll_option_labels(root)
        display_labels, using_poll_option_labels = _display_labels_for_categories(
            cmap,
            option_labels=option_labels,
            use_poll_config=True,
        )
        category_display_map = _category_display_map(cmap, display_labels, option_labels)
        display_answer_like_cols = _answer_like_columns(response_col, truth_col)

        reported_raw = [r.get(response_col, "") for r in poll_rows]
        reported = encode_categories(reported_raw, cmap)

        truth = None
        if truth_col is not None:
            truth_raw = [r.get(truth_col, "") for r in poll_rows]
            truth = encode_categories(truth_raw, cmap)

        reported, truth, valid_mask = filter_valid(reported, truth)
        n = int(reported.size)

        if n == 0:
            st.error("No valid rows after category mapping. Check your reported choice column.")
        else:
            valid_indices = np.where(valid_mask)[0].tolist()
            poll_rows_valid = [poll_rows[i] for i in valid_indices]

            # ---------- Overall baseline estimate ----------
            # Validate k (number of categories)
            try:
                k_int = int(k)
            except Exception:
                k_int = 0
            if k_int < 2:
                st.error(
                    "k must be >= 2. Your current settings/data imply only 0–1 categories. "
                    "Pick the correct 'Reported choice' column (not respondent_id), or set 'Override k' to 2+."
                )
                st.stop()
            k = k_int

            p_baseline = estimate_distribution(reported, epsilon=float(epsilon), k=int(k))

            # Truth distribution (if available)
            p_true = None
            if truth is not None and truth.size == n:
                truth_arr = np.asarray(truth, dtype=int).reshape(-1)
                bad_truth = int(np.sum(truth_arr < 0))
                if bad_truth > 0:
                    st.warning(
                        f"Truth column contains {bad_truth} unmapped/invalid values (likely wrong reported-choice column "
                        f"or label mismatch). Skipping truth-based accuracy metrics."
                    )
                else:
                    p_true = np.bincount(truth_arr, minlength=k).astype(float)
                    denom = float(p_true.sum()) if float(p_true.sum()) > 0 else 1.0
                    p_true = p_true / denom
            # Bootstrap CI (baseline only)
            p_lo, p_hi = (None, None)
            if int(n_boot) > 0:
                p_lo, p_hi = bootstrap_ci(
                    reported, epsilon=float(epsilon), k=int(k), n_boot=int(n_boot), seed=int(boot_seed)
                )

            st.subheader("Overall estimate")
            if using_poll_option_labels and is_jsonl_upload:
                st.caption(
                    "Real respondent exports store randomized category indices; "
                    "the dashboard maps them to poll option labels for display only. "
                    "Underlying data and calculations remain numeric."
                )
            table_rows = []
            for i, lab in enumerate(display_labels):
                row = {"category_id": i, "label": lab, "rr_debias_p": float(p_baseline[i])}
                if p_lo is not None and p_hi is not None:
                    row["ci_low"] = float(p_lo[i])
                    row["ci_high"] = float(p_hi[i])
                if p_true is not None:
                    row["true_p"] = float(p_true[i])
                table_rows.append(row)
            st.dataframe(table_rows, use_container_width=True)

            # ---------- Group audit (baseline) ----------
            group_rows: list[dict[str, Any]] = []
            group_estimates: dict[tuple[str, ...], np.ndarray] = {}
            group_true: dict[tuple[str, ...], np.ndarray] = {}

            if group_cols:
                st.subheader("Group audit (baseline)")

                group_to_idx: dict[tuple[str, ...], list[int]] = {}
                for pos, original_i in enumerate(valid_indices):
                    key = group_keys(poll_rows[int(original_i)], group_cols)
                    group_to_idx.setdefault(key, []).append(pos)

                for g, idxs in group_to_idx.items():
                    idx_arr = np.asarray(idxs, dtype=int)
                    rep_g = reported[idx_arr]
                    p_g = estimate_distribution(rep_g, epsilon=float(epsilon), k=int(k))
                    group_estimates[g] = p_g

                    mass = float(rep_g.size) / float(n)
                    major = mass >= float(major_mass)

                    # If truth exists, compute group truth distribution and L1 error
                    l1_err = float("nan")
                    if p_true is not None and truth is not None:
                        tru_g = truth[idx_arr]
                        p_true_g = np.bincount(tru_g, minlength=k).astype(float) / max(1.0, float(tru_g.size))
                        group_true[g] = p_true_g
                        l1_err = float(np.sum(np.abs(p_g - p_true_g)))

                    key_str = _format_group_key(g, group_cols, category_display_map, display_answer_like_cols)
                    group_rows.append(
                        {
                            "group": key_str,
                            "n": int(rep_g.size),
                            "mass": mass,
                            "major": bool(major),
                            "baseline_l1": l1_err,
                        }
                    )

                group_rows.sort(key=lambda r: float(r.get("mass", 0.0)), reverse=True)
                st.dataframe(group_rows, use_container_width=True)

            # ---------- Direct post-strat (baseline) ----------
            p_post_direct = None
            pop_weights = None
            if pop_rows and post_cols and count_col is not None:
                pop_weights = read_population_weights(pop_rows, post_cols, str(count_col))
                if pop_weights:
                    # Need estimates per post-strat group keys:
                    if list(post_cols) == list(group_cols) and group_estimates:
                        post_est = group_estimates
                    else:
                        post_to_idx: dict[tuple[str, ...], list[int]] = {}
                        for pos, original_i in enumerate(valid_indices):
                            key = group_keys(poll_rows[int(original_i)], post_cols)
                            post_to_idx.setdefault(key, []).append(pos)

                        post_est: dict[tuple[str, ...], np.ndarray] = {}
                        for g, idxs in post_to_idx.items():
                            rep_g = reported[np.asarray(idxs, dtype=int)]
                            post_est[g] = estimate_distribution(rep_g, epsilon=float(epsilon), k=int(k))

                    p_post_direct = poststratify_from_groups(post_est, pop_weights, fallback=p_baseline)

                    st.subheader("Post-stratified estimate (direct baseline)")
                    post_rows = [
                        {"category_id": i, "label": lab, "poststrat_p": float(p_post_direct[i])}
                        for i, lab in enumerate(display_labels)
                    ]
                    st.dataframe(post_rows, use_container_width=True)

            # ---------- Learned RR-aware MRP methods (optional) ----------
            p_mrp_post = None
            p_mrp_sample = None
            group_rows_mrp: list[dict[str, Any]] | None = None  # learned-model fairness comparisons
            learned_method_label = _method_short_label(method)
            learned_prefix = _method_prefix(method)
            learned_l1_key = f"{learned_prefix}_l1"
            learned_sample_col = f"{learned_prefix}_sample_p"
            learned_post_col = f"{learned_prefix}_poststrat_p"

            if _is_learned_mrp_method(method):
                st.subheader(learned_method_label)
                if not _HAS_RR_MRP or DesignMatrix is None:
                    st.warning("Design-matrix support is unavailable, so learned MRP cannot run.")
                elif "mrp_feature_cols" not in locals() or not mrp_feature_cols:
                    st.error("Select at least one feature column for learned MRP in the sidebar.")
                else:
                    design = DesignMatrix(mrp_feature_cols).fit(poll_rows_valid)
                    X = design.transform(poll_rows_valid)
                    P_true = None

                    try:
                        if method == "Linear RR-aware MRP":
                            model = MRPRRMultinomialModel(
                                k=int(k), epsilon=float(epsilon), l2=float(mrp_l2), seed=int(mrp_seed)
                            )
                            with st.spinner("Fitting linear RR-aware MRP model..."):
                                info = model.fit(
                                    X,
                                    reported,
                                    lr=float(mrp_lr),
                                    steps=int(mrp_steps),
                                    batch_size=int(mrp_batch),
                                    verbose_every=0,
                                    keep_history=False,
                                )
                            st.write(
                                f"Fitted linear RR-aware MRP: steps={info.steps}, final_loss={_fmt(info.final_loss, 6)}"
                            )
                            P_true = model.predict_true_proba(X)

                        elif method == "Neural RR-aware MRP":
                            hidden_layers = _parse_hidden_layers(neural_hidden_layers_text)
                            RRNeuralMRPModel = _load_neural_mrp_model()
                            model = RRNeuralMRPModel(
                                k=int(k),
                                epsilon=float(epsilon),
                                hidden_layers=hidden_layers,
                                dropout=float(neural_dropout),
                                weight_decay=float(neural_weight_decay),
                                seed=int(mrp_seed),
                            )
                            with st.spinner("Fitting neural RR-aware MRP model..."):
                                info = model.fit(
                                    X,
                                    reported,
                                    lr=float(mrp_lr),
                                    steps=int(mrp_steps),
                                    batch_size=int(mrp_batch),
                                    keep_history=False,
                                    verbose_every=0,
                                )
                            st.write(
                                f"Fitted neural RR-aware MRP: hidden_layers={hidden_layers}, "
                                f"steps={info.steps}, final_loss={_fmt(info.final_loss, 6)}"
                            )
                            P_true = model.predict_true_proba(X)

                        elif method == "Misreport-aware RR-MRP":
                            shy_category = int(misreport_shy_category)
                            M = shy_misreport_matrix(
                                int(k), shy_category=shy_category, honesty=float(misreport_honesty)
                            )
                            model = MisreportRRMultinomialModel(
                                k=int(k), l2=float(mrp_l2), seed=int(mrp_seed), misreport=M
                            )
                            with st.spinner("Fitting misreport-aware RR-MRP model..."):
                                model.fit(
                                    X,
                                    reported,
                                    eps=float(epsilon),
                                    lr=float(mrp_lr),
                                    steps=int(mrp_steps),
                                    batch_size=int(mrp_batch),
                                    verbose_every=0,
                                )
                            st.write(
                                f"Fitted misreport-aware RR-MRP: shy_category={display_labels[shy_category] if 0 <= shy_category < len(display_labels) else cmap.labels[shy_category]}, "
                                f"honesty={float(misreport_honesty):.2f}, steps={int(mrp_steps)}"
                            )
                            P_true = model.predict_theta(X)

                    except Exception as e:
                        st.error(f"{learned_method_label} failed to fit: {e}")
                        P_true = None

                    if P_true is not None:
                        # Sample-averaged latent true probabilities; this works without a population file.
                        p_mrp_sample = _normalised_mean_probability(P_true)

                        st.caption(f"{learned_method_label} sample-averaged estimate (not post-stratified):")
                        st.dataframe(
                            [
                                {"category_id": i, "label": lab, learned_sample_col: float(p_mrp_sample[i])}
                                for i, lab in enumerate(display_labels)
                            ],
                            use_container_width=True,
                        )

                        # Learned MRP post-strat requires population file and matching keys.
                        if pop_rows and post_cols and count_col is not None:
                            if sorted(post_cols) != sorted(mrp_feature_cols):
                                st.warning(
                                    "For learned MRP post-stratification, set post-strat key columns to match the feature columns.\n\n"
                                    f"MRP features: {mrp_feature_cols}\nPost-strat keys: {post_cols}"
                                )
                            else:
                                pop_cells: list[dict[str, str]] = []
                                pop_counts: list[float] = []
                                for r in pop_rows:
                                    try:
                                        cval = float(r.get(str(count_col), "nan"))
                                    except Exception:
                                        cval = float("nan")
                                    if not np.isfinite(cval) or cval <= 0.0:
                                        continue
                                    pop_cells.append({c: str(r.get(c, "")).strip() for c in mrp_feature_cols})
                                    pop_counts.append(float(cval))

                                if pop_cells:
                                    X_pop = design.transform(pop_cells)
                                    w = np.asarray(pop_counts, dtype=float)
                                    if hasattr(model, "poststratify"):
                                        p_mrp_post = model.poststratify(X_pop, w)
                                    elif hasattr(model, "predict_theta"):
                                        p_mrp_post = _poststratify_probabilities(model.predict_theta(X_pop), w)
                                    else:
                                        raise RuntimeError("learned model does not support poststratification")
                                    st.subheader(f"{learned_method_label} post-stratified estimate")
                                    st.dataframe(
                                        [
                                            {"category_id": i, "label": lab, learned_post_col: float(p_mrp_post[i])}
                                            for i, lab in enumerate(display_labels)
                                        ],
                                        use_container_width=True,
                                    )
                                else:
                                    st.error("Population CSV yielded no valid rows for the chosen count/key columns.")

                        # Group-level learned-model predictions for the fairness dashboard.
                        if group_cols:
                            group_rows_mrp = []
                            group_to_idx: dict[tuple[str, ...], list[int]] = {}
                            for pos, original_i in enumerate(valid_indices):
                                key = group_keys(poll_rows[int(original_i)], group_cols)
                                group_to_idx.setdefault(key, []).append(pos)

                            for g, idxs in group_to_idx.items():
                                idx_arr = np.asarray(idxs, dtype=int)
                                p_g_mrp = _normalised_mean_probability(P_true[idx_arr])
                                mass = float(idx_arr.size) / float(n)
                                major = mass >= float(major_mass)

                                # In real polling mode, do not require true labels. Use divergence from model overall as a proxy.
                                if p_true is not None and truth is not None:
                                    tru_g = truth[idx_arr]
                                    p_true_g = np.bincount(tru_g, minlength=k).astype(float) / max(
                                        1.0, float(tru_g.size)
                                    )
                                    learned_l1 = float(np.sum(np.abs(p_g_mrp - p_true_g)))
                                else:
                                    learned_l1 = float(np.sum(np.abs(p_g_mrp - p_mrp_sample)))

                                key_str = _format_group_key(
                                    g, group_cols, category_display_map, display_answer_like_cols
                                )
                                group_rows_mrp.append(
                                    {
                                        "group": key_str,
                                        "n": int(idx_arr.size),
                                        "mass": mass,
                                        "major": bool(major),
                                        learned_l1_key: learned_l1,
                                    }
                                )

                            group_rows_mrp.sort(key=lambda r: float(r.get("mass", 0.0)), reverse=True)

            # ===========================
            # Fairness / worst-group dashboard + report plots + bundle export
            # ===========================

            st.subheader("Fairness & worst-group dashboard")

            if not group_cols or not group_rows:
                st.info("Select group columns in the sidebar to enable fairness / worst-group metrics.")
            else:
                has_truth = p_true is not None and truth is not None
                metric_label = "L1 error vs truth" if has_truth else "L1 divergence vs overall (proxy)"
                st.caption(
                    "If you upload synthetic data with a true_choice column, these are true errors. "
                    "Otherwise we show divergence vs the overall estimate as a robustness proxy."
                )

                # Add baseline proxy metric when truth is absent: baseline_l1 := ||p_g - p_overall||_1
                if not has_truth:
                    p_ref = p_baseline
                    for r in group_rows:
                        key_str = r["group"]
                        # Recover the group key by matching string is hard; instead compute proxy during build:
                        # Here we approximate using baseline_l1 already set nan; recompute by matching on group string
                        # by rebuilding from group_estimates dict (safe and small).
                    # Rebuild a map from display group string -> l1 proxy
                    disp_to_proxy: dict[str, float] = {}
                    for g, p_g in group_estimates.items():
                        key_str = _format_group_key(g, group_cols, category_display_map, display_answer_like_cols)
                        disp_to_proxy[key_str] = float(np.sum(np.abs(p_g - p_ref)))
                    for r in group_rows:
                        r["baseline_l1"] = disp_to_proxy.get(str(r["group"]), float("nan"))

                # Controls
                colA, colB, colC = st.columns([1, 1, 1])
                with colA:
                    major_only = st.checkbox(
                        "Major groups only", value=True, help="Only include groups with mass >= major_mass."
                    )
                with colB:
                    show_top = st.number_input(
                        "Show top N groups (by mass)", min_value=5, max_value=100, value=20, step=5
                    )
                with colC:
                    compare_options = ["RR debiasing"]
                    if group_rows_mrp is not None:
                        compare_options.append(learned_method_label)
                    compare_method = st.selectbox("Compare method", options=compare_options)

                # Pick which group rows to use
                if compare_method == learned_method_label and group_rows_mrp is not None:
                    g_rows = group_rows_mrp
                    key = learned_l1_key
                    title_prefix = f"{learned_method_label} ({metric_label})"
                else:
                    g_rows = group_rows
                    key = "baseline_l1"
                    title_prefix = f"RR debiasing ({metric_label})"

                # Apply the major-group filter for display. It may remove every
                # group, so the empty case is handled explicitly below.
                g_rows_show = []
                for r in g_rows:
                    if major_only and float(r.get("mass", 0.0)) < float(major_mass):
                        continue
                    g_rows_show.append(r)
                g_rows_show = sorted(g_rows_show, key=lambda r: float(r.get("mass", 0.0)), reverse=True)[
                    : int(show_top)
                ]

                if not g_rows_show:
                    # Do not attempt to render unavailable subgroup metrics as
                    # if they were valid estimates. The rest of the dashboard
                    # remains usable when this filter is too strict.
                    st.warning(
                        "No groups meet the current major-group mass threshold. "
                        "Lower the threshold or disable 'Major groups only' to view group metrics."
                    )
                else:
                    st.dataframe(g_rows_show, use_container_width=True)

                # Summary metrics
                summ = _group_metric_summary(
                    g_rows, metric_key=key, major_only=major_only, major_mass=float(major_mass)
                )
                if all(not np.isfinite(float(summ.get(m, float("nan")))) for m in ("worst", "p90", "weighted")):
                    st.info("Group summary metrics are unavailable for the current filter settings.")
                else:
                    st.write(
                        f"**Worst-group {metric_label}:** {_fmt(summ.get('worst', float('nan')), 6)}   |   "
                        f"**P90 {metric_label}:** {_fmt(summ.get('p90', float('nan')), 6)}   |   "
                        f"**Mass-weighted mean:** {_fmt(summ.get('weighted', float('nan')), 6)}   |   "
                        f"**Error ratio:** {_fmt(summ.get('error_ratio', float('nan')), 3)}"
                    )

                # Overall metrics (if truth)
                if has_truth:
                    overall_base = _overall_metrics(p_baseline, p_true)
                    st.write(
                        f"RR debiasing overall: L1={_fmt(overall_base['overall_l1'], 6)}, MAE={_fmt(overall_base['overall_mae'], 6)}, Correct winner: {bool(overall_base['correct_winner'])}"
                    )
                    if p_post_direct is not None:
                        overall_post = _overall_metrics(p_post_direct, p_true)
                        st.write(
                            f"Direct post-strat overall: L1={_fmt(overall_post['overall_l1'], 6)}, MAE={_fmt(overall_post['overall_mae'], 6)}, Correct winner: {bool(overall_post['correct_winner'])}"
                        )
                    if p_mrp_post is not None:
                        overall_mrp = _overall_metrics(p_mrp_post, p_true)
                        st.write(
                            f"{learned_method_label} post-strat overall: L1={_fmt(overall_mrp['overall_l1'], 6)}, MAE={_fmt(overall_mrp['overall_mae'], 6)}, Correct winner: {bool(overall_mrp['correct_winner'])}"
                        )

            # ---------- Plots (for report) ----------
            st.subheader("Plots (for report)")

            plot_bytes: dict[str, bytes] = {}

            series = [("baseline", p_baseline)]
            if p_post_direct is not None:
                series.append(("direct_poststrat", p_post_direct))
            if p_mrp_post is not None:
                series.append((f"{learned_prefix}_poststrat", p_mrp_post))
            if p_true is not None:
                series.append(("truth", p_true))

            overall_png = _plot_overall_distributions(
                labels=display_labels,
                series=series,
                title="Overall vote share estimate (comparison)",
            )
            if overall_png is not None:
                try:
                    st.image(overall_png, caption="Overall estimate comparison", use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not render overall plot image in the browser. ({e})")
                    st.info("Use the download buttons to view the plot file locally.")
                plot_bytes["overall_comparison.png"] = overall_png
                st.download_button(
                    "Download overall plot (PNG)",
                    data=overall_png,
                    file_name="overall_comparison.png",
                    mime="image/png",
                )

            # Group plot (baseline or mrp)
            if group_cols and group_rows:
                has_truth = p_true is not None and truth is not None
                metric_key = "baseline_l1"
                if not has_truth:
                    # ensure proxy exists
                    p_ref = p_baseline
                    disp_to_proxy: dict[str, float] = {}
                    for g, p_g in group_estimates.items():
                        key_str = _format_group_key(g, group_cols, category_display_map, display_answer_like_cols)
                        disp_to_proxy[key_str] = float(np.sum(np.abs(p_g - p_ref)))
                    for r in group_rows:
                        r["baseline_l1"] = disp_to_proxy.get(str(r["group"]), float("nan"))

                title = (
                    "Top groups by mass: L1 error (baseline vs truth)"
                    if has_truth
                    else "Top groups by mass: L1 divergence vs overall (baseline proxy)"
                )
                grp_png = _plot_group_bars(group_rows, title=title, metric_key=metric_key, top_n=20)
                if grp_png is not None:
                    st.image(grp_png, caption="Group metric (baseline)", use_container_width=True)
                    plot_bytes["group_metric_baseline.png"] = grp_png
                    st.download_button(
                        "Download group plot (PNG)",
                        data=grp_png,
                        file_name="group_metric_baseline.png",
                        mime="image/png",
                    )

            if plot_bytes:
                st.download_button(
                    "Download all plots (ZIP)",
                    data=build_plot_zip(plot_bytes),
                    file_name="fairvote_plots.zip",
                    mime="application/zip",
                )

            # ---------- Results bundle (ZIP) ----------
            st.subheader("Export results bundle (ZIP)")

            overall_csv_bytes = build_overall_estimates_csv(
                display_labels=display_labels,
                p_baseline=p_baseline,
                p_lo=p_lo,
                p_hi=p_hi,
                p_post_direct=p_post_direct,
                p_mrp_post=p_mrp_post,
                p_mrp_sample=p_mrp_sample,
                p_true=p_true,
                learned_post_col=learned_post_col,
                learned_sample_col=learned_sample_col,
            )
            group_csv_bytes = build_group_audit_csv(
                group_rows,
                group_rows_mrp=group_rows_mrp,
                learned_l1_key=learned_l1_key,
            )

            now = now_string()
            has_truth = p_true is not None and truth is not None
            summary_md_bytes = build_results_summary_markdown(
                generated_at=now,
                n_rows_used=n,
                epsilon=float(epsilon),
                k=int(k),
                method=method,
                group_cols=list(group_cols),
                group_rows=group_rows,
                group_rows_mrp=group_rows_mrp,
                learned_l1_key=learned_l1_key,
                learned_method_label=learned_method_label,
                major_mass=float(major_mass),
                p_baseline=p_baseline,
                p_true=p_true if has_truth else None,
                p_post_direct=p_post_direct,
                p_mrp_post=p_mrp_post,
                plot_names=list(plot_bytes.keys()),
            )

            extra_meta: dict[str, Any] = {}
            if _is_learned_mrp_method(method) and "mrp_feature_cols" in locals():
                extra_meta["learned_method"] = learned_method_label
                extra_meta["mrp_feature_cols"] = mrp_feature_cols
                extra_meta["mrp_lr"] = float(mrp_lr)
                extra_meta["mrp_steps"] = int(mrp_steps)
                extra_meta["mrp_batch"] = int(mrp_batch)
                extra_meta["mrp_seed"] = int(mrp_seed)
                if "mrp_l2" in locals():
                    extra_meta["mrp_l2"] = float(mrp_l2)
                if method == "Neural RR-aware MRP":
                    extra_meta["neural_hidden_layers"] = list(_parse_hidden_layers(neural_hidden_layers_text))
                    extra_meta["neural_dropout"] = float(neural_dropout)
                    extra_meta["neural_weight_decay"] = float(neural_weight_decay)
                if method == "Misreport-aware RR-MRP":
                    shy_idx = int(misreport_shy_category)
                    extra_meta["misreport_shy_label"] = (
                        display_labels[shy_idx] if 0 <= shy_idx < len(display_labels) else str(shy_idx)
                    )
                    extra_meta["misreport_honesty"] = float(misreport_honesty)
            extra_meta["synthetic_evaluation_mode"] = bool(synthetic_evaluation_mode)
            if pop_rows and post_cols and count_col is not None:
                extra_meta["post_cols"] = post_cols
                extra_meta["count_col"] = str(count_col)

            meta_bytes = build_metadata_json(
                generated_at=now,
                n_rows_used=n,
                epsilon=float(epsilon),
                k=int(k),
                method=method,
                response_col=response_col,
                truth_col=truth_col,
                group_cols=list(group_cols),
                major_mass=float(major_mass),
                has_truth=bool(has_truth),
                has_population=bool(pop_rows),
                extra=extra_meta,
            )
            bundle_bytes = build_results_bundle(
                overall_csv_bytes=overall_csv_bytes,
                group_csv_bytes=group_csv_bytes,
                summary_md_bytes=summary_md_bytes,
                meta_bytes=meta_bytes,
                plot_bytes=plot_bytes,
            )

            st.download_button(
                "Download Results Bundle (ZIP)",
                data=bundle_bytes,
                file_name="fairvote_results_bundle.zip",
                mime="application/zip",
            )

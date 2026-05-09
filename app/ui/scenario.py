"""Scenario-simulator Streamlit page."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st

from app.plotting.charts import plot_group_bars as _plot_group_bars
from app.plotting.charts import plot_overall_distributions as _plot_overall_distributions
from app.services.category import CategoryMap, encode_categories, filter_valid, poststratify_from_groups, read_population_weights
from app.services.exports import build_scenario_bundle, to_csv_bytes as _to_csv_bytes
from app.services.inference import DesignMatrix, MRPRRMultinomialModel, _HAS_RR_MRP, estimate_distribution
from app.services.metrics import fmt as _fmt
from app.services.metrics import group_metric_summary as _group_metric_summary
from app.services.scenario import (
    age_labels as _age_labels,
    apply_misreport as _apply_misreport,
    apply_nonresponse as _apply_nonresponse,
    apply_shy_effect as _apply_shy_effect,
    generate_population as _generate_population,
    party_labels_for_k as _party_labels_for_k,
    prefs_for_cell as _prefs_for_cell,
    region_labels as _region_labels,
    sample_from_population as _sample_from_population,
)


def render_scenario_tab(root: Path) -> None:
    # -----------------------
    # Controls
    # -----------------------
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        scenario = st.selectbox("Scenario", ["no_bias", "nonresponse", "shy_privacy_helps", "misreport"], index=1)
    with c2:
        n_resp = st.number_input("Respondents (n)", min_value=500, max_value=50000, value=5000, step=500)
    with c3:
        eps = st.selectbox("epsilon", [0.2, 0.5, 1.0, 2.0], index=2)
    with c4:
        seed = st.number_input("Seed", min_value=0, max_value=10_000_000, value=123, step=1)

    c5, c6, c7, c8 = st.columns([1, 1, 1, 1])
    with c5:
        k = st.selectbox("k parties", [3, 4, 5], index=2)
    with c6:
        n_regions = st.selectbox("regions", [4, 6, 8], index=2)
    with c7:
        age_options = [3, 4, 5, 6]
    default_age = 5
    n_ages = st.selectbox("age bands", age_options, index=age_options.index(default_age) if default_age in age_options else 0)
    with c8:
        total_pop = st.number_input("Population total (for post-strat)", min_value=50_000, max_value=10_000_000, value=1_000_000, step=50_000)

    # Scenario params
    st.markdown("#### Bias parameters")
    b1, b2, b3 = st.columns([1, 1, 1])
    with b1:
        nonresponse_base = st.slider("Base nonresponse rate", 0.0, 0.5, 0.15, 0.01, help="Only used in nonresponse scenario.")
    with b2:
        shy_base = st.slider("Shy misreport base", 0.0, 0.8, 0.40, 0.02, help="Only used in shy_privacy_helps scenario.")
    with b3:
        honesty = st.slider("Misreport honesty", 0.0, 1.0, 0.80, 0.01, help="Only used in misreport scenario.")

    run = st.button("Generate & run comparison", type="primary")

    if run:
        rng = np.random.default_rng(int(seed))
        parties = _party_labels_for_k(int(k))
        regions = _region_labels(int(n_regions))
        ages = _age_labels(int(n_ages))

        pop_rows = _generate_population(regions, ages, int(total_pop), rng)

        # sample demographics from population
        demos = _sample_from_population(pop_rows, int(n_resp), rng)

        # choose shy party index (if exists)
        shy_idx = 1 if ("Conservative" in parties) else 0  # default: party 1 if possible

        poll_rows = []
        for (reg, age) in demos:
            # response filtering (nonresponse)
            if scenario == "nonresponse":
                if not _apply_nonresponse(reg, age, float(nonresponse_base), rng):
                    continue

            p = _prefs_for_cell(parties, reg, age, rng)
            true_idx = int(rng.choice(len(parties), p=p))

            declared_idx = true_idx
            if scenario == "misreport":
                declared_idx = _apply_misreport(true_idx, len(parties), float(honesty), rng)
            elif scenario == "shy_privacy_helps":
                declared_idx = _apply_shy_effect(true_idx, shy_idx, len(parties), float(shy_base), float(eps), rng)

            from fairvote.privacy.mechanisms.kary_rr import privatize_one
            reported_idx = privatize_one(declared_idx, float(eps), len(parties), rng)

            poll_rows.append(
                {
                    "region": reg,
                    "age_band": age,
                    "true_choice": parties[true_idx],
                    "declared_choice": parties[declared_idx],
                    "reported_choice": parties[reported_idx],
                    "epsilon": str(float(eps)),
                }
            )

        if len(poll_rows) < 100:
            st.error("Too few respondents after bias (increase n or reduce nonresponse).")
        else:
            st.success(f"Generated poll: {len(poll_rows)} respondents (after bias), k={len(parties)}, eps={eps}")

            # Downloads
            poll_csv = _to_csv_bytes(poll_rows, ["region", "age_band", "true_choice", "declared_choice", "reported_choice", "epsilon"])
            pop_csv = _to_csv_bytes(pop_rows, ["region", "age_band", "count"])
            st.download_button("Download synthetic poll CSV", data=poll_csv, file_name="synthetic_poll.csv", mime="text/csv")
            st.download_button("Download synthetic population CSV", data=pop_csv, file_name="synthetic_population.csv", mime="text/csv")

            # --------------------------------
            # Run the same analysis pipeline as Upload tab
            # (fixed columns: reported_choice, true_choice, region, age_band)
            # --------------------------------
            cmap = CategoryMap(labels=parties, to_int={lab: i for i, lab in enumerate(parties)})
            k_eff = len(parties)

            reported = encode_categories([r["reported_choice"] for r in poll_rows], cmap)
            truth = encode_categories([r["true_choice"] for r in poll_rows], cmap)
            reported, truth, valid_mask = filter_valid(reported, truth)
            n_eff = int(reported.size)

            p_baseline = estimate_distribution(reported, epsilon=float(eps), k=k_eff)
            p_true = np.bincount(truth, minlength=k_eff).astype(float) / max(1.0, float(n_eff))
            
            from fairvote.privacy.mechanisms.laplace_mechanism import estimate_distribution_central_dp
            p_central_dp = estimate_distribution_central_dp(truth, epsilon=float(eps), k=k_eff, rng=rng)

            # Group audit by region|age_band
            major_mass = 0.02
            group_rows = []
            group_estimates = {}
            group_to_idx = {}
            for i, row in enumerate(poll_rows):
                key = (row["region"], row["age_band"])
                group_to_idx.setdefault(key, []).append(i)

            for g, idxs in group_to_idx.items():
                idx_arr = np.asarray(idxs, dtype=int)
                rep_g = reported[idx_arr]
                tru_g = truth[idx_arr]
                p_g = estimate_distribution(rep_g, epsilon=float(eps), k=k_eff)
                p_tg = np.bincount(tru_g, minlength=k_eff).astype(float) / max(1.0, float(tru_g.size))
                group_estimates[g] = p_g
                mass = float(rep_g.size) / float(n_eff)
                group_rows.append(
                    {
                        "group": f"{g[0]} | {g[1]}",
                        "n": int(rep_g.size),
                        "mass": mass,
                        "major": bool(mass >= major_mass),
                        "baseline_l1": float(np.sum(np.abs(p_g - p_tg))),
                    }
                )
            group_rows.sort(key=lambda r: float(r["mass"]), reverse=True)

            # Direct post-strat (keys match)
            pop_weights = read_population_weights(pop_rows, ["region", "age_band"], "count")
            p_post_direct = poststratify_from_groups(group_estimates, pop_weights, fallback=p_baseline) if pop_weights else None

            # RR-aware MRP (optional, if available)
            p_mrp_post = None
            group_rows_mrp = None
            p_mrp_sample = None
            if _HAS_RR_MRP and MRPRRMultinomialModel is not None and DesignMatrix is not None:
                # Fit MRP with region + age_band
                design = DesignMatrix(["region", "age_band"]).fit(poll_rows)
                X = design.transform(poll_rows)
                model = MRPRRMultinomialModel(k=k_eff, epsilon=float(eps), l2=1.0, seed=int(seed))
                with st.spinner("Fitting RR-aware MRP..."):
                    model.fit(X, reported, lr=0.05, steps=2000, batch_size=512, verbose_every=0, keep_history=False)

                P_true = model.predict_true_proba(X)
                p_mrp_sample = np.mean(P_true, axis=0)
                p_mrp_sample = np.clip(p_mrp_sample, 0.0, 1.0)
                s = float(p_mrp_sample.sum())
                if s > 0:
                    p_mrp_sample /= s

                # Post-strat with population
                pop_cells = [{"region": r["region"], "age_band": r["age_band"]} for r in pop_rows]
                pop_counts = np.asarray([float(r["count"]) for r in pop_rows], dtype=float)
                Xp = design.transform(pop_cells)
                p_mrp_post = model.poststratify(Xp, pop_counts)

                # Group-level MRP L1 vs truth
                group_rows_mrp = []
                for g, idxs in group_to_idx.items():
                    idx_arr = np.asarray(idxs, dtype=int)
                    p_g_mrp = np.mean(P_true[idx_arr], axis=0)
                    p_g_mrp = np.clip(p_g_mrp, 0.0, 1.0)
                    s = float(p_g_mrp.sum())
                    if s > 0:
                        p_g_mrp /= s
                    tru_g = truth[idx_arr]
                    p_tg = np.bincount(tru_g, minlength=k_eff).astype(float) / max(1.0, float(tru_g.size))
                    mass = float(idx_arr.size) / float(n_eff)
                    group_rows_mrp.append(
                        {
                            "group": f"{g[0]} | {g[1]}",
                            "n": int(idx_arr.size),
                            "mass": mass,
                            "major": bool(mass >= major_mass),
                            "mrp_l1": float(np.sum(np.abs(p_g_mrp - p_tg))),
                        }
                    )
                group_rows_mrp.sort(key=lambda r: float(r["mass"]), reverse=True)

            # -----------------------
            # Display: overall table
            # -----------------------
            st.subheader("Overall comparison (truth known)")
            rows = []
            for i, lab in enumerate(parties):
                row = {"label": lab, "true_p": float(p_true[i]), "baseline_p": float(p_baseline[i]), "central_dp_p": float(p_central_dp[i])}
                if p_post_direct is not None:
                    row["direct_poststrat_p"] = float(p_post_direct[i])
                if p_mrp_post is not None:
                    row["mrp_poststrat_p"] = float(p_mrp_post[i])
                rows.append(row)
            st.dataframe(rows, use_container_width=True)

            # -----------------------
            # Fairness summary
            # -----------------------
            st.subheader("Worst-group / fairness metrics (truth known)")
            base_s = _group_metric_summary(group_rows, metric_key="baseline_l1", major_only=True, major_mass=major_mass)
            st.write(
                f"Baseline major-groups: worst={_fmt(base_s['worst'],6)} | p90={_fmt(base_s['p90'],6)} | weighted={_fmt(base_s['weighted'],6)}"
            )
            if group_rows_mrp is not None:
                mrp_s = _group_metric_summary(group_rows_mrp, metric_key="mrp_l1", major_only=True, major_mass=major_mass)
                st.write(
                    f"MRP major-groups: worst={_fmt(mrp_s['worst'],6)} | p90={_fmt(mrp_s['p90'],6)} | weighted={_fmt(mrp_s['weighted'],6)}"
                )
            st.caption("Group table below is sorted by mass (largest groups first).")
            st.dataframe(group_rows, use_container_width=True)
            if group_rows_mrp is not None:
                st.dataframe(group_rows_mrp, use_container_width=True)

            # -----------------------
            # Plots + bundle
            # -----------------------
            st.subheader("Report-ready plots + bundle")
            plot_bytes = {}
            series = [("truth", p_true), ("baseline (LDP)", p_baseline), ("central DP", p_central_dp)]
            if p_post_direct is not None:
                series.append(("direct_poststrat", p_post_direct))
            if p_mrp_post is not None:
                series.append(("mrp_poststrat", p_mrp_post))

            overall_png = _plot_overall_distributions(parties, series, "Overall vote share (truth vs methods)")
            if overall_png is not None:
                st.image(overall_png, use_container_width=True)
                plot_bytes["overall_comparison.png"] = overall_png

            grp_png = _plot_group_bars(group_rows, "Top groups by mass: baseline L1 vs truth", "baseline_l1", top_n=20)
            if grp_png is not None:
                st.image(grp_png, use_container_width=True)
                plot_bytes["group_baseline_l1.png"] = grp_png

            # Bundle ZIP
            overall_csv = _to_csv_bytes(rows, list(rows[0].keys()))
            group_csv = _to_csv_bytes(group_rows, ["group", "n", "mass", "major", "baseline_l1"])
            md = "\n".join(
                [
                    "# FairVote-AI Scenario Simulator Run",
                    "",
                    f"- scenario: {scenario}",
                    f"- n: {len(poll_rows)} (after bias)",
                    f"- epsilon: {eps}",
                    f"- k: {k_eff}",
                    "",
                    "## Major-group fairness metrics",
                    f"- baseline: worst={base_s['worst']:.6f}, p90={base_s['p90']:.6f}, weighted={base_s['weighted']:.6f}",
                ]
            )
            if group_rows_mrp is not None:
                mrp_s = _group_metric_summary(group_rows_mrp, metric_key="mrp_l1", major_only=True, major_mass=major_mass)
                md += "\n" + f"- mrp: worst={mrp_s['worst']:.6f}, p90={mrp_s['p90']:.6f}, weighted={mrp_s['weighted']:.6f}"

            meta = {
                "scenario": scenario,
                "n_after_bias": len(poll_rows),
                "epsilon": float(eps),
                "seed": int(seed),
                "k": int(k_eff),
                "regions": regions,
                "ages": ages,
                "nonresponse_base": float(nonresponse_base),
                "shy_base": float(shy_base),
                "honesty": float(honesty),
                "has_mrp": bool(p_mrp_post is not None),
            }

            bundle_bytes = build_scenario_bundle(
                poll_csv=poll_csv,
                population_csv=pop_csv,
                overall_csv=overall_csv,
                group_csv=group_csv,
                summary_md=md.encode("utf-8"),
                metadata=meta,
                plot_bytes=plot_bytes,
            )

            st.download_button(
                "Download scenario bundle (ZIP)",
                data=bundle_bytes,
                file_name="fairvote_scenario_bundle.zip",
                mime="application/zip",
            )
            st.success("Done. Use the ZIP contents directly in your report.")


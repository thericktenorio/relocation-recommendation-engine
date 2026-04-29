from __future__ import annotations

import json
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from layer1_constraints import ConstraintInputs, run_constraint_layer, build_constraint_diagnostics
from data_loader import load_housing_data, load_job_data

from pair_generation import generate_candidate_pairs
from layer2_pair_builder import build_layer2_pair_representations

from data.image_comparator.image_metadata import IMAGE_PAIRS
from image_preference_vector import build_image_preference_output

from layer3_weighted_inference import run_weighted_inference
from layer3_fuzzy_inference import rank_pairs_fuzzy

from layer4_evaluation import (
    build_layer4_comparison_set,
    add_user_ratings,
    evaluate_model_alignment,
    summarize_layer4_results,
    normalize_layer4_model_scores,
)

from layer5_recommendation_presentation import (
    determine_final_ranking_method,
    build_final_rankings,
    render_layer5_recommendations,
    build_model_comparison_summary,
    render_model_comparison_summary,
)


st.set_page_config(
    page_title="Relocation Recommendation Engine",
    layout="wide",
)


@st.cache_data
def load_data():
    housing_df = load_housing_data("data/enriched/final_recommendation_engine_dataset.csv")
    jobs_df = load_job_data("data/clean/adzuna_jobs_clean.csv")
    return housing_df, jobs_df


def ensure_pair_ids(pairs: list[dict]) -> list[dict]:
    updated_pairs = []

    for pair in pairs:
        pair_copy = dict(pair)

        if not pair_copy.get("pair_id"):
            home_id = pair_copy.get("home_id")
            job_id = pair_copy.get("job_id")
            pair_copy["pair_id"] = f"{home_id}__{job_id}"

        updated_pairs.append(pair_copy)

    return updated_pairs


def display_constraint_warnings(diagnostics: dict) -> None:
    housing_final = diagnostics["housing"].get("final", 0)
    jobs_final = diagnostics["jobs"].get("final", 0)

    housing_after_city = diagnostics["housing"].get("after_city_filter", 0)
    housing_after_pref = diagnostics["housing"].get("after_housing_preference_filter", 0)
    housing_after_household = diagnostics["housing"].get("after_household_size_filter", 0)

    jobs_after_city = diagnostics["jobs"].get("after_city_filter", 0)
    jobs_after_industry = diagnostics["jobs"].get("after_industry_filter", 0)
    jobs_after_career = diagnostics["jobs"].get("after_career_stage_filter", 0)

    if housing_final == 0:
        st.warning(
            "No housing listings remain after your filters. "
            "Try broadening the housing preference, household size rule, or selected cities."
        )

    if jobs_final == 0:
        st.warning(
            "No job listings remain after your filters. "
            "Try broadening the industry, career stage, or selected cities."
        )

    if housing_after_city == 0:
        st.warning("No housing listings were found in the selected city or cities.")

    if housing_after_pref == 0:
        st.warning(
            "The selected housing preference removed all housing options in the selected city pool."
        )

    if housing_after_household == 0 and housing_after_pref > 0:
        st.warning(
            "The household-size bedroom rule removed all remaining housing options."
        )

    if jobs_after_city == 0:
        st.warning("No jobs were found in the selected city or cities.")

    if jobs_after_industry == 0 and jobs_after_city > 0:
        st.warning("The selected industry removed all remaining job options.")

    if jobs_after_career == 0 and jobs_after_industry > 0:
        st.warning("The selected career stage removed all remaining job options.")

    if 0 < housing_final < 10:
        st.info(
            f"Only {housing_final} housing listing(s) remain after filtering. "
            "This may be too narrow for strong recommendations."
        )

    if 0 < jobs_final < 10:
        st.info(
            f"Only {jobs_final} job listing(s) remain after filtering. "
            "This may be too narrow for strong recommendations."
        )


def reset_downstream_state() -> None:
    keys_to_clear = [
        "layer2b_normalized_vector",
        "layer3a_results",
        "layer3b_results",
        "layer4_comparison_df",
        "layer4_evaluation_result",
        "recommendations_ready",
        "engine_signature",
    ]

    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]


def safe_get_nested(row: pd.Series, parent_key: str, child_key: str, default: Any = "N/A") -> Any:
    value = row.get(parent_key, {})

    if isinstance(value, dict):
        return value.get(child_key, default)

    return default


def build_engine_signature(inputs: ConstraintInputs) -> str:
    selected_images = st.session_state.get("selected_images", {})

    image_signature = {
        pair_id: selected.get("description", "")
        for pair_id, selected in sorted(selected_images.items())
    }

    payload = {
        "selected_cities": inputs.selected_cities,
        "selected_industry": inputs.selected_industry,
        "selected_career_stage": inputs.selected_career_stage,
        "housing_preference": inputs.housing_preference,
        "household_size": inputs.household_size,
        "ignore_household_size": inputs.ignore_household_size,
        "selected_images": image_signature,
    }

    return json.dumps(payload, sort_keys=True)


def build_dimension_diagnostic_display_df(eval_res: dict) -> pd.DataFrame:
    dimension_summary_df = eval_res.get("dimension_alignment_summary", pd.DataFrame())

    if dimension_summary_df is None or dimension_summary_df.empty:
        return pd.DataFrame()

    display_df = dimension_summary_df.copy()

    dimension_labels = {
        "job_fit": "Job Fit",
        "housing_fit": "Housing Fit",
        "lifestyle_fit": "Lifestyle Fit",
        "commute_fit": "Commute Fit",
        "overall_fit": "Overall Pair Fit",
    }

    display_df["Dimension"] = display_df["dimension"].map(dimension_labels).fillna(
        display_df["dimension"]
    )

    display_df["Layer 3A MAE"] = display_df["layer3a_mae"].round(4)
    display_df["Layer 3B MAE"] = display_df["layer3b_mae"].round(4)
    display_df["Better Aligned Model"] = display_df["better_aligned_model"]

    return display_df[
        [
            "Dimension",
            "Layer 3A MAE",
            "Layer 3B MAE",
            "Better Aligned Model",
        ]
    ]


def build_rated_results_display_df(rated_results_df: pd.DataFrame) -> pd.DataFrame:
    if rated_results_df is None or rated_results_df.empty:
        return pd.DataFrame()

    columns_to_keep = [
        "pair_id",
        "home_id",
        "job_id",
        "final_score",
        "layer3b_fuzzy_score",
        "user_alignment_score",
        "user_job_fit",
        "user_housing_fit",
        "user_lifestyle_fit",
        "user_commute_fit",
        "user_overall_fit",
        "layer3a_abs_error",
        "layer3b_abs_error",
    ]

    existing_columns = [
        col for col in columns_to_keep
        if col in rated_results_df.columns
    ]

    display_df = rated_results_df[existing_columns].copy()

    rename_map = {
        "pair_id": "Pair ID",
        "home_id": "Home ID",
        "job_id": "Job ID",
        "final_score": "Weighted Logic Score",
        "layer3b_fuzzy_score": "Fuzzy Logic Score",
        "user_alignment_score": "User Alignment Score",
        "user_job_fit": "User Job Fit",
        "user_housing_fit": "User Housing Fit",
        "user_lifestyle_fit": "User Lifestyle Fit",
        "user_commute_fit": "User Commute Fit",
        "user_overall_fit": "User Overall Fit",
        "layer3a_abs_error": "Weighted Logic Abs Error",
        "layer3b_abs_error": "Fuzzy Logic Abs Error",
    }

    display_df = display_df.rename(columns=rename_map)

    numeric_columns = display_df.select_dtypes(include="number").columns
    display_df[numeric_columns] = display_df[numeric_columns].round(4)

    return display_df



def build_advanced_diagnostics_df(rated_results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build row-level diagnostic fields used by the advanced Layer 4 charts.

    The diagnostics intentionally use the same scored and rated rows that already
    appear in the Rated Pair-Level Results table. This keeps the charts aligned
    with the user's actual evaluation inputs.
    """
    required_columns = [
        "pair_id",
        "final_score",
        "layer3b_fuzzy_score",
        "user_alignment_score",
    ]

    if rated_results_df is None or rated_results_df.empty:
        return pd.DataFrame()

    missing_columns = [
        col for col in required_columns
        if col not in rated_results_df.columns
    ]

    if missing_columns:
        return pd.DataFrame()

    diagnostics_df = rated_results_df.copy()

    numeric_columns = [
        "final_score",
        "layer3b_fuzzy_score",
        "user_alignment_score",
    ]

    for col in numeric_columns:
        diagnostics_df[col] = pd.to_numeric(diagnostics_df[col], errors="coerce")

    diagnostics_df = diagnostics_df.dropna(subset=numeric_columns)

    if diagnostics_df.empty:
        return pd.DataFrame()

    diagnostics_df["weighted_signed_error"] = (
        diagnostics_df["final_score"] - diagnostics_df["user_alignment_score"]
    )
    diagnostics_df["fuzzy_signed_error"] = (
        diagnostics_df["layer3b_fuzzy_score"] - diagnostics_df["user_alignment_score"]
    )

    diagnostics_df["weighted_abs_error"] = diagnostics_df["weighted_signed_error"].abs()
    diagnostics_df["fuzzy_abs_error"] = diagnostics_df["fuzzy_signed_error"].abs()
    diagnostics_df["model_disagreement"] = (
        diagnostics_df["final_score"] - diagnostics_df["layer3b_fuzzy_score"]
    ).abs()

    diagnostics_df["weighted_rank"] = diagnostics_df["final_score"].rank(
        ascending=False,
        method="min",
    )
    diagnostics_df["fuzzy_rank"] = diagnostics_df["layer3b_fuzzy_score"].rank(
        ascending=False,
        method="min",
    )
    diagnostics_df["user_rank"] = diagnostics_df["user_alignment_score"].rank(
        ascending=False,
        method="min",
    )

    def determine_closer_model(row: pd.Series) -> str:
        weighted_error = row["weighted_abs_error"]
        fuzzy_error = row["fuzzy_abs_error"]

        if weighted_error < fuzzy_error:
            return "Weighted Logic"
        if fuzzy_error < weighted_error:
            return "Fuzzy Logic"
        return "Tie"

    diagnostics_df["closer_model"] = diagnostics_df.apply(
        determine_closer_model,
        axis=1,
    )

    diagnostics_df["pair_label"] = [
        f"Pair {i}"
        for i in range(1, len(diagnostics_df) + 1)
    ]

    return diagnostics_df


def build_top_n_agreement_df(
    diagnostics_df: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    if diagnostics_df is None or diagnostics_df.empty:
        return pd.DataFrame()

    safe_top_n = max(1, min(int(top_n), len(diagnostics_df)))

    user_top_pairs = set(
        diagnostics_df.nlargest(safe_top_n, "user_alignment_score")["pair_id"]
    )
    weighted_top_pairs = set(
        diagnostics_df.nlargest(safe_top_n, "final_score")["pair_id"]
    )
    fuzzy_top_pairs = set(
        diagnostics_df.nlargest(safe_top_n, "layer3b_fuzzy_score")["pair_id"]
    )

    weighted_overlap = len(user_top_pairs.intersection(weighted_top_pairs))
    fuzzy_overlap = len(user_top_pairs.intersection(fuzzy_top_pairs))

    return pd.DataFrame(
        {
            "Model": ["Weighted Logic", "Fuzzy Logic"],
            "Top-N Agreement Count": [weighted_overlap, fuzzy_overlap],
            "Top-N Agreement Rate": [
                weighted_overlap / safe_top_n,
                fuzzy_overlap / safe_top_n,
            ],
            "Top N": [safe_top_n, safe_top_n],
        }
    )


def classify_bias_direction(bias_value: float, tolerance: float = 0.025) -> str:
    if pd.isna(bias_value):
        return "N/A"

    if bias_value > tolerance:
        return "Overestimates User Fit"
    if bias_value < -tolerance:
        return "Underestimates User Fit"
    return "Approximately Balanced"


def build_model_bias_df(diagnostics_df: pd.DataFrame) -> pd.DataFrame:
    if diagnostics_df is None or diagnostics_df.empty:
        return pd.DataFrame()

    weighted_bias = diagnostics_df["weighted_signed_error"].mean()
    fuzzy_bias = diagnostics_df["fuzzy_signed_error"].mean()

    return pd.DataFrame(
        {
            "Model": ["Weighted Logic", "Fuzzy Logic"],
            "Average Signed Error": [weighted_bias, fuzzy_bias],
            "Bias Direction": [
                classify_bias_direction(weighted_bias),
                classify_bias_direction(fuzzy_bias),
            ],
        }
    )


def render_advanced_model_diagnostics(eval_res: dict) -> None:
    rated_results_df = eval_res.get("rated_results", pd.DataFrame())
    diagnostics_df = build_advanced_diagnostics_df(rated_results_df)

    with st.expander("Advanced Model Diagnostics", expanded=True):
        st.markdown(
            """
            These diagnostics compare the model scores against the user's own rated
            preference baseline. They are intended to explain *why* one model may be
            better aligned, not merely report which model had the lower MAE.
            """
        )

        if diagnostics_df.empty:
            st.info("Advanced diagnostics are not available for the current rated results.")
            return

        top_n_default = min(5, len(diagnostics_df))
        top_n = st.slider(
            "Top-N agreement size",
            min_value=1,
            max_value=len(diagnostics_df),
            value=top_n_default,
            key="advanced_diagnostics_top_n",
        )

        top_n_df = build_top_n_agreement_df(diagnostics_df, top_n)
        bias_df = build_model_bias_df(diagnostics_df)

        st.markdown("#### Top-N Agreement with User Preferences")
        st.caption(
            "This shows how many of the user's top-rated pairs also appear in each model's top-N list."
        )

        agreement_col1, agreement_col2 = st.columns(2)

        weighted_agreement_row = top_n_df[top_n_df["Model"] == "Weighted Logic"].iloc[0]
        fuzzy_agreement_row = top_n_df[top_n_df["Model"] == "Fuzzy Logic"].iloc[0]

        with agreement_col1:
            st.metric(
                "Weighted Logic Top-N Agreement",
                f"{int(weighted_agreement_row['Top-N Agreement Count'])}/{top_n}",
            )

        with agreement_col2:
            st.metric(
                "Fuzzy Logic Top-N Agreement",
                f"{int(fuzzy_agreement_row['Top-N Agreement Count'])}/{top_n}",
            )

        top_n_chart = (
            alt.Chart(top_n_df)
            .mark_bar()
            .encode(
                x=alt.X("Model:N", title="Model"),
                y=alt.Y(
                    "Top-N Agreement Rate:Q",
                    title="Top-N Agreement Rate",
                    scale=alt.Scale(domain=[0, 1]),
                ),
                tooltip=[
                    "Model",
                    alt.Tooltip("Top-N Agreement Count:Q", format=".0f"),
                    alt.Tooltip("Top N:Q", format=".0f"),
                    alt.Tooltip("Top-N Agreement Rate:Q", format=".2%"),
                ],
            )
        )

        st.altair_chart(top_n_chart, use_container_width=True)

        st.markdown("#### Error Direction / Model Bias")
        st.caption(
            "Signed error is model score minus user alignment score. Positive values mean the model overestimated fit; negative values mean it underestimated fit."
        )

        bias_col1, bias_col2 = st.columns(2)

        weighted_bias_row = bias_df[bias_df["Model"] == "Weighted Logic"].iloc[0]
        fuzzy_bias_row = bias_df[bias_df["Model"] == "Fuzzy Logic"].iloc[0]

        with bias_col1:
            st.metric(
                "Weighted Logic Avg Signed Error",
                f"{weighted_bias_row['Average Signed Error']:.4f}",
                help=weighted_bias_row["Bias Direction"],
            )

        with bias_col2:
            st.metric(
                "Fuzzy Logic Avg Signed Error",
                f"{fuzzy_bias_row['Average Signed Error']:.4f}",
                help=fuzzy_bias_row["Bias Direction"],
            )

        bias_chart = (
            alt.Chart(bias_df)
            .mark_bar()
            .encode(
                x=alt.X("Model:N", title="Model"),
                y=alt.Y("Average Signed Error:Q", title="Average Signed Error"),
                tooltip=[
                    "Model",
                    "Bias Direction",
                    alt.Tooltip("Average Signed Error:Q", format=".4f"),
                ],
            )
        )

        zero_line = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule().encode(y="y:Q")
        st.altair_chart(bias_chart + zero_line, use_container_width=True)

        st.markdown("#### Model Disagreement Analysis")
        st.caption(
            "This highlights evaluated pairs where weighted logic and fuzzy logic disagreed most strongly, then shows which model was closer to the user's rating."
        )

        disagreement_count = min(5, len(diagnostics_df))
        top_disagreement_df = diagnostics_df.nlargest(
            disagreement_count,
            "model_disagreement",
        ).copy()

        disagreement_winner_counts = top_disagreement_df["closer_model"].value_counts()
        weighted_wins = int(disagreement_winner_counts.get("Weighted Logic", 0))
        fuzzy_wins = int(disagreement_winner_counts.get("Fuzzy Logic", 0))
        ties = int(disagreement_winner_counts.get("Tie", 0))

        disagreement_col1, disagreement_col2, disagreement_col3 = st.columns(3)

        with disagreement_col1:
            st.metric("Weighted Logic Closer", weighted_wins)

        with disagreement_col2:
            st.metric("Fuzzy Logic Closer", fuzzy_wins)

        with disagreement_col3:
            st.metric("Ties", ties)

        disagreement_chart = (
            alt.Chart(top_disagreement_df)
            .mark_bar()
            .encode(
                x=alt.X(
                    "pair_label:N",
                    title="Highest-Disagreement Evaluated Pairs",
                    sort=None,
                ),
                y=alt.Y("model_disagreement:Q", title="Absolute Model Disagreement"),
                color=alt.Color("closer_model:N", title="Model Closer to User"),
                tooltip=[
                    "pair_id",
                    alt.Tooltip("final_score:Q", title="Weighted Logic Score", format=".4f"),
                    alt.Tooltip("layer3b_fuzzy_score:Q", title="Fuzzy Logic Score", format=".4f"),
                    alt.Tooltip("user_alignment_score:Q", title="User Alignment Score", format=".4f"),
                    alt.Tooltip("model_disagreement:Q", title="Model Disagreement", format=".4f"),
                    alt.Tooltip("weighted_abs_error:Q", title="Weighted Logic Abs Error", format=".4f"),
                    alt.Tooltip("fuzzy_abs_error:Q", title="Fuzzy Logic Abs Error", format=".4f"),
                    alt.Tooltip("closer_model:N", title="Model Closer to User"),
                ],
            )
        )

        st.altair_chart(disagreement_chart, use_container_width=True)

        with st.expander("Advanced Diagnostic Data", expanded=False):
            diagnostic_display_df = diagnostics_df[
                [
                    "pair_id",
                    "final_score",
                    "layer3b_fuzzy_score",
                    "user_alignment_score",
                    "weighted_signed_error",
                    "fuzzy_signed_error",
                    "weighted_abs_error",
                    "fuzzy_abs_error",
                    "model_disagreement",
                    "closer_model",
                    "user_rank",
                    "weighted_rank",
                    "fuzzy_rank",
                ]
            ].copy()

            diagnostic_display_df = diagnostic_display_df.rename(
                columns={
                    "pair_id": "Pair ID",
                    "final_score": "Weighted Logic Score",
                    "layer3b_fuzzy_score": "Fuzzy Logic Score",
                    "user_alignment_score": "User Alignment Score",
                    "weighted_signed_error": "Weighted Logic Signed Error",
                    "fuzzy_signed_error": "Fuzzy Logic Signed Error",
                    "weighted_abs_error": "Weighted Logic Abs Error",
                    "fuzzy_abs_error": "Fuzzy Logic Abs Error",
                    "model_disagreement": "Model Disagreement",
                    "closer_model": "Model Closer to User",
                    "user_rank": "User Rank",
                    "weighted_rank": "Weighted Logic Rank",
                    "fuzzy_rank": "Fuzzy Logic Rank",
                }
            )

            numeric_columns = diagnostic_display_df.select_dtypes(include="number").columns
            diagnostic_display_df[numeric_columns] = diagnostic_display_df[numeric_columns].round(4)

            st.dataframe(diagnostic_display_df, use_container_width=True)

def render_layer4_diagnostics(eval_res: dict) -> None:
    st.subheader("Layer 4: Evaluation and Model Comparison")

    layer3a_mae = eval_res["layer3a_mean_absolute_error"]
    layer3b_mae = eval_res["layer3b_mean_absolute_error"]
    better_model = eval_res["better_aligned_model"]
    pair_count = eval_res["evaluated_pair_count"]

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        st.metric("Evaluated Pairs", pair_count)

    with metric_col2:
        st.metric("Weighted Logic MAE", f"{layer3a_mae:.4f}")

    with metric_col3:
        st.metric("Fuzzy Logic MAE", f"{layer3b_mae:.4f}")

    with metric_col4:
        st.metric("Better Aligned Model", better_model)

    with st.expander("Overall Evaluation Summary", expanded=False):
        st.text(summarize_layer4_results(eval_res))

        error_df = pd.DataFrame(
            {
                "Model": [
                    "Weighted Logic",
                    "Fuzzy Logic",
                ],
                "MAE": [
                    layer3a_mae,
                    layer3b_mae,
                ],
            }
        )

        overall_chart = (
            alt.Chart(error_df)
            .mark_bar()
            .encode(
                x=alt.X("Model:N", title="Model"),
                y=alt.Y("MAE:Q", title="MAE"),
                tooltip=[
                    "Model",
                    alt.Tooltip("MAE:Q", format=".4f"),
                ],
            )
        )

        st.altair_chart(overall_chart, use_container_width=True)

    dimension_display_df = build_dimension_diagnostic_display_df(eval_res)

    with st.expander("Dimension-Level Diagnostic Evaluation", expanded=True):
        st.markdown(
            """
            This table compares each model against the same user rating dimensions:
            job fit, housing fit, lifestyle fit, commute fit, and overall pair fit.
            Lower MAE indicates stronger alignment with the user's ratings for that dimension.
            """
        )

        if dimension_display_df.empty:
            st.info("Dimension-level diagnostic results are not available.")
        else:
            st.dataframe(dimension_display_df, use_container_width=True)

            dimension_chart_df = dimension_display_df.rename(
                columns={
                    "Layer 3A MAE": "Weighted Logic MAE",
                    "Layer 3B MAE": "Fuzzy Logic MAE",
                }
            )

            dimension_chart_long_df = dimension_chart_df.melt(
                id_vars="Dimension",
                value_vars=["Weighted Logic MAE", "Fuzzy Logic MAE"],
                var_name="Model",
                value_name="MAE",
            )

            dimension_chart = (
                alt.Chart(dimension_chart_long_df)
                .mark_bar()
                .encode(
                    x=alt.X("Dimension:N", title="Dimension"),
                    y=alt.Y("MAE:Q", title="MAE"),
                    color=alt.Color("Model:N", title="Model"),
                    xOffset="Model:N",
                    tooltip=[
                        "Dimension",
                        "Model",
                        alt.Tooltip("MAE:Q", format=".4f"),
                    ],
                )
            )

            st.altair_chart(dimension_chart, use_container_width=True)

    render_advanced_model_diagnostics(eval_res)

    rated_results_df = eval_res.get("rated_results", pd.DataFrame())
    rated_results_display_df = build_rated_results_display_df(rated_results_df)

    with st.expander("Rated Pair-Level Results", expanded=False):
        st.markdown(
            """
            This table is the most detailed evaluation output. It shows each evaluated
            job-housing pair, both model scores, the user's composite alignment score,
            the user's individual ratings, and each model's absolute error.
            """
        )

        if rated_results_display_df.empty:
            st.info("Rated result details are not available.")
        else:
            st.dataframe(rated_results_display_df, use_container_width=True)


def build_engine_outputs(
    filtered_housing_df: pd.DataFrame,
    filtered_jobs_df: pd.DataFrame,
    inputs: ConstraintInputs,
) -> None:
    selected_images = st.session_state.get("selected_images", {})

    if len(selected_images) < len(IMAGE_PAIRS):
        return

    preference_result = build_image_preference_output(
        list(selected_images.values())
    )

    normalized_vector = preference_result["normalized_vector"]
    st.session_state["layer2b_normalized_vector"] = normalized_vector

    effective_household_size = (
        1 if inputs.ignore_household_size else int(inputs.household_size)
    )

    candidate_pairs = generate_candidate_pairs(
        housing_df=filtered_housing_df,
        jobs_df=filtered_jobs_df,
        max_jobs_per_home=5,
        max_commute_miles=25.0,
    )

    if not candidate_pairs:
        st.warning("No job-housing pairs could be generated. Please adjust your filters.")
        return

    inferred_profile = {
        "urbanism_preference": 0.5,
        "frugality_preference": 0.3,
        "outdoor_preference": 0.8,
        "family_space_preference": 0.4,
    }

    representations = build_layer2_pair_representations(
        candidate_pairs=candidate_pairs,
        housing_df=filtered_housing_df,
        preferred_cities=inputs.selected_cities,
        industry=inputs.selected_industry,
        career_stage=inputs.selected_career_stage,
        housing_preference=inputs.housing_preference,
        household_size=effective_household_size,
        inferred_profile=inferred_profile,
    )

    scored_pairs = run_weighted_inference(
        pair_representations=representations,
        preference_vector=normalized_vector,
    )

    fuzzy_pairs = rank_pairs_fuzzy(
        pair_representations=representations,
        preference_vector=normalized_vector,
    )

    scored_pairs = ensure_pair_ids(scored_pairs)
    fuzzy_pairs = ensure_pair_ids(fuzzy_pairs)

    comparison_df = build_layer4_comparison_set(
        layer3a_results=scored_pairs,
        layer3b_results=fuzzy_pairs,
        top_n=5,
    )

    comparison_df = normalize_layer4_model_scores(comparison_df)

    st.session_state["layer3a_results"] = scored_pairs
    st.session_state["layer3b_results"] = fuzzy_pairs
    st.session_state["layer4_comparison_df"] = comparison_df
    st.session_state["recommendations_ready"] = True
    st.session_state["engine_signature"] = build_engine_signature(inputs)

    if "layer4_evaluation_result" in st.session_state:
        del st.session_state["layer4_evaluation_result"]


def render_filter_section(
    housing_df: pd.DataFrame,
    jobs_df: pd.DataFrame,
) -> tuple[ConstraintInputs, pd.DataFrame, pd.DataFrame]:
    st.header("Step 1: Select Filters")

    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:
        selected_cities = st.multiselect(
            "City Preference",
            options=["Dallas, TX", "Raleigh, NC", "San Diego, CA"],
            default=["Dallas, TX", "Raleigh, NC", "San Diego, CA"],
            on_change=reset_downstream_state,
        )

    with filter_col2:
        selected_industry = st.selectbox(
            "Industry",
            options=[
                "All Available Industries",
                "Finance",
                "Healthcare",
                "Technology",
            ],
            index=0,
            on_change=reset_downstream_state,
        )

    with filter_col3:
        selected_career_stage = st.selectbox(
            "Career Stage",
            options=["All", "Entry", "Mid", "Senior"],
            index=0,
            on_change=reset_downstream_state,
        )

    filter_col4, filter_col5, filter_col6 = st.columns(3)

    with filter_col4:
        housing_preference = st.selectbox(
            "Housing Preference",
            options=[
                "Lowest rent possible",
                "Fairly priced",
                "Luxury",
                "Open",
            ],
            index=3,
            on_change=reset_downstream_state,
        )

    with filter_col5:
        ignore_household_size = st.checkbox(
            "Ignore household size",
            value=False,
            on_change=reset_downstream_state,
        )

    with filter_col6:
        household_size = None
        if not ignore_household_size:
            household_size = st.number_input(
                "Household size (including self)",
                min_value=1,
                step=1,
                value=1,
                on_change=reset_downstream_state,
            )

    inputs = ConstraintInputs(
        selected_cities=selected_cities,
        selected_industry=selected_industry,
        selected_career_stage=selected_career_stage,
        housing_preference=housing_preference,
        household_size=household_size,
        ignore_household_size=ignore_household_size,
    )

    filtered_housing_df, filtered_jobs_df = run_constraint_layer(
        housing_df=housing_df,
        jobs_df=jobs_df,
        inputs=inputs,
    )

    diagnostics = build_constraint_diagnostics(
        housing_df=housing_df,
        jobs_df=jobs_df,
        inputs=inputs,
    )

    display_constraint_warnings(diagnostics)

    metric_col1, metric_col2 = st.columns(2)

    with metric_col1:
        st.metric("Available Housing Listings", len(filtered_housing_df))

    with metric_col2:
        st.metric("Available Job Listings", len(filtered_jobs_df))

    return inputs, filtered_housing_df, filtered_jobs_df


def render_image_preference_section() -> None:
    st.header("Step 2: Select an Image from Each Pair")

    st.markdown(
        """
        Choose the image in each pair that better reflects your relocation preferences.
        These selections help the recommendation engine infer your lifestyle priorities.
        """
    )

    if "selected_images" not in st.session_state:
        st.session_state.selected_images = {}

    for pair in IMAGE_PAIRS:
        with st.container():
            st.subheader(f"Preference Pair {pair['pair_id'].replace('pair_', '')}")

            col1, col2 = st.columns(2)

            with col1:
                st.image(pair["left"]["image_path"], width="stretch")
                st.caption(pair["left"]["description"])

            with col2:
                st.image(pair["right"]["image_path"], width="stretch")
                st.caption(pair["right"]["description"])

            selection = st.radio(
                "Which option better matches your preference?",
                options=["Left", "Right"],
                index=None,
                key=pair["pair_id"],
                horizontal=True,
            )

            previous_selection = st.session_state.selected_images.get(pair["pair_id"])

            if selection == "Left":
                if previous_selection != pair["left"]:
                    reset_downstream_state()
                st.session_state.selected_images[pair["pair_id"]] = pair["left"]

            elif selection == "Right":
                if previous_selection != pair["right"]:
                    reset_downstream_state()
                st.session_state.selected_images[pair["pair_id"]] = pair["right"]

            st.divider()

    selected_count = len(st.session_state.selected_images)
    total_count = len(IMAGE_PAIRS)

    st.progress(selected_count / total_count)
    st.write(f"Selections completed: {selected_count} / {total_count}")


def maybe_auto_run_engine(
    filtered_housing_df: pd.DataFrame,
    filtered_jobs_df: pd.DataFrame,
    inputs: ConstraintInputs,
) -> None:
    selected_count = len(st.session_state.get("selected_images", {}))
    total_count = len(IMAGE_PAIRS)

    if selected_count < total_count:
        st.info("Complete all image selections to prepare personalized recommendation candidates.")
        return

    current_signature = build_engine_signature(inputs)
    previous_signature = st.session_state.get("engine_signature")

    if previous_signature == current_signature and st.session_state.get("recommendations_ready"):
        st.success("Recommendation candidates are ready for personalized rating.")
        return

    with st.spinner("Preparing personalized recommendation candidates..."):
        build_engine_outputs(
            filtered_housing_df=filtered_housing_df,
            filtered_jobs_df=filtered_jobs_df,
            inputs=inputs,
        )

    if st.session_state.get("recommendations_ready"):
        st.success("Recommendation candidates are ready for personalized rating.")


def render_rating_section() -> None:
    st.header("Step 3: Personalized Ratings")

    st.markdown(
        """
        Rate the jobs, housing options, and job-housing pairings below.
        Your ratings create a user-defined baseline that the system uses to evaluate
        whether the weighted logic model or fuzzy inference model better matches your preferences.
        """
    )

    if "layer4_comparison_df" not in st.session_state:
        st.info("Complete Step 2 first. Recommendation candidates will be prepared automatically.")
        return

    comparison_df = st.session_state["layer4_comparison_df"]

    st.subheader("Preference Importance")

    st.info(
        "Use these sliders to indicate how much each rating category matters to you."
    )

    importance_col1, importance_col2, importance_col3, importance_col4, importance_col5 = st.columns(5)

    with importance_col1:
        job_importance = st.slider("Job", 1, 5, 4)

    with importance_col2:
        housing_importance = st.slider("Housing", 1, 5, 4)

    with importance_col3:
        commute_importance = st.slider("Commute", 1, 5, 3)

    with importance_col4:
        lifestyle_importance = st.slider("Lifestyle", 1, 5, 3)

    with importance_col5:
        overall_importance = st.slider("Overall", 1, 5, 5)

    importance_weights = {
        "user_job_fit": job_importance,
        "user_housing_fit": housing_importance,
        "user_commute_fit": commute_importance,
        "user_lifestyle_fit": lifestyle_importance,
        "user_overall_fit": overall_importance,
    }

    st.markdown("---")
    st.subheader("Rate Jobs")

    job_ratings_by_job_id = {}
    unique_jobs_df = comparison_df.drop_duplicates(subset=["job_id"])

    for _, row in unique_jobs_df.iterrows():
        job_id = str(row["job_id"])
        job_row = row.get("job_row", {}) or {}

        with st.expander(
            f"{job_row.get('raw_job_title', 'Unknown Job')} | {job_row.get('employer', 'Unknown Employer')}",
            expanded=False,
        ):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.write("**Industry:**", job_row.get("industry", "N/A"))
                st.write("**Career Stage:**", job_row.get("career_stage", "N/A"))

            with col2:
                st.write("**Region:**", job_row.get("region_id", "N/A"))
                st.write("**Salary:**", f"${job_row.get('annual_salary', 0):,.0f}")

            with col3:
                job_fit = st.slider(
                    "Job Fit",
                    min_value=1,
                    max_value=5,
                    value=3,
                    key=f"{job_id}_user_job_fit",
                )

            job_ratings_by_job_id[job_id] = {
                "user_job_fit": job_fit,
            }

    st.markdown("---")
    st.subheader("Rate Housing and Neighborhood Fit")

    housing_ratings_by_home_id = {}
    unique_homes_df = comparison_df.drop_duplicates(subset=["home_id"])

    for _, row in unique_homes_df.iterrows():
        home_id = str(row["home_id"])
        housing_row = row.get("housing_row", {}) or {}

        city = housing_row.get("city", "N/A")
        state = housing_row.get("state", "N/A")
        rent = housing_row.get("housing_cost", 0)

        with st.expander(
            f"{home_id} | {city}, {state} | ${rent:,.0f}/month",
            expanded=False,
        ):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.write("**Property Type:**", housing_row.get("property_type", "N/A"))
                st.write("**Beds:**", housing_row.get("beds", "N/A"))
                st.write("**Baths:**", housing_row.get("baths", "N/A"))
                st.write("**Sqft:**", housing_row.get("sqft", "N/A"))

            with col2:
                st.write("**Parks within 1 mile:**", housing_row.get("parks_within_1mi", "N/A"))
                st.write("**Restaurants within 2 miles:**", housing_row.get("restaurants_within_2mi", "N/A"))
                st.write("**Grocery stores within 2 miles:**", housing_row.get("grocery_stores_within_2mi", "N/A"))
                st.write("**Schools within 2 miles:**", housing_row.get("schools_within_2mi", "N/A"))

            with col3:
                housing_fit = st.slider(
                    "Housing Fit",
                    min_value=1,
                    max_value=5,
                    value=3,
                    key=f"{home_id}_user_housing_fit",
                )

                lifestyle_fit = st.slider(
                    "Lifestyle / Neighborhood Fit",
                    min_value=1,
                    max_value=5,
                    value=3,
                    key=f"{home_id}_user_lifestyle_fit",
                )

            housing_ratings_by_home_id[home_id] = {
                "user_housing_fit": housing_fit,
                "user_lifestyle_fit": lifestyle_fit,
            }

    st.markdown("---")
    st.subheader("Rate Job-Housing Pairings")

    pair_ratings_by_pair_id = {}

    for _, row in comparison_df.iterrows():
        pair_id = str(row["pair_id"])
        home_id = row.get("home_id")
        job_id = row.get("job_id")

        housing_row = row.get("housing_row", {}) or {}
        job_row = row.get("job_row", {}) or {}
        pair_features = row.get("pair_feature_vector", {}) or {}

        job_title = job_row.get("raw_job_title", "Unknown Job")
        employer = job_row.get("employer", "Unknown Employer")
        city = housing_row.get("city", "N/A")
        state = housing_row.get("state", "N/A")

        with st.expander(
            f"{job_title} at {employer} + Home {home_id} in {city}, {state}",
            expanded=False,
        ):
            pair_col1, pair_col2, pair_col3 = st.columns(3)

            with pair_col1:
                st.markdown("#### Job")
                st.write("**Job ID:**", job_id)
                st.write("**Title:**", job_title)
                st.write("**Employer:**", employer)
                st.write("**Salary:**", f"${job_row.get('annual_salary', 0):,.0f}")

            with pair_col2:
                st.markdown("#### Housing")
                st.write("**Home ID:**", home_id)
                st.write("**Rent:**", f"${housing_row.get('housing_cost', 0):,.0f}/month")
                st.write("**Beds:**", housing_row.get("beds", "N/A"))
                st.write("**Baths:**", housing_row.get("baths", "N/A"))

            with pair_col3:
                st.markdown("#### Commute")
                st.write(
                    "**Commute Distance:**",
                    f"{pair_features.get('commute_distance_miles', 0):.2f} mi",
                )

                commute_fit = st.slider(
                    "Commute Fit",
                    min_value=1,
                    max_value=5,
                    value=3,
                    key=f"{pair_id}_user_commute_fit",
                )

                overall_fit = st.slider(
                    "Overall Pair Fit",
                    min_value=1,
                    max_value=5,
                    value=3,
                    key=f"{pair_id}_user_overall_fit",
                )

            pair_ratings_by_pair_id[pair_id] = {
                "user_commute_fit": commute_fit,
                "user_overall_fit": overall_fit,
            }

    if st.button("Evaluate My Preferences", type="primary"):
        rated_df = add_user_ratings(
            comparison_df=comparison_df,
            job_ratings_by_job_id=job_ratings_by_job_id,
            housing_ratings_by_home_id=housing_ratings_by_home_id,
            pair_ratings_by_pair_id=pair_ratings_by_pair_id,
            importance_weights=importance_weights,
        )

        evaluation_result = evaluate_model_alignment(rated_df)
        st.session_state["layer4_evaluation_result"] = evaluation_result

        st.success("Your ratings have been evaluated. Final recommendations are ready.")


def build_export_summary_df(
    ranked_df: pd.DataFrame,
    model_summary_df: pd.DataFrame,
    eval_res: dict,
    model_used: str,
) -> pd.DataFrame:
    export_rows = []

    dimension_summary_df = eval_res.get("dimension_alignment_summary", pd.DataFrame())

    dimension_metrics = {}

    if dimension_summary_df is not None and not dimension_summary_df.empty:
        for _, dimension_row in dimension_summary_df.iterrows():
            dimension_name = str(dimension_row.get("dimension", "")).strip()

            if not dimension_name:
                continue

            dimension_metrics[f"{dimension_name}_weighted_logic_mae"] = dimension_row.get("layer3a_mae", "N/A")
            dimension_metrics[f"{dimension_name}_fuzzy_logic_mae"] = dimension_row.get("layer3b_mae", "N/A")
            dimension_metrics[f"{dimension_name}_better_aligned_model"] = dimension_row.get("better_aligned_model", "N/A")

    for rank_index, (_, row) in enumerate(ranked_df.iterrows(), start=1):
        export_row = {
            "rank": rank_index,
            "pair_id": row.get("pair_id", "N/A"),
            "home_id": row.get("home_id", "N/A"),
            "job_id": row.get("job_id", "N/A"),
            "job_title": safe_get_nested(row, "job_row", "raw_job_title"),
            "employer": safe_get_nested(row, "job_row", "employer"),
            "industry": safe_get_nested(row, "job_row", "industry"),
            "career_stage": safe_get_nested(row, "job_row", "career_stage"),
            "annual_salary": safe_get_nested(row, "job_row", "annual_salary"),
            "city": safe_get_nested(row, "housing_row", "city"),
            "state": safe_get_nested(row, "housing_row", "state"),
            "housing_cost": safe_get_nested(row, "housing_row", "housing_cost"),
            "beds": safe_get_nested(row, "housing_row", "beds"),
            "baths": safe_get_nested(row, "housing_row", "baths"),
            "sqft": safe_get_nested(row, "housing_row", "sqft"),
            "property_type": safe_get_nested(row, "housing_row", "property_type"),
            "commute_distance_miles": safe_get_nested(
                row,
                "pair_feature_vector",
                "commute_distance_miles",
            ),
            "affordability_score": safe_get_nested(
                row,
                "pair_feature_vector",
                "affordability_score",
            ),
            "commute_score": safe_get_nested(
                row,
                "pair_feature_vector",
                "commute_score",
            ),
            "amenity_score": safe_get_nested(
                row,
                "pair_feature_vector",
                "amenity_score",
            ),
            "space_fit_score": safe_get_nested(
                row,
                "pair_feature_vector",
                "space_fit_score",
            ),
            "weighted_model_score": row.get("final_score", "N/A"),
            "fuzzy_model_score": row.get("layer3b_fuzzy_score", "N/A"),
            "fuzzy_label": row.get("layer3b_fuzzy_label", "N/A"),
            "user_alignment_score": row.get("user_alignment_score", "N/A"),
            "user_job_fit": row.get("user_job_fit", "N/A"),
            "user_housing_fit": row.get("user_housing_fit", "N/A"),
            "user_lifestyle_fit": row.get("user_lifestyle_fit", "N/A"),
            "user_commute_fit": row.get("user_commute_fit", "N/A"),
            "user_overall_fit": row.get("user_overall_fit", "N/A"),
            "weighted_logic_abs_error": row.get("layer3a_abs_error", "N/A"),
            "fuzzy_logic_abs_error": row.get("layer3b_abs_error", "N/A"),
            "model_used_for_final_ranking": model_used,
            "weighted_logic_mean_absolute_error": eval_res.get("layer3a_mean_absolute_error", "N/A"),
            "fuzzy_logic_mean_absolute_error": eval_res.get("layer3b_mean_absolute_error", "N/A"),
            "better_aligned_model_overall": eval_res.get("better_aligned_model", "N/A"),
            "weighted_logic_dimension_wins": eval_res.get("layer3a_dimension_wins", "N/A"),
            "fuzzy_logic_dimension_wins": eval_res.get("layer3b_dimension_wins", "N/A"),
            "dimension_ties": eval_res.get("dimension_ties", "N/A"),
        }

        export_row.update(dimension_metrics)
        export_rows.append(export_row)

    export_df = pd.DataFrame(export_rows)

    if not model_summary_df.empty:
        for _, summary_row in model_summary_df.iterrows():
            metric_name = str(summary_row.get("Metric", "")).strip()
            metric_value = summary_row.get("Value", "")

            if metric_name:
                export_df[f"summary_{metric_name}"] = metric_value

    return export_df


def render_results_section() -> None:
    st.header("Step 4: Final Recommendations")

    if "layer4_evaluation_result" not in st.session_state:
        st.info("Complete Step 3 to generate your final ranked recommendations.")
        return

    if "layer4_comparison_df" not in st.session_state:
        st.warning("Recommendation candidates are missing. Please complete Step 2 again.")
        return

    comparison_df = st.session_state["layer4_comparison_df"]
    eval_res = st.session_state["layer4_evaluation_result"]

    rated_results_df = eval_res.get("rated_results", comparison_df)

    layer3a_mae = eval_res["layer3a_mean_absolute_error"]
    layer3b_mae = eval_res["layer3b_mean_absolute_error"]

    if layer3a_mae < layer3b_mae:
        st.success("The weighted logic model best matched your ratings.")
    elif layer3b_mae < layer3a_mae:
        st.success("The fuzzy inference model best matched your ratings.")
    else:
        st.info("Both models matched your ratings equally well.")

    render_layer4_diagnostics(eval_res)

    top_n_layer5 = st.slider(
        "Move slider to adjust number of displayed recommendations",
        min_value=1,
        max_value=10,
        value=5,
        key="layer5_top_n",
    )

    ranking_method = determine_final_ranking_method(
        layer3a_mae=layer3a_mae,
        layer3b_mae=layer3b_mae,
    )

    ranked_recommendations, model_used = build_final_rankings(
        scored_pairs_df=rated_results_df,
        ranking_method=ranking_method,
        top_n=top_n_layer5,
    )

    render_layer5_recommendations(
        ranked_df=ranked_recommendations,
        model_used=model_used,
    )

    summary_df = build_model_comparison_summary(
        layer3a_mae=layer3a_mae,
        layer3b_mae=layer3b_mae,
        ranking_method=ranking_method,
        model_used=model_used,
    )

    render_model_comparison_summary(summary_df)

    export_df = build_export_summary_df(
        ranked_df=ranked_recommendations,
        model_summary_df=summary_df,
        eval_res=eval_res,
        model_used=model_used,
    )

    csv_data = export_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Recommendation Summary as CSV",
        data=csv_data,
        file_name="relocation_recommendation_summary.csv",
        mime="text/csv",
    )


def main():
    st.title("Relocation Recommendation Engine")

    housing_df, jobs_df = load_data()

    inputs, filtered_housing_df, filtered_jobs_df = render_filter_section(
        housing_df=housing_df,
        jobs_df=jobs_df,
    )

    if len(filtered_housing_df) == 0 or len(filtered_jobs_df) == 0:
        st.stop()

    st.markdown("---")
    render_image_preference_section()

    maybe_auto_run_engine(
        filtered_housing_df=filtered_housing_df,
        filtered_jobs_df=filtered_jobs_df,
        inputs=inputs,
    )

    st.markdown("---")
    render_rating_section()

    st.markdown("---")
    render_results_section()


if __name__ == "__main__":
    main()

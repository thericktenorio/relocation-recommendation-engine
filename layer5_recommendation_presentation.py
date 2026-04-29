from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd
import streamlit as st


TIE_THRESHOLD = 0.01


def determine_final_ranking_method(
    layer3a_mae: Optional[float],
    layer3b_mae: Optional[float],
) -> str:
    """
    Determines which model should drive the final Layer 5 ranking.

    Lower MAE means better alignment with user judgment.
    """

    if layer3a_mae is None or layer3b_mae is None:
        return "hybrid"

    mae_difference = abs(layer3a_mae - layer3b_mae)

    if mae_difference <= TIE_THRESHOLD:
        return "hybrid"

    if layer3a_mae < layer3b_mae:
        return "weighted"

    return "fuzzy"


def apply_final_ranking_score(
    scored_pairs_df: pd.DataFrame,
    ranking_method: str,
) -> Tuple[pd.DataFrame, str]:
    """
    Adds final_recommendation_score based on the selected ranking method.
    """

    df = scored_pairs_df.copy()

    if ranking_method == "weighted":
        df["final_recommendation_score"] = df["final_score"]
        model_used = "Layer 3A: Weighted Logic"

    elif ranking_method == "fuzzy":
        df["final_recommendation_score"] = df["layer3b_fuzzy_score"]
        model_used = "Layer 3B: Fuzzy Inference"

    else:
        df["final_recommendation_score"] = (df["final_score"] + df["layer3b_fuzzy_score"]) / 2
        model_used = "Hybrid Average: Weighted Logic + Fuzzy Inference"

    return df, model_used


def flatten_recommendation_rows(df: pd.DataFrame) -> pd.DataFrame:
    flattened_rows = []

    for _, row in df.iterrows():
        row_dict = row.to_dict()

        housing_row = row_dict.get("housing_row", {}) or {}
        job_row = row_dict.get("job_row", {}) or {}
        pair_features = row_dict.get("pair_feature_vector", {}) or {}
        dimension_scores = row_dict.get("dimension_scores", {}) or {}

        flattened = {
            **row_dict,

            # Housing display fields
            "housing_cost": housing_row.get("housing_cost"),
            "beds": housing_row.get("beds"),
            "baths": housing_row.get("baths"),
            "sqft": housing_row.get("sqft"),
            "city": housing_row.get("city"),
            "state": housing_row.get("state"),
            "property_type": housing_row.get("property_type"),
            "location_type": housing_row.get("location_type"),

            # Job display fields
            "raw_job_title": job_row.get("raw_job_title"),
            "employer": job_row.get("employer"),
            "annual_salary": job_row.get("annual_salary"),
            "industry": job_row.get("industry"),
            "career_stage": job_row.get("career_stage"),

            # Pair metrics
            "dist_nearest_job": pair_features.get("commute_distance_miles"),
            "affordability_score": pair_features.get("affordability_score"),
            "commute_score": pair_features.get("commute_score"),
            "amenity_score": pair_features.get("amenity_score"),
            "opportunity_score": pair_features.get("opportunity_score"),
            "space_fit_score": pair_features.get("space_fit_score"),

            # Layer 3A dimension scores
            "economic_score": dimension_scores.get("economic_score"),
            "lifestyle_score": dimension_scores.get("lifestyle_score"),
            "space_score": dimension_scores.get("space_score"),

            # Fuzzy display field
            "fuzzy_label": row_dict.get("layer3b_fuzzy_label"),
        }

        flattened_rows.append(flattened)

    return pd.DataFrame(flattened_rows)


def build_final_rankings(
    scored_pairs_df: pd.DataFrame,
    ranking_method: str,
    top_n: int = 5,
) -> Tuple[pd.DataFrame, str]:
    """
    Builds the final ranked recommendation list.
    """

    scored_pairs_df = flatten_recommendation_rows(scored_pairs_df)

    ranked_df, model_used = apply_final_ranking_score(
        scored_pairs_df=scored_pairs_df,
        ranking_method=ranking_method,
    )

    ranked_df = ranked_df.sort_values(
        by="final_recommendation_score",
        ascending=False,
    ).reset_index(drop=True)

    ranked_df["final_rank"] = ranked_df.index + 1

    return ranked_df.head(top_n), model_used


def generate_recommendation_explanation(row: pd.Series, model_used: str) -> str:
    """
    Generates a simple explanation based on existing dimension scores.
    """

    reasons = []

    affordability_score = row.get("affordability_score", 0)
    commute_score = row.get("commute_score", 0)
    amenity_score = row.get("amenity_score", 0)
    opportunity_score = row.get("opportunity_score", 0)
    space_fit_score = row.get("space_fit_score", 0)

    if affordability_score >= 0.75:
        reasons.append(f"strong affordability ({affordability_score:.2f})")

    if commute_score >= 0.75:
        reasons.append(f"a short commute fit ({commute_score:.2f})")

    if amenity_score >= 0.70:
        reasons.append(f"good access to nearby amenities ({amenity_score:.2f})")

    if opportunity_score >= 0.70:
        reasons.append(f"strong nearby job opportunity ({opportunity_score:.2f})")

    if space_fit_score >= 0.70:
        reasons.append(f"a good housing size fit ({space_fit_score:.2f})")

    fuzzy_label = row.get("fuzzy_label", None)

    if "Fuzzy" in model_used and fuzzy_label:
        reasons.append(f"a fuzzy recommendation label of '{fuzzy_label}'")

    if not reasons:
        return (
            "This recommendation is suggested because it performs consistently "
            "across the selected ranking criteria."
        )

    if len(reasons) == 1:
        reason_text = reasons[0]
    else:
        reason_text = ", ".join(reasons[:-1]) + f", and {reasons[-1]}"

    return f"This recommendation scores highly due to {reason_text}."


def render_layer5_recommendations(
    ranked_df: pd.DataFrame,
    model_used: str,
) -> None:
    """
    Renders final recommendations in Streamlit.
    """

    st.header("Layer 5: Final Ranked Recommendations")

    st.write(
        "Layer 5 presents the final job-house recommendations using the "
        f"selected ranking method: **{model_used}**."
    )

    for _, row in ranked_df.iterrows():
        rank = int(row["final_rank"])
        final_score = row.get("final_recommendation_score", 0)

        with st.expander(
            f"#{rank} Recommendation | Final Score: {final_score:.3f}",
            expanded=(rank == 1),
        ):
            job_col, housing_col = st.columns(2)

            with job_col:
                st.subheader("Job Details")
                st.write(f"**Title:** {row.get('raw_job_title', 'N/A')}")
                st.write(f"**Employer:** {row.get('employer', 'N/A')}")
                st.write(f"**Salary:** ${row.get('annual_salary', 0):,.0f}")
                st.write(f"**Industry:** {row.get('industry', 'N/A')}")
                st.write(f"**Career Stage:** {row.get('career_stage', 'N/A')}")

            with housing_col:
                st.subheader("Housing Details")
                st.write(f"**Rent:** ${row.get('housing_cost', 0):,.0f}/month")
                st.write(f"**Beds:** {row.get('beds', 'N/A')}")
                st.write(f"**Baths:** {row.get('baths', 'N/A')}")
                st.write(f"**Sqft:** {row.get('sqft', 'N/A')}")
                st.write(f"**Location:** {row.get('city', 'N/A')}, {row.get('state', 'N/A')}")
                st.write(f"**Property Type:** {row.get('property_type', 'N/A')}")

            st.markdown("---")

            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

            metric_col1.metric(
                "Commute Distance",
                f"{row.get('dist_nearest_job', 0):.1f} mi",
            )

            metric_col2.metric(
                "Affordability",
                f"{row.get('affordability_score', 0):.3f}",
            )

            metric_col3.metric(
                "Commute Score",
                f"{row.get('commute_score', 0):.3f}",
            )

            metric_col4.metric(
                "Final Score",
                f"{final_score:.3f}",
            )

            if "normalized_weighted_score" in row and "normalized_fuzzy_score" in row:
                score_col1, score_col2 = st.columns(2)

                score_col1.metric(
                    "Layer 3A Weighted Score",
                    f"{row.get('normalized_weighted_score', 0):.3f}",
                )

                score_col2.metric(
                    "Layer 3B Fuzzy Score",
                    f"{row.get('normalized_fuzzy_score', 0):.3f}",
                )

            if row.get("fuzzy_label", None):
                st.write(f"**Fuzzy Label:** {row.get('fuzzy_label')}")

            explanation = generate_recommendation_explanation(row, model_used)
            st.info(explanation)


def build_model_comparison_summary(
    layer3a_mae: Optional[float],
    layer3b_mae: Optional[float],
    ranking_method: str,
    model_used: str,
) -> pd.DataFrame:
    """
    Builds the final model comparison summary table.
    """

    if layer3a_mae is None or layer3b_mae is None:
        better_model = "Not enough evaluation data"
    elif ranking_method == "weighted":
        better_model = "Layer 3A: Weighted Logic"
    elif ranking_method == "fuzzy":
        better_model = "Layer 3B: Fuzzy Inference"
    else:
        better_model = "Tie / Hybrid Average"

    return pd.DataFrame(
        {
            "Metric": [
                "Mean Absolute Error",
                "Final Ranking Method",
                "Selected Model",
            ],
            "Layer 3A: Weighted Logic": [
                round(layer3a_mae, 4) if layer3a_mae is not None else "N/A",
                "Used" if ranking_method == "weighted" else "Compared",
                "Yes" if ranking_method == "weighted" else "No",
            ],
            "Layer 3B: Fuzzy Inference": [
                round(layer3b_mae, 4) if layer3b_mae is not None else "N/A",
                "Used" if ranking_method == "fuzzy" else "Compared",
                "Yes" if ranking_method == "fuzzy" else "No",
            ],
            "Conclusion": [
                better_model,
                model_used,
                better_model,
            ],
        }
    )


def render_model_comparison_summary(summary_df: pd.DataFrame) -> None:
    """
    Renders final Layer 4 / Layer 5 evidence table.
    """

    st.subheader("Final Model Comparison Summary")

    st.write(
        "This table summarizes the evaluation evidence used to justify the "
        "final recommendation ranking method."
    )

    st.dataframe(summary_df, use_container_width=True)
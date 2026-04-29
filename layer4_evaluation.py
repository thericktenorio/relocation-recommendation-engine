# layer4_evaluation.py

from __future__ import annotations

from typing import Any, Dict, List, Optional
import pandas as pd


# ============================================================
# User Rating Fields
# ============================================================

JOB_RATING_FIELDS = [
    "user_job_fit",
]

HOUSING_RATING_FIELDS = [
    "user_housing_fit",
    "user_lifestyle_fit",
]

PAIR_RATING_FIELDS = [
    "user_commute_fit",
    "user_overall_fit",
]

USER_RATING_FIELDS = (
    JOB_RATING_FIELDS
    + HOUSING_RATING_FIELDS
    + PAIR_RATING_FIELDS
)


# ============================================================
# Default User Importance Weights
# ============================================================

DEFAULT_IMPORTANCE_WEIGHTS = {
    "user_job_fit": 0.20,
    "user_housing_fit": 0.20,
    "user_commute_fit": 0.15,
    "user_lifestyle_fit": 0.10,
    "user_overall_fit": 0.35,
}


# ============================================================
# Dimension Mapping
# ============================================================

DIMENSION_EVALUATION_MAP = {
    "job_fit": {
        "user_field": "user_job_fit",
        "layer3a_field": "layer3a_job_fit_score",
        "layer3b_field": "layer3b_job_fit_score",
    },
    "housing_fit": {
        "user_field": "user_housing_fit",
        "layer3a_field": "layer3a_housing_fit_score",
        "layer3b_field": "layer3b_housing_fit_score",
    },
    "lifestyle_fit": {
        "user_field": "user_lifestyle_fit",
        "layer3a_field": "layer3a_lifestyle_fit_score",
        "layer3b_field": "layer3b_lifestyle_fit_score",
    },
    "commute_fit": {
        "user_field": "user_commute_fit",
        "layer3a_field": "layer3a_commute_fit_score",
        "layer3b_field": "layer3b_commute_fit_score",
    },
    "overall_fit": {
        "user_field": "user_overall_fit",
        "layer3a_field": "layer3a_overall_fit_score",
        "layer3b_field": "layer3b_overall_fit_score",
    },
}


# ============================================================
# General Helpers
# ============================================================

def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return low

    return max(low, min(high, value))


def safe_get_nested_dict_value(
    source: Any,
    key: str,
    default: float = 0.0,
) -> float:
    """
    Safely extracts a numeric value from a dictionary-like object.
    """

    if not isinstance(source, dict):
        return default

    try:
        return float(source.get(key, default))
    except (TypeError, ValueError):
        return default


def normalize_user_rating(value: float, max_rating: float = 5.0) -> float:
    """
    Converts a user rating such as 1-5 into a 0-1 score.

    Example:
        1 -> 0.20
        3 -> 0.60
        5 -> 1.00
    """

    if value is None:
        return 0.0

    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0

    value = max(0.0, min(value, max_rating))
    return value / max_rating


def min_max_normalize(values: pd.Series) -> pd.Series:
    """
    Session-relative min-max normalization.

    If all values are identical, all values are set to 1.0 so that
    identical top scores are treated as equally strong within the
    comparison set.
    """

    min_value = values.min()
    max_value = values.max()

    if max_value == min_value:
        return pd.Series([1.0 for _ in values], index=values.index)

    return (values - min_value) / (max_value - min_value)


# ============================================================
# User Baseline Scoring
# ============================================================

def normalize_importance_weights(
    importance_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Converts user-stated importance values into normalized weights.

    Example input:
        {
            "user_job_fit": 5,
            "user_housing_fit": 4,
            "user_commute_fit": 5,
            "user_lifestyle_fit": 3,
            "user_overall_fit": 5,
        }

    Output:
        Weights that sum to 1.0.
    """

    if not importance_weights:
        return DEFAULT_IMPORTANCE_WEIGHTS.copy()

    cleaned_weights = {}

    for field in USER_RATING_FIELDS:
        try:
            cleaned_weights[field] = max(
                0.0,
                float(importance_weights.get(field, 0.0)),
            )
        except (TypeError, ValueError):
            cleaned_weights[field] = 0.0

    total_weight = sum(cleaned_weights.values())

    if total_weight == 0:
        return DEFAULT_IMPORTANCE_WEIGHTS.copy()

    return {
        field: weight / total_weight
        for field, weight in cleaned_weights.items()
    }


def compute_user_alignment_score(
    row: Dict[str, Any],
    max_rating: float = 5.0,
    importance_weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Computes a single composite user judgment score from all user
    rating dimensions.

    This value is the primary user baseline against which Layer 3A
    and Layer 3B are evaluated.
    """

    weights = normalize_importance_weights(importance_weights)

    score = 0.0

    for field, weight in weights.items():
        score += normalize_user_rating(row.get(field, 0.0), max_rating) * weight

    return clamp(score)


# ============================================================
# Layer 4 Comparison Set Construction
# ============================================================

def build_layer4_comparison_set(
    layer3a_results: List[Dict[str, Any]],
    layer3b_results: List[Dict[str, Any]],
    top_n: int = 5,
    pair_id_field: str = "pair_id",
) -> pd.DataFrame:
    """
    Builds a comparison set using the union of top recommendations
    from Layer 3A and Layer 3B.

    This avoids only evaluating one model's preferred recommendations.
    """

    layer3a_df = pd.DataFrame(layer3a_results)
    layer3b_df = pd.DataFrame(layer3b_results)

    if layer3a_df.empty or layer3b_df.empty:
        raise ValueError("Layer 3A and Layer 3B results must both contain data.")

    if pair_id_field not in layer3a_df.columns or pair_id_field not in layer3b_df.columns:
        raise ValueError(f"Both result sets must include '{pair_id_field}'.")

    if "final_score" not in layer3a_df.columns:
        raise ValueError("Layer 3A results must include 'final_score'.")

    if "layer3b_fuzzy_score" not in layer3b_df.columns:
        raise ValueError("Layer 3B results must include 'layer3b_fuzzy_score'.")

    top_3a = layer3a_df.sort_values("final_score", ascending=False).head(top_n)
    top_3b = layer3b_df.sort_values("layer3b_fuzzy_score", ascending=False).head(top_n)

    comparison_pair_ids = set(top_3a[pair_id_field]).union(
        set(top_3b[pair_id_field])
    )

    layer3b_columns_to_keep = [
        pair_id_field,
        "layer3b_fuzzy_score",
        "layer3b_fuzzy_label",
        "layer3b_explanation",
        "layer3b_adjusted_features",
        "layer3b_output_memberships",
    ]

    existing_layer3b_columns = [
        col for col in layer3b_columns_to_keep
        if col in layer3b_df.columns
    ]

    merged = pd.merge(
        layer3a_df,
        layer3b_df[existing_layer3b_columns],
        on=pair_id_field,
        how="inner",
    )

    comparison_df = merged[
        merged[pair_id_field].isin(comparison_pair_ids)
    ].copy()

    comparison_df["layer3a_rank"] = comparison_df["final_score"].rank(
        ascending=False,
        method="min",
    )

    comparison_df["layer3b_rank"] = comparison_df["layer3b_fuzzy_score"].rank(
        ascending=False,
        method="min",
    )

    comparison_df = add_model_dimension_scores(comparison_df)

    return comparison_df.sort_values(
        ["layer3a_rank", "layer3b_rank"],
        ascending=True,
    )


# ============================================================
# Add User Ratings
# ============================================================

def add_user_ratings(
    comparison_df: pd.DataFrame,
    job_ratings_by_job_id: Dict[str, Dict[str, float]],
    housing_ratings_by_home_id: Dict[str, Dict[str, float]],
    pair_ratings_by_pair_id: Dict[str, Dict[str, float]],
    importance_weights: Optional[Dict[str, float]] = None,
    pair_id_field: str = "pair_id",
    job_id_field: str = "job_id",
    home_id_field: str = "home_id",
) -> pd.DataFrame:
    """
    Adds user ratings to the comparison dataframe.

    Job ratings are applied once per unique job.
    Housing and lifestyle ratings are applied once per unique home.
    Commute and overall ratings are applied once per job-house pair.
    """

    df = comparison_df.copy()

    required_id_columns = [pair_id_field, job_id_field, home_id_field]

    for col in required_id_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required ID column: {col}")

    df["user_job_fit"] = df[job_id_field].apply(
        lambda job_id: job_ratings_by_job_id
        .get(str(job_id), {})
        .get("user_job_fit", 0.0)
    )

    df["user_housing_fit"] = df[home_id_field].apply(
        lambda home_id: housing_ratings_by_home_id
        .get(str(home_id), {})
        .get("user_housing_fit", 0.0)
    )

    df["user_lifestyle_fit"] = df[home_id_field].apply(
        lambda home_id: housing_ratings_by_home_id
        .get(str(home_id), {})
        .get("user_lifestyle_fit", 0.0)
    )

    df["user_commute_fit"] = df[pair_id_field].apply(
        lambda pair_id: pair_ratings_by_pair_id
        .get(str(pair_id), {})
        .get("user_commute_fit", 0.0)
    )

    df["user_overall_fit"] = df[pair_id_field].apply(
        lambda pair_id: pair_ratings_by_pair_id
        .get(str(pair_id), {})
        .get("user_overall_fit", 0.0)
    )

    normalized_weights = normalize_importance_weights(importance_weights)

    df["user_alignment_score"] = df.apply(
        lambda row: compute_user_alignment_score(
            row=row.to_dict(),
            importance_weights=normalized_weights,
        ),
        axis=1,
    )

    for field in USER_RATING_FIELDS:
        df[f"{field}_normalized"] = df[field].apply(normalize_user_rating)

    for field, weight in normalized_weights.items():
        df[f"{field}_importance_weight"] = weight

    return df


# ============================================================
# Model Dimension Score Mapping
# ============================================================

def extract_layer3a_dimension_inputs(row: pd.Series) -> Dict[str, float]:
    """
    Extracts model-side dimension values from Layer 3A output.

    Preferred source:
        row["dimension_scores"]

    Fallback source:
        row["pair_feature_vector"]
    """

    dimension_scores = row.get("dimension_scores", {})
    pair_features = row.get("pair_feature_vector", {})

    opportunity = safe_get_nested_dict_value(
        dimension_scores,
        "opportunity_score",
        safe_get_nested_dict_value(pair_features, "opportunity_score", 0.0),
    )

    affordability = safe_get_nested_dict_value(
        dimension_scores,
        "affordability_score",
        safe_get_nested_dict_value(pair_features, "affordability_score", 0.0),
    )

    commute = safe_get_nested_dict_value(
        dimension_scores,
        "commute_score",
        safe_get_nested_dict_value(pair_features, "commute_score", 0.0),
    )

    lifestyle = safe_get_nested_dict_value(
        dimension_scores,
        "lifestyle_score",
        safe_get_nested_dict_value(pair_features, "amenity_score", 0.0),
    )

    space = safe_get_nested_dict_value(
        dimension_scores,
        "space_score",
        safe_get_nested_dict_value(pair_features, "space_fit_score", 0.0),
    )

    return {
        "opportunity": clamp(opportunity),
        "affordability": clamp(affordability),
        "commute": clamp(commute),
        "lifestyle": clamp(lifestyle),
        "space": clamp(space),
    }


def extract_layer3b_dimension_inputs(row: pd.Series) -> Dict[str, float]:
    """
    Extracts model-side dimension values from Layer 3B output.

    Preferred source:
        row["layer3b_adjusted_features"]

    Fallback source:
        row["pair_feature_vector"]
    """

    adjusted_features = row.get("layer3b_adjusted_features", {})
    pair_features = row.get("pair_feature_vector", {})

    opportunity = safe_get_nested_dict_value(
        adjusted_features,
        "opportunity",
        safe_get_nested_dict_value(pair_features, "opportunity_score", 0.0),
    )

    affordability = safe_get_nested_dict_value(
        adjusted_features,
        "affordability",
        safe_get_nested_dict_value(pair_features, "affordability_score", 0.0),
    )

    commute = safe_get_nested_dict_value(
        adjusted_features,
        "commute",
        safe_get_nested_dict_value(pair_features, "commute_score", 0.0),
    )

    amenity = safe_get_nested_dict_value(
        adjusted_features,
        "amenity",
        safe_get_nested_dict_value(pair_features, "amenity_score", 0.0),
    )

    space_fit = safe_get_nested_dict_value(
        adjusted_features,
        "space_fit",
        safe_get_nested_dict_value(pair_features, "space_fit_score", 0.0),
    )

    return {
        "opportunity": clamp(opportunity),
        "affordability": clamp(affordability),
        "commute": clamp(commute),
        "amenity": clamp(amenity),
        "space_fit": clamp(space_fit),
    }


def add_model_dimension_scores(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds model-side scores that align with the user-side rating dimensions.

    This enables diagnostic evaluation such as:
        model job fit       vs user job fit
        model housing fit   vs user housing fit
        model lifestyle fit vs user lifestyle fit
        model commute fit   vs user commute fit
        model overall fit   vs user overall fit
    """

    df = comparison_df.copy()

    required_columns = [
        "final_score",
        "layer3b_fuzzy_score",
    ]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required model score column: {col}")

    layer3a_dimension_rows = df.apply(extract_layer3a_dimension_inputs, axis=1)
    layer3b_dimension_rows = df.apply(extract_layer3b_dimension_inputs, axis=1)

    # -----------------------------
    # Layer 3A model-side dimensions
    # -----------------------------
    df["layer3a_job_fit_score"] = layer3a_dimension_rows.apply(
        lambda values: values["opportunity"]
    )

    df["layer3a_housing_fit_score"] = layer3a_dimension_rows.apply(
        lambda values: clamp(
            0.5 * values["affordability"]
            + 0.5 * values["space"]
        )
    )

    df["layer3a_lifestyle_fit_score"] = layer3a_dimension_rows.apply(
        lambda values: values["lifestyle"]
    )

    df["layer3a_commute_fit_score"] = layer3a_dimension_rows.apply(
        lambda values: values["commute"]
    )

    df["layer3a_overall_fit_score"] = df["final_score"].apply(clamp)

    # -----------------------------
    # Layer 3B model-side dimensions
    # -----------------------------
    df["layer3b_job_fit_score"] = layer3b_dimension_rows.apply(
        lambda values: values["opportunity"]
    )

    df["layer3b_housing_fit_score"] = layer3b_dimension_rows.apply(
        lambda values: clamp(
            0.5 * values["affordability"]
            + 0.5 * values["space_fit"]
        )
    )

    df["layer3b_lifestyle_fit_score"] = layer3b_dimension_rows.apply(
        lambda values: values["amenity"]
    )

    df["layer3b_commute_fit_score"] = layer3b_dimension_rows.apply(
        lambda values: values["commute"]
    )

    df["layer3b_overall_fit_score"] = df["layer3b_fuzzy_score"].apply(clamp)

    return df


# ============================================================
# Score Normalization
# ============================================================

def normalize_layer4_model_scores(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """
    Session-relative normalization of Layer 3A and Layer 3B scores.

    Raw scores are preserved before normalization.
    """

    df = comparison_df.copy()

    if "final_score" not in df.columns:
        raise ValueError("Missing required column: final_score")

    if "layer3b_fuzzy_score" not in df.columns:
        raise ValueError("Missing required column: layer3b_fuzzy_score")

    df["raw_layer3a_score"] = df["final_score"]
    df["raw_layer3b_fuzzy_score"] = df["layer3b_fuzzy_score"]

    df["final_score"] = min_max_normalize(df["final_score"])
    df["layer3b_fuzzy_score"] = min_max_normalize(df["layer3b_fuzzy_score"])

    df = add_model_dimension_scores(df)

    return df


# ============================================================
# Dimension-Level Evaluation
# ============================================================

def evaluate_dimension_alignment(
    rated_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Computes dimension-level MAE between model-side dimension scores
    and user-side normalized ratings.

    This is diagnostic and interpretability-oriented.

    The primary evaluation metric remains the overall MAE against
    user_alignment_score.
    """

    df = rated_df.copy()

    df = add_model_dimension_scores(df)

    dimension_rows = []

    for dimension_name, mapping in DIMENSION_EVALUATION_MAP.items():
        user_field = mapping["user_field"]
        layer3a_field = mapping["layer3a_field"]
        layer3b_field = mapping["layer3b_field"]

        required_columns = [
            user_field,
            layer3a_field,
            layer3b_field,
        ]

        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column for dimension evaluation: {col}")

        user_normalized_field = f"{user_field}_normalized"

        if user_normalized_field not in df.columns:
            df[user_normalized_field] = df[user_field].apply(normalize_user_rating)

        layer3a_error_field = f"layer3a_{dimension_name}_abs_error"
        layer3b_error_field = f"layer3b_{dimension_name}_abs_error"

        df[layer3a_error_field] = abs(
            df[layer3a_field] - df[user_normalized_field]
        )

        df[layer3b_error_field] = abs(
            df[layer3b_field] - df[user_normalized_field]
        )

        layer3a_mae = df[layer3a_error_field].mean()
        layer3b_mae = df[layer3b_error_field].mean()

        if layer3a_mae < layer3b_mae:
            better_model = "Layer 3A Weighted Logic"
        elif layer3b_mae < layer3a_mae:
            better_model = "Layer 3B Fuzzy Inference"
        else:
            better_model = "Tie"

        dimension_rows.append(
            {
                "dimension": dimension_name,
                "user_rating_field": user_field,
                "layer3a_model_field": layer3a_field,
                "layer3b_model_field": layer3b_field,
                "layer3a_mae": layer3a_mae,
                "layer3b_mae": layer3b_mae,
                "better_aligned_model": better_model,
            }
        )

    dimension_summary_df = pd.DataFrame(dimension_rows)

    layer3a_wins = int(
        (dimension_summary_df["better_aligned_model"] == "Layer 3A Weighted Logic").sum()
    )

    layer3b_wins = int(
        (dimension_summary_df["better_aligned_model"] == "Layer 3B Fuzzy Inference").sum()
    )

    ties = int(
        (dimension_summary_df["better_aligned_model"] == "Tie").sum()
    )

    return {
        "dimension_alignment_summary": dimension_summary_df,
        "dimension_alignment_details": df,
        "layer3a_dimension_wins": layer3a_wins,
        "layer3b_dimension_wins": layer3b_wins,
        "dimension_ties": ties,
    }


# ============================================================
# Overall Model Alignment Evaluation
# ============================================================

def evaluate_model_alignment(
    rated_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Compares Layer 3A and Layer 3B against user ratings.

    Primary metric:
        MAE against user_alignment_score.

    Diagnostic metric:
        Dimension-level MAE against individual user rating categories.
    """

    df = rated_df.copy()

    df = add_model_dimension_scores(df)

    required_columns = [
        "final_score",
        "layer3b_fuzzy_score",
        "user_alignment_score",
    ]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df["layer3a_abs_error"] = abs(
        df["final_score"] - df["user_alignment_score"]
    )

    df["layer3b_abs_error"] = abs(
        df["layer3b_fuzzy_score"] - df["user_alignment_score"]
    )

    layer3a_mae = df["layer3a_abs_error"].mean()
    layer3b_mae = df["layer3b_abs_error"].mean()

    if layer3a_mae < layer3b_mae:
        better_model = "Layer 3A Weighted Logic"
    elif layer3b_mae < layer3a_mae:
        better_model = "Layer 3B Fuzzy Inference"
    else:
        better_model = "Tie"

    dimension_result = evaluate_dimension_alignment(df)

    return {
        "layer3a_mean_absolute_error": layer3a_mae,
        "layer3b_mean_absolute_error": layer3b_mae,
        "better_aligned_model": better_model,
        "evaluated_pair_count": len(df),
        "rated_results": dimension_result["dimension_alignment_details"],
        "dimension_alignment_summary": dimension_result["dimension_alignment_summary"],
        "layer3a_dimension_wins": dimension_result["layer3a_dimension_wins"],
        "layer3b_dimension_wins": dimension_result["layer3b_dimension_wins"],
        "dimension_ties": dimension_result["dimension_ties"],
    }


# ============================================================
# Summary Output
# ============================================================

def summarize_layer4_results(evaluation_result: Dict[str, Any]) -> str:
    """
    Creates a human-readable Layer 4 summary.

    Includes:
        - Overall MAE comparison
        - Better aligned model overall
        - Dimension-level diagnostic wins
    """

    layer3a_mae = evaluation_result["layer3a_mean_absolute_error"]
    layer3b_mae = evaluation_result["layer3b_mean_absolute_error"]
    better_model = evaluation_result["better_aligned_model"]
    pair_count = evaluation_result["evaluated_pair_count"]

    layer3a_dimension_wins = evaluation_result.get("layer3a_dimension_wins", 0)
    layer3b_dimension_wins = evaluation_result.get("layer3b_dimension_wins", 0)
    dimension_ties = evaluation_result.get("dimension_ties", 0)

    summary = (
        "Layer 4 Evaluation Summary\n"
        "--------------------------\n"
        f"Evaluated recommendation pairs: {pair_count}\n"
        f"Layer 3A Weighted Logic MAE: {layer3a_mae:.4f}\n"
        f"Layer 3B Fuzzy Inference MAE: {layer3b_mae:.4f}\n"
        f"Better aligned model overall: {better_model}\n\n"
        "Dimension-Level Diagnostic Summary\n"
        "----------------------------------\n"
        f"Layer 3A dimension wins: {layer3a_dimension_wins}\n"
        f"Layer 3B dimension wins: {layer3b_dimension_wins}\n"
        f"Dimension ties: {dimension_ties}\n"
    )

    return summary
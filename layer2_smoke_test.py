from __future__ import annotations

from typing import Any, Dict

from normalization_helpers import build_normalization_context
from exact_job_house_pair_features import build_exact_job_house_pair_features
from alignment_features import build_alignment_feature_set


def run_layer2_smoke_test(
    housing_df,
    jobs_df,
    inferred_preferences: Dict[str, float] | None = None,
    household_size: int = 1,
) -> Dict[str, Any]:
    """
    Lightweight internal test for Layer 2 feature generation.

    Uses the first housing row and first job row to verify that:
    - normalization context builds
    - pair features compute
    - alignment features compute
    """

    if inferred_preferences is None:
        inferred_preferences = {
            "urbanism_preference": 0.5,
            "frugality_preference": 0.3,
            "outdoor_preference": 0.8,
            "family_space_preference": 0.4,
        }

    if housing_df.empty or jobs_df.empty:
        return {
            "status": "error",
            "message": "Housing or jobs dataframe is empty.",
        }

    normalization_context = build_normalization_context(housing_df)

    housing_row = housing_df.iloc[0].to_dict()
    job_row = jobs_df.iloc[0].to_dict()

    pair_feature_vector = build_exact_job_house_pair_features(
        housing_row=housing_row,
        job_row=job_row,
        household_size=household_size,
        normalization_context=normalization_context,
    )

    alignment_feature_set = build_alignment_feature_set(
        housing_row=housing_row,
        pair_features=pair_feature_vector,
        inferred_preferences=inferred_preferences,
        context=normalization_context,
    )

    return {
        "status": "ok",
        "pair_feature_vector": pair_feature_vector,
        "alignment_feature_set": alignment_feature_set,
    }
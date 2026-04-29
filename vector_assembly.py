from __future__ import annotations

from typing import Any, Dict

from exact_job_house_pair_features import build_exact_job_house_pair_features
from alignment_features import build_alignment_feature_set
from user_profile import (
    build_explicit_preference_profile,
    build_user_preference_vector,
    build_weight_modifier_profile,
)


def build_layer2_representation(
    housing_row: Dict[str, Any],
    job_row: Dict[str, Any],
    preferred_cities: list[str],
    industry: str,
    career_stage: str,
    housing_preference: str,
    household_size: int,
    inferred_profile: Dict[str, float],
    normalization_context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build the full Layer 2 representation for one exact job-house pair.
    """

    pair_feature_vector = build_exact_job_house_pair_features(
        housing_row=housing_row,
        job_row=job_row,
        household_size=household_size,
        normalization_context=normalization_context,
    )

    explicit_profile = build_explicit_preference_profile(
        preferred_cities=preferred_cities,
        industry=industry,
        career_stage=career_stage,
        housing_preference=housing_preference,
        household_size=household_size,
    )

    weight_modifier_profile = build_weight_modifier_profile(
        urbanism_preference=inferred_profile.get("urbanism_preference", 0.0),
        frugality_preference=inferred_profile.get("frugality_preference", 0.0),
        outdoor_preference=inferred_profile.get("outdoor_preference", 0.0),
        family_space_preference=inferred_profile.get("family_space_preference", 0.0),
    )

    user_preference_vector = build_user_preference_vector(
        explicit_profile=explicit_profile,
        inferred_profile=inferred_profile,
        weight_modifier_profile=weight_modifier_profile,
    )

    alignment_feature_set = build_alignment_feature_set(
        housing_row=housing_row,
        pair_features=pair_feature_vector,
        inferred_preferences=inferred_profile,
        context=normalization_context,
    )

    return {
        "pair_feature_vector": pair_feature_vector,
        "user_preference_vector": user_preference_vector,
        "alignment_feature_set": alignment_feature_set,
        "weight_modifier_profile": weight_modifier_profile,
        "home_id": housing_row.get("home_id"),
        "job_id": job_row.get("job_id"),
    }

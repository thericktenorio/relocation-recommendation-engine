from __future__ import annotations

from typing import Dict, List


def build_explicit_preference_profile(
    preferred_cities: List[str],
    industry: str,
    career_stage: str,
    housing_preference: str,
    household_size: int,
) -> Dict:
    return {
        "preferred_cities": preferred_cities,
        "industry": industry,
        "career_stage": career_stage,
        "housing_preference": housing_preference,
        "household_size": household_size,
    }


def build_weight_modifier_profile(
    urbanism_preference: float,
    frugality_preference: float,
    outdoor_preference: float,
    family_space_preference: float,
) -> Dict[str, float]:
    """
    Baseline modifiers for Layer 3 weighting.

    Values are centered around 1.0.
    > 1.0 increases importance
    < 1.0 decreases importance
    """
    return {
        "opportunity_weight_modifier": 0.9 + 0.2 * urbanism_preference,
        "affordability_weight_modifier": 0.9 + 0.2 * frugality_preference,
        "commute_weight_modifier": 1.0,
        "amenity_weight_modifier": 0.9 + 0.1 * urbanism_preference + 0.1 * outdoor_preference,
        "space_weight_modifier": 0.9 + 0.2 * family_space_preference,
    }


def build_user_preference_vector(
    explicit_profile: Dict,
    inferred_profile: Dict[str, float],
    weight_modifier_profile: Dict[str, float],
) -> Dict:
    return {
        "explicit_profile": explicit_profile,
        "inferred_profile": inferred_profile,
        "weight_modifier_profile": weight_modifier_profile,
    }
from __future__ import annotations

from typing import Any, Dict, List

from normalization_helpers import build_normalization_context
from vector_assembly import build_layer2_representation
from exact_job_house_pair_features import build_exact_job_house_pair_features


def build_layer2_pair_representations(
    candidate_pairs: List[Dict[str, Any]],
    housing_df,
    preferred_cities: list[str],
    industry: str,
    career_stage: str,
    housing_preference: str,
    household_size: int,
    inferred_profile: Dict[str, float],
) -> List[Dict[str, Any]]:
    """
    Convert raw candidate pairs into full Layer 2 representations.

    Each representation includes:
    - Layer 2 representation output
    - raw housing/job pair identifiers
    - pair_feature_vector for Layer 3 scoring
    """
    if not candidate_pairs:
        return []

    normalization_context = build_normalization_context(housing_df)

    representations: List[Dict[str, Any]] = []

    for pair in candidate_pairs:
        housing_row = pair["housing_row"]
        job_row = pair["job_row"]

        pair_feature_vector = build_exact_job_house_pair_features(
            housing_row=housing_row,
            job_row=job_row,
            household_size=household_size,
            normalization_context=normalization_context,
        )

        layer2_output = build_layer2_representation(
            housing_row=housing_row,
            job_row=job_row,
            preferred_cities=preferred_cities,
            industry=industry,
            career_stage=career_stage,
            housing_preference=housing_preference,
            household_size=household_size,
            inferred_profile=inferred_profile,
            normalization_context=normalization_context,
        )

        representation = {
            **layer2_output,
            "pair_feature_vector": pair_feature_vector,
            "home_id": housing_row.get("home_id"),
            "job_id": job_row.get("job_id"),
            "housing_row": housing_row,
            "job_row": job_row,
        }

        representations.append(representation)

    return representations
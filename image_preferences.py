from __future__ import annotations

from typing import Dict


PREFERENCE_KEYS = [
    "urbanism_preference",
    "frugality_preference",
    "outdoor_preference",
    "family_space_preference",
]


def initialize_image_preference_scores() -> Dict[str, float]:
    return {key: 0.0 for key in PREFERENCE_KEYS}


def apply_image_choice(
    current_scores: Dict[str, float],
    image_metadata: Dict[str, float],
) -> Dict[str, float]:
    """
    Add the selected image's contribution scores into the running total.

    Example image_metadata:
    {
        "urbanism_preference": 2,
        "frugality_preference": 0,
        "outdoor_preference": 1,
        "family_space_preference": 0,
    }
    """
    updated_scores = current_scores.copy()

    for key in PREFERENCE_KEYS:
        updated_scores[key] = updated_scores.get(key, 0.0) + image_metadata.get(key, 0.0)

    return updated_scores


def normalize_image_preference_scores(
    raw_scores: Dict[str, float],
) -> Dict[str, float]:
    """
    Normalize inferred preference scores into [0, 1].

    Simple max-based normalization:
    - if all scores are zero, return zeros
    - otherwise divide each score by the maximum observed score
    """
    if not raw_scores:
        return {key: 0.0 for key in PREFERENCE_KEYS}

    max_score = max(raw_scores.values()) if raw_scores.values() else 0.0

    if max_score <= 0:
        return {key: 0.0 for key in PREFERENCE_KEYS}

    return {
        key: float(raw_scores.get(key, 0.0)) / float(max_score)
        for key in PREFERENCE_KEYS
    }
from typing import List, Dict, Any


# -----------------------------
# 1. Weight Transformation
# -----------------------------
def transform_preferences_to_weights(pref_vector: Dict[str, float]) -> Dict[str, float]:
    """
    Convert [-1, 1] preference vector into [0, 1] weights.
    """
    return {
        "economic": (pref_vector.get("economic_practicality", 0) + 1) / 2,
        "leisure": (pref_vector.get("leisure_preference", 0) + 1) / 2,
        "commute": (pref_vector.get("commute_affinity", 0) + 1) / 2,
        "family": (pref_vector.get("family_friendliness", 0) + 1) / 2,
    }


# -----------------------------
# 2. Dimension Scoring
# -----------------------------
def compute_dimension_scores(pair_features: Dict[str, Any]) -> Dict[str, float]:
    """
    Build interpretable Layer 3 dimension scores from the exact job-house pair feature vector.
    All main scores are expected to be normalized to [0, 1].
    """

    opportunity_score = pair_features.get("opportunity_score", 0.5)
    affordability_score = pair_features.get("affordability_score", 0.5)
    commute_score = pair_features.get("commute_score", 0.5)
    amenity_score = pair_features.get("amenity_score", 0.5)
    space_fit_score = pair_features.get("space_fit_score", 0.5)

    economic_score = (
        0.5 * opportunity_score
        + 0.5 * affordability_score
    )

    return {
        "economic_score": economic_score,
        "opportunity_score": opportunity_score,
        "affordability_score": affordability_score,
        "commute_score": commute_score,
        "lifestyle_score": amenity_score,
        "space_score": space_fit_score,
    }


# -----------------------------
# 3. Final Weighted Score
# -----------------------------
def compute_weighted_score(
    dimension_scores: Dict[str, float],
    weights: Dict[str, float]
) -> float:
    weighted_sum = (
        weights["economic"] * dimension_scores["economic_score"]
        + weights["leisure"] * dimension_scores["lifestyle_score"]
        + weights["commute"] * dimension_scores["commute_score"]
        + weights["family"] * dimension_scores["space_score"]
    )

    total_weight = (
        weights["economic"]
        + weights["leisure"]
        + weights["commute"]
        + weights["family"]
    )

    if total_weight == 0:
        return 0.0

    return weighted_sum / total_weight


# -----------------------------
# 4. Main Layer 3 Pipeline
# -----------------------------
def run_weighted_inference(
    pair_representations: List[Dict[str, Any]],
    preference_vector: Dict[str, float]
) -> List[Dict[str, Any]]:
    """
    Main entry point for Layer 3 Weighted Logic Inference.
    """

    weights = transform_preferences_to_weights(preference_vector)

    scored_pairs = []

    for pair in pair_representations:

        pair_features = pair.get("pair_feature_vector", {})

        dimension_scores = compute_dimension_scores(pair_features)

        # -----------------------------
        # Weighted score (normalized)
        # -----------------------------
        weighted_sum = (
            weights["economic"] * dimension_scores["economic_score"]
            + weights["leisure"] * dimension_scores["lifestyle_score"]
            + weights["commute"] * dimension_scores["commute_score"]
            + weights["family"] * dimension_scores["space_score"]
        )

        total_weight = (
            weights["economic"]
            + weights["leisure"]
            + weights["commute"]
            + weights["family"]
        )

        if total_weight == 0:
            final_score = 0.0
        else:
            final_score = weighted_sum / total_weight

        enriched_pair = {
            **pair,
            "dimension_scores": dimension_scores,
            "weights": weights,
            "final_score": final_score
        }

        scored_pairs.append(enriched_pair)

    # -----------------------------
    # Remove duplicate pairs
    # -----------------------------
    seen_pairs = set()
    unique_scored_pairs = []

    for pair in scored_pairs:
        pair_key = (pair.get("home_id"), pair.get("job_id"))

        if pair_key in seen_pairs:
            continue

        seen_pairs.add(pair_key)
        unique_scored_pairs.append(pair)

    # -----------------------------
    # Sort descending
    # -----------------------------
    unique_scored_pairs.sort(
        key=lambda x: x["final_score"],
        reverse=True
    )

    return unique_scored_pairs

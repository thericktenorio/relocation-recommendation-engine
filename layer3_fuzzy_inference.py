from __future__ import annotations

from typing import Any, Dict, List, Optional


# ------------------------------------------------------------
# Basic fuzzy membership helpers
# ------------------------------------------------------------

def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def low_membership(x: float) -> float:
    x = clamp(x)
    if x <= 0.25:
        return 1.0
    if x >= 0.50:
        return 0.0
    return (0.50 - x) / 0.25


def medium_membership(x: float) -> float:
    x = clamp(x)
    if x <= 0.25 or x >= 0.75:
        return 0.0
    if x == 0.50:
        return 1.0
    if x < 0.50:
        return (x - 0.25) / 0.25
    return (0.75 - x) / 0.25


def high_membership(x: float) -> float:
    x = clamp(x)
    if x <= 0.50:
        return 0.0
    if x >= 0.75:
        return 1.0
    return (x - 0.50) / 0.25


def fuzzify_score(x: float) -> Dict[str, float]:
    return {
        "low": low_membership(x),
        "medium": medium_membership(x),
        "high": high_membership(x),
    }


# ------------------------------------------------------------
# Extraction helpers
# ------------------------------------------------------------

def get_pair_feature_vector(pair: Dict[str, Any]) -> Dict[str, Any]:
    """
    Supports either:
    1. pair["pair_feature_vector"]
    2. pair["feature_vector"]
    3. flat pair dictionary containing the feature keys directly
    """
    if "pair_feature_vector" in pair:
        return pair["pair_feature_vector"]

    if "feature_vector" in pair:
        return pair["feature_vector"]

    return pair


def get_feature(features: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(features.get(key, default))
    except (TypeError, ValueError):
        return default


def normalize_signed_preference(value: Any, default: float = 0.0) -> float:
    """
    Convert a Layer 2b preference value from [-1, 1] into [0, 1].

    Layer 2b image comparator output:
    - -1.0 = strong negative preference
    -  0.0 = neutral preference
    - +1.0 = strong positive preference

    Layer 3B adjustment logic expects:
    - 0.0 = lowest preference influence
    - 0.5 = neutral
    - 1.0 = highest preference influence
    """
    try:
        signed_value = float(value)
    except (TypeError, ValueError):
        signed_value = default

    signed_value = max(-1.0, min(1.0, signed_value))
    return (signed_value + 1.0) / 2.0


# ------------------------------------------------------------
# Preference adjustment
# ------------------------------------------------------------

def apply_preference_adjustments(
    features: Dict[str, Any],
    preference_vector: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Uses the Layer 2b image preference vector as a light adjustment layer.

    The Layer 2b vector is expected to be normalized to [-1, 1].
    This function converts each preference to [0, 1] before applying
    the multiplicative adjustment.

    Adjustment factor:
    - strong negative preference: 0.90
    - neutral preference:         1.00
    - strong positive preference: 1.10

    This keeps personalization light and prevents the image comparator
    from overpowering the original pair-level features.
    """

    opportunity = get_feature(features, "opportunity_score")
    affordability = get_feature(features, "affordability_score")
    commute = get_feature(features, "commute_score")
    amenity = get_feature(features, "amenity_score")
    space_fit = get_feature(features, "space_fit_score")

    if not preference_vector:
        return {
            "opportunity": clamp(opportunity),
            "affordability": clamp(affordability),
            "commute": clamp(commute),
            "amenity": clamp(amenity),
            "space_fit": clamp(space_fit),
        }

    economic_pref = normalize_signed_preference(
        preference_vector.get("economic_practicality", 0.0)
    )
    leisure_pref = normalize_signed_preference(
        preference_vector.get("leisure_preference", 0.0)
    )
    commute_pref = normalize_signed_preference(
        preference_vector.get("commute_affinity", 0.0)
    )
    family_pref = normalize_signed_preference(
        preference_vector.get("family_friendliness", 0.0)
    )

    return {
        "opportunity": clamp(opportunity),
        "affordability": clamp(affordability * (0.90 + 0.20 * economic_pref)),
        "commute": clamp(commute * (0.90 + 0.20 * commute_pref)),
        "amenity": clamp(amenity * (0.90 + 0.20 * leisure_pref)),
        "space_fit": clamp(space_fit * (0.90 + 0.20 * family_pref)),
    }


# ------------------------------------------------------------
# Fuzzy inference
# ------------------------------------------------------------

def evaluate_fuzzy_rules(adjusted: Dict[str, float]) -> Dict[str, float]:
    """
    Mamdani-style fuzzy rules using min for AND and max for aggregation.

    Output categories:
    - poor
    - fair
    - good
    - excellent
    """

    opportunity = fuzzify_score(adjusted["opportunity"])
    affordability = fuzzify_score(adjusted["affordability"])
    commute = fuzzify_score(adjusted["commute"])
    amenity = fuzzify_score(adjusted["amenity"])
    space_fit = fuzzify_score(adjusted["space_fit"])

    output = {
        "poor": 0.0,
        "fair": 0.0,
        "good": 0.0,
        "excellent": 0.0,
    }

    def apply_rule(category: str, strength: float) -> None:
        output[category] = max(output[category], clamp(strength))

    # Excellent rules
    apply_rule(
        "excellent",
        min(opportunity["high"], affordability["high"], commute["high"]),
    )

    apply_rule(
        "excellent",
        min(affordability["high"], commute["high"], amenity["high"], space_fit["high"]),
    )

    # Good rules
    apply_rule(
        "good",
        min(opportunity["high"], affordability["medium"], commute["medium"]),
    )

    apply_rule(
        "good",
        min(opportunity["medium"], affordability["high"], commute["high"]),
    )

    apply_rule(
        "good",
        min(amenity["high"], space_fit["high"]),
    )

    apply_rule(
        "good",
        min(opportunity["high"], commute["high"]),
    )

    # Fair rules
    apply_rule(
        "fair",
        min(opportunity["medium"], affordability["medium"], commute["medium"]),
    )

    apply_rule(
        "fair",
        min(affordability["medium"], commute["low"]),
    )

    apply_rule(
        "fair",
        min(opportunity["low"], affordability["high"], commute["high"]),
    )

    # Poor rules
    apply_rule(
        "poor",
        min(affordability["low"], commute["low"]),
    )

    apply_rule(
        "poor",
        min(opportunity["low"], affordability["low"]),
    )

    apply_rule(
        "poor",
        min(commute["low"], amenity["low"], space_fit["low"]),
    )

    return output


def defuzzify_output(output_memberships: Dict[str, float]) -> float:
    """
    Weighted-average defuzzification.

    Representative output values:
    poor      -> 0.20
    fair      -> 0.45
    good      -> 0.70
    excellent -> 0.92
    """

    output_values = {
        "poor": 0.20,
        "fair": 0.45,
        "good": 0.70,
        "excellent": 0.92,
    }

    numerator = 0.0
    denominator = 0.0

    for label, membership in output_memberships.items():
        numerator += output_values[label] * membership
        denominator += membership

    if denominator == 0:
        return 0.0

    return clamp(numerator / denominator)


def classify_fuzzy_score(score: float) -> str:
    score = clamp(score)

    if score >= 0.80:
        return "excellent"
    if score >= 0.60:
        return "good"
    if score >= 0.40:
        return "fair"
    return "poor"


def explain_fuzzy_result(
    adjusted: Dict[str, float],
    output_memberships: Dict[str, float],
    fuzzy_score: float,
) -> str:
    output_priority = {
        "poor": 1,
        "fair": 2,
        "good": 3,
        "excellent": 4,
    }

    strongest_output = max(
        output_memberships,
        key=lambda label: (output_memberships[label], output_priority[label]),
    )

    strongest_inputs = sorted(
        adjusted.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    top_input_1, top_value_1 = strongest_inputs[0]
    top_input_2, top_value_2 = strongest_inputs[1]

    return (
        f"Fuzzy result classified as {classify_fuzzy_score(fuzzy_score)}. "
        f"Strongest output membership: {strongest_output}. "
        f"Top contributing dimensions: {top_input_1}={top_value_1:.3f}, "
        f"{top_input_2}={top_value_2:.3f}."
    )


# ------------------------------------------------------------
# Public ranking function
# ------------------------------------------------------------

def rank_pairs_fuzzy(
    pair_representations: List[Dict[str, Any]],
    preference_vector: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """
    Main Layer 3B entry point.

    Parameters
    ----------
    pair_representations:
        List of candidate job-house pair dictionaries from Layer 2.

    preference_vector:
        Optional Layer 2b image preference vector normalized to [-1, 1].

    Returns
    -------
    List of pair dictionaries ranked by layer3b_fuzzy_score descending.
    """

    ranked_pairs: List[Dict[str, Any]] = []

    for pair in pair_representations:
        features = get_pair_feature_vector(pair)

        adjusted_features = apply_preference_adjustments(
            features=features,
            preference_vector=preference_vector,
        )

        output_memberships = evaluate_fuzzy_rules(adjusted_features)
        fuzzy_score = defuzzify_output(output_memberships)
        fuzzy_label = classify_fuzzy_score(fuzzy_score)

        enriched_pair = dict(pair)
        enriched_pair["layer3b_fuzzy_score"] = fuzzy_score
        enriched_pair["layer3b_fuzzy_label"] = fuzzy_label
        enriched_pair["layer3b_adjusted_features"] = adjusted_features
        enriched_pair["layer3b_output_memberships"] = output_memberships
        enriched_pair["layer3b_explanation"] = explain_fuzzy_result(
            adjusted=adjusted_features,
            output_memberships=output_memberships,
            fuzzy_score=fuzzy_score,
        )

        ranked_pairs.append(enriched_pair)

    ranked_pairs.sort(
        key=lambda pair: pair.get("layer3b_fuzzy_score", 0.0),
        reverse=True,
    )

    return ranked_pairs
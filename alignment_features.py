from __future__ import annotations

from typing import Dict, Any

from normalization_helpers import min_max_scale, inverse_min_max_scale, clip_01


def compute_urbanism_proxy(
    restaurants_within_2mi: float,
    grocery_stores_within_2mi: float,
    jobs_within_5mi: float,
    context: Dict[str, Any],
) -> float:
    restaurants_norm = min_max_scale(
        restaurants_within_2mi,
        context["restaurants_within_2mi_min"],
        context["restaurants_within_2mi_max"],
    )

    grocery_norm = min_max_scale(
        grocery_stores_within_2mi,
        context["grocery_stores_within_2mi_min"],
        context["grocery_stores_within_2mi_max"],
    )

    jobs_norm = min_max_scale(
        jobs_within_5mi,
        context["jobs_within_5mi_min"],
        context["jobs_within_5mi_max"],
    )

    proxy = 0.4 * restaurants_norm + 0.3 * grocery_norm + 0.3 * jobs_norm
    return clip_01(proxy)


def compute_urbanism_alignment(
    urbanism_preference: float,
    urbanism_proxy: float,
) -> float:
    return clip_01(1 - abs(urbanism_preference - urbanism_proxy))


def compute_affordability_alignment(
    frugality_preference: float,
    affordability_score: float,
) -> float:
    return clip_01(1 - abs(frugality_preference - affordability_score))


def compute_outdoor_proxy(
    parks_within_1mi: float,
    dist_nearest_park: float,
    avg_dist_top3_parks: float,
    dist_nearest_waterfront: float,
    context: Dict[str, Any],
) -> float:
    parks_norm = min_max_scale(
        parks_within_1mi,
        context["parks_within_1mi_min"],
        context["parks_within_1mi_max"],
    )

    park_prox = inverse_min_max_scale(
        dist_nearest_park,
        context["dist_nearest_park_min"],
        context["dist_nearest_park_max"],
    )

    top3_park_prox = inverse_min_max_scale(
        avg_dist_top3_parks,
        context["avg_dist_top3_parks_min"],
        context["avg_dist_top3_parks_max"],
    )

    waterfront_prox = inverse_min_max_scale(
        dist_nearest_waterfront,
        context["dist_nearest_waterfront_min"],
        context["dist_nearest_waterfront_max"],
    )

    proxy = (
        0.4 * parks_norm +
        0.3 * park_prox +
        0.2 * top3_park_prox +
        0.1 * waterfront_prox
    )

    return clip_01(proxy)


def compute_outdoor_alignment(
    outdoor_preference: float,
    outdoor_proxy: float,
) -> float:
    return clip_01(1 - abs(outdoor_preference - outdoor_proxy))


def compute_space_proxy(
    space_fit_score: float,
) -> float:
    """
    We reuse space_fit_score as the housing-side proxy.
    """
    return clip_01(space_fit_score)


def compute_space_alignment(
    family_space_preference: float,
    space_proxy: float,
) -> float:
    return clip_01(1 - abs(family_space_preference - space_proxy))


def build_alignment_feature_set(
    housing_row: Dict[str, Any],
    pair_features: Dict[str, float],
    inferred_preferences: Dict[str, float],
    context: Dict[str, Any],
) -> Dict[str, float]:

    urbanism_proxy = compute_urbanism_proxy(
        housing_row.get("restaurants_within_2mi"),
        housing_row.get("grocery_stores_within_2mi"),
        housing_row.get("jobs_within_5mi"),
        context,
    )

    outdoor_proxy = compute_outdoor_proxy(
        housing_row.get("parks_within_1mi"),
        housing_row.get("dist_nearest_park"),
        housing_row.get("avg_dist_top3_parks"),
        housing_row.get("dist_nearest_waterfront"),
        context,
    )

    space_proxy = compute_space_proxy(
        pair_features.get("space_fit_score")
    )

    return {
        "urbanism_alignment": compute_urbanism_alignment(
            inferred_preferences.get("urbanism_preference"),
            urbanism_proxy,
        ),
        "affordability_alignment": compute_affordability_alignment(
            inferred_preferences.get("frugality_preference"),
            pair_features.get("affordability_score"),
        ),
        "outdoor_alignment": compute_outdoor_alignment(
            inferred_preferences.get("outdoor_preference"),
            outdoor_proxy,
        ),
        "space_alignment": compute_space_alignment(
            inferred_preferences.get("family_space_preference"),
            space_proxy,
        ),
    }
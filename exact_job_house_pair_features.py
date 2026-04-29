from __future__ import annotations

from typing import Any, Dict
import math

import pandas as pd

from normalization_helpers import (
    clip_01,
    clipped_min_max_scale,
    inverse_min_max_scale,
    min_max_scale,
    safe_divide,
)

from normalization_helpers import build_normalization_context


EARTH_RADIUS_MILES = 3958.8


def compute_opportunity_score(
    log_opportunity_score_10mi: float,
    min_value: float,
    max_value: float,
    clip_min: float | None = None,
    clip_max: float | None = None,
) -> float:
    return clipped_min_max_scale(
        value=log_opportunity_score_10mi,
        min_value=min_value,
        max_value=max_value,
        lower_clip=clip_min,
        upper_clip=clip_max,
        default=0.0,
    )


def compute_affordability_score(
    housing_cost: float,
    annual_salary: float,
) -> float:
    monthly_income = safe_divide(annual_salary, 12.0, default=0.0)
    rent_burden = safe_divide(housing_cost, monthly_income, default=float("inf"))

    if rent_burden <= 0.30:
        return 1.0
    if rent_burden <= 0.40:
        return 0.7
    if rent_burden <= 0.50:
        return 0.4
    return 0.1


def compute_haversine_distance_miles(
    home_latitude: float,
    home_longitude: float,
    job_latitude: float,
    job_longitude: float,
) -> float:
    values = [home_latitude, home_longitude, job_latitude, job_longitude]
    if any(v is None or pd.isna(v) for v in values):
        return float("inf")

    lat1 = math.radians(float(home_latitude))
    lon1 = math.radians(float(home_longitude))
    lat2 = math.radians(float(job_latitude))
    lon2 = math.radians(float(job_longitude))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * math.asin(math.sqrt(a))
    return EARTH_RADIUS_MILES * c


def compute_commute_score(distance_miles: float) -> float:
    if distance_miles is None or pd.isna(distance_miles):
        return 0.0

    if distance_miles <= 5:
        return 1.0
    if distance_miles <= 10:
        return 0.8
    if distance_miles <= 20:
        return 0.5
    if distance_miles <= 30:
        return 0.2
    return 0.0


def compute_amenity_score(
    parks_within_1mi: float,
    restaurants_within_2mi: float,
    grocery_stores_within_2mi: float,
    dist_nearest_park: float,
    dist_nearest_grocery_store: float,
    parks_min: float,
    parks_max: float,
    restaurants_min: float,
    restaurants_max: float,
    grocery_min: float,
    grocery_max: float,
    park_dist_min: float,
    park_dist_max: float,
    grocery_dist_min: float,
    grocery_dist_max: float,
) -> float:
    parks_norm = min_max_scale(parks_within_1mi, parks_min, parks_max, default=0.0)
    restaurants_norm = min_max_scale(
        restaurants_within_2mi,
        restaurants_min,
        restaurants_max,
        default=0.0,
    )
    grocery_norm = min_max_scale(
        grocery_stores_within_2mi,
        grocery_min,
        grocery_max,
        default=0.0,
    )

    park_prox_norm = inverse_min_max_scale(
        dist_nearest_park,
        park_dist_min,
        park_dist_max,
        default=0.0,
    )
    grocery_prox_norm = inverse_min_max_scale(
        dist_nearest_grocery_store,
        grocery_dist_min,
        grocery_dist_max,
        default=0.0,
    )

    score = (
        0.25 * restaurants_norm
        + 0.25 * grocery_norm
        + 0.20 * parks_norm
        + 0.15 * park_prox_norm
        + 0.15 * grocery_prox_norm
    )
    return clip_01(score)


def compute_bedroom_adequacy_score(
    beds: float,
    household_size: int,
) -> float:
    if beds is None or pd.isna(beds):
        return 0.0

    beds = float(beds)
    household_size = int(household_size)

    if household_size <= 1:
        if beds == 1:
            return 1.0
        if beds >= 2:
            return 0.7
        return 0.2

    if household_size == 2:
        if 1 <= beds <= 2:
            return 1.0
        if beds >= 3:
            return 0.7
        return 0.2

    if 3 <= household_size <= 4:
        if 2 <= beds <= 3:
            return 1.0
        if beds >= 4:
            return 0.7
        return 0.2

    if beds >= 3:
        return 1.0
    if beds == 2:
        return 0.7
    return 0.2


def compute_space_fit_score(
    beds: float,
    sqft: float,
    household_size: int,
    sqft_min: float,
    sqft_max: float,
) -> float:
    bedroom_adequacy = compute_bedroom_adequacy_score(beds, household_size)
    sqft_score = min_max_scale(sqft, sqft_min, sqft_max, default=0.0)

    score = 0.7 * bedroom_adequacy + 0.3 * sqft_score
    return clip_01(score)


def build_exact_job_house_pair_features(
    housing_row: Dict[str, Any],
    job_row: Dict[str, Any],
    household_size: int,
    normalization_context: Dict[str, Any],
) -> Dict[str, float]:
    commute_distance_miles = compute_haversine_distance_miles(
        home_latitude=housing_row.get("home_latitude"),
        home_longitude=housing_row.get("home_longitude"),
        job_latitude=job_row.get("job_latitude"),
        job_longitude=job_row.get("job_longitude"),
    )

    pair_feature_vector = {
        "opportunity_score": compute_opportunity_score(
            log_opportunity_score_10mi=housing_row.get("log_opportunity_score_10mi"),
            min_value=normalization_context["log_opportunity_score_10mi_min"],
            max_value=normalization_context["log_opportunity_score_10mi_max"],
            clip_min=normalization_context.get("log_opportunity_score_10mi_clip_min"),
            clip_max=normalization_context.get("log_opportunity_score_10mi_clip_max"),
        ),
        "affordability_score": compute_affordability_score(
            housing_cost=housing_row.get("housing_cost"),
            annual_salary=job_row.get("annual_salary"),
        ),
        "commute_score": compute_commute_score(commute_distance_miles),
        "amenity_score": compute_amenity_score(
            parks_within_1mi=housing_row.get("parks_within_1mi"),
            restaurants_within_2mi=housing_row.get("restaurants_within_2mi"),
            grocery_stores_within_2mi=housing_row.get("grocery_stores_within_2mi"),
            dist_nearest_park=housing_row.get("dist_nearest_park"),
            dist_nearest_grocery_store=housing_row.get("dist_nearest_grocery_store"),
            parks_min=normalization_context["parks_within_1mi_min"],
            parks_max=normalization_context["parks_within_1mi_max"],
            restaurants_min=normalization_context["restaurants_within_2mi_min"],
            restaurants_max=normalization_context["restaurants_within_2mi_max"],
            grocery_min=normalization_context["grocery_stores_within_2mi_min"],
            grocery_max=normalization_context["grocery_stores_within_2mi_max"],
            park_dist_min=normalization_context["dist_nearest_park_min"],
            park_dist_max=normalization_context["dist_nearest_park_max"],
            grocery_dist_min=normalization_context["dist_nearest_grocery_store_min"],
            grocery_dist_max=normalization_context["dist_nearest_grocery_store_max"],
        ),
        "space_fit_score": compute_space_fit_score(
            beds=housing_row.get("beds"),
            sqft=housing_row.get("sqft"),
            household_size=household_size,
            sqft_min=normalization_context["sqft_min"],
            sqft_max=normalization_context["sqft_max"],
        ),
        "commute_distance_miles": commute_distance_miles,
    }

    return pair_feature_vector
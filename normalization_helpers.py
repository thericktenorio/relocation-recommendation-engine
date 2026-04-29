from __future__ import annotations

from typing import Any, Dict
import math
import pandas as pd


EPSILON = 1e-9


def clip_01(value: float) -> float:
    if value is None or pd.isna(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    if numerator is None or denominator is None:
        return default
    if pd.isna(numerator) or pd.isna(denominator):
        return default
    if abs(float(denominator)) < EPSILON:
        return default
    return float(numerator) / float(denominator)


def min_max_scale(value: float, min_value: float, max_value: float, default: float = 0.0) -> float:
    if value is None or min_value is None or max_value is None:
        return default
    if pd.isna(value) or pd.isna(min_value) or pd.isna(max_value):
        return default

    denominator = float(max_value) - float(min_value)
    if abs(denominator) < EPSILON:
        return default

    scaled = (float(value) - float(min_value)) / denominator
    return clip_01(scaled)


def inverse_min_max_scale(value: float, min_value: float, max_value: float, default: float = 0.0) -> float:
    scaled = min_max_scale(value, min_value, max_value, default=default)
    return clip_01(1.0 - scaled)


def safe_log(value: float, default: float = 0.0) -> float:
    if value is None or pd.isna(value):
        return default
    if float(value) <= 0:
        return default
    return math.log(float(value))


def get_percentile_bounds(
    series: pd.Series,
    lower_q: float = 0.0,
    upper_q: float = 1.0
) -> tuple[float, float]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return 0.0, 1.0

    lower = float(clean.quantile(lower_q))
    upper = float(clean.quantile(upper_q))

    if abs(upper - lower) < EPSILON:
        upper = lower + 1.0

    return lower, upper


def clipped_min_max_scale(
    value: float,
    min_value: float,
    max_value: float,
    lower_clip: float | None = None,
    upper_clip: float | None = None,
    default: float = 0.0
) -> float:
    if value is None or pd.isna(value):
        return default

    working_value = float(value)

    if lower_clip is not None:
        working_value = max(working_value, float(lower_clip))
    if upper_clip is not None:
        working_value = min(working_value, float(upper_clip))

    return min_max_scale(working_value, min_value, max_value, default=default)


def build_normalization_context(housing_df: pd.DataFrame) -> Dict[str, Any]:
    context: Dict[str, Any] = {}

    columns_to_bound = [
        "log_opportunity_score_10mi",
        "sqft",
        "parks_within_1mi",
        "restaurants_within_2mi",
        "grocery_stores_within_2mi",
        "jobs_within_5mi",
        "dist_nearest_park",
        "dist_nearest_grocery_store",
        "avg_dist_top3_parks",
        "dist_nearest_waterfront",
    ]

    for col in columns_to_bound:
        if col in housing_df.columns:
            series = pd.to_numeric(housing_df[col], errors="coerce")
            context[f"{col}_min"] = float(series.min(skipna=True)) if not series.dropna().empty else 0.0
            context[f"{col}_max"] = float(series.max(skipna=True)) if not series.dropna().empty else 1.0
        else:
            context[f"{col}_min"] = 0.0
            context[f"{col}_max"] = 1.0

    if "log_opportunity_score_10mi" in housing_df.columns:
        lower, upper = get_percentile_bounds(
            pd.to_numeric(housing_df["log_opportunity_score_10mi"], errors="coerce"),
            lower_q=0.00,
            upper_q=0.95,
        )
        context["log_opportunity_score_10mi_clip_min"] = lower
        context["log_opportunity_score_10mi_clip_max"] = upper
    else:
        context["log_opportunity_score_10mi_clip_min"] = 0.0
        context["log_opportunity_score_10mi_clip_max"] = 1.0

    return context
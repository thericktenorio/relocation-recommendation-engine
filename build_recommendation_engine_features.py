import json
from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# PATHS
# =========================================================
INPUT_PATH = Path("data/enriched/final_recommendation_engine_dataset.csv")
OUTPUT_PATH = Path("data/features/normalized/normalized_recommendation_engine_features.csv")
METADATA_PATH = Path("data/features/metadata/normalized_recommendation_engine_feature_metadata.json")
DISTANCE_DIAGNOSTICS_PATH = Path("data/features/diagnostics/distance_feature_diagnostics.csv")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
DISTANCE_DIAGNOSTICS_PATH.parent.mkdir(parents=True, exist_ok=True)


# =========================================================
# COLUMN GROUPS
# =========================================================
ANCHOR_COLUMNS = [
    "home_id",
    "city",
    "state",
    "region_id",
    "property_type",
    "location_type",
]

INVERTED_MINMAX_COLUMNS = [
    "housing_cost",
    "days_on_market",
    "dist_nearest_park",
    "dist_nearest_grocery_store",
    "dist_nearest_hospital",
    "dist_nearest_school",
    "dist_nearest_waterfront",
    "avg_dist_top3_parks",
    "dist_nearest_job",
    "avg_dist_nearest5_jobs",
    "avg_dist_nearest10_jobs",
]

DIRECT_MINMAX_COLUMNS = [
    "beds",
    "baths",
    "sqft",
    "parks_within_1mi",
    "restaurants_within_2mi",
    "grocery_stores_within_2mi",
    "schools_within_2mi",
    "jobs_within_5mi",
    "jobs_within_10mi",
    "avg_salary_nearest5_jobs",
    "max_salary_within_10mi",
    "log_opportunity_score_10mi",
]

LOG1P_COLUMNS = [
    "avg_salary_nearest5_jobs",
    "max_salary_within_10mi",
    "jobs_within_5mi",
    "jobs_within_10mi",
    "restaurants_within_2mi",
]

COUNT_COLUMNS = [
    "parks_within_1mi",
    "restaurants_within_2mi",
    "grocery_stores_within_2mi",
    "schools_within_2mi",
    "jobs_within_5mi",
    "jobs_within_10mi",
]

SALARY_COLUMNS = [
    "avg_salary_nearest5_jobs",
    "max_salary_within_10mi",
]

DISTANCE_COLUMNS = [
    "dist_nearest_park",
    "dist_nearest_grocery_store",
    "dist_nearest_hospital",
    "dist_nearest_school",
    "dist_nearest_waterfront",
    "avg_dist_top3_parks",
    "dist_nearest_job",
    "avg_dist_nearest5_jobs",
    "avg_dist_nearest10_jobs",
]

ALL_REQUIRED_COLUMNS = (
    ANCHOR_COLUMNS
    + INVERTED_MINMAX_COLUMNS
    + DIRECT_MINMAX_COLUMNS
)


# =========================================================
# HELPERS
# =========================================================
def validate_required_columns(df: pd.DataFrame) -> None:
    missing_cols = [col for col in ALL_REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            "The following required columns are missing from the input dataset:\n"
            + "\n".join(missing_cols)
        )


def minmax_normalize(series: pd.Series) -> pd.Series:
    min_val = series.min()
    max_val = series.max()

    if pd.isna(min_val) or pd.isna(max_val):
        return pd.Series(np.nan, index=series.index)

    if max_val == min_val:
        return pd.Series(0.5, index=series.index)

    return (series - min_val) / (max_val - min_val)


def inverted_minmax_normalize(series: pd.Series) -> pd.Series:
    min_val = series.min()
    max_val = series.max()

    if pd.isna(min_val) or pd.isna(max_val):
        return pd.Series(np.nan, index=series.index)

    if max_val == min_val:
        return pd.Series(0.5, index=series.index)

    return (max_val - series) / (max_val - min_val)


def build_distance_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    diagnostics = []

    total_rows = len(df)

    for col in DISTANCE_COLUMNS:
        series = df[col]

        null_count = int(series.isna().sum())
        zero_count = int((series == 0).sum())

        diagnostics.append({
            "feature": col,
            "row_count": total_rows,
            "null_count": null_count,
            "null_pct": round((null_count / total_rows) * 100, 4) if total_rows else 0.0,
            "zero_count": zero_count,
            "zero_pct": round((zero_count / total_rows) * 100, 4) if total_rows else 0.0,
            "min_value": None if series.dropna().empty else float(series.min()),
            "median_value": None if series.dropna().empty else float(series.median()),
            "max_value": None if series.dropna().empty else float(series.max()),
        })

    return pd.DataFrame(diagnostics)


def print_distance_warnings(distance_diag_df: pd.DataFrame) -> None:
    print("\n=== DISTANCE FEATURE DIAGNOSTICS ===")
    print(distance_diag_df.to_string(index=False))

    warning_rows = distance_diag_df[
        (distance_diag_df["null_pct"] > 20) | (distance_diag_df["zero_pct"] > 20)
    ]

    if not warning_rows.empty:
        print("\nWARNING: Some distance features appear null-heavy or zero-heavy.")
        print("Review these columns carefully:")
        print(warning_rows[["feature", "null_pct", "zero_pct"]].to_string(index=False))
    else:
        print("\nNo distance features exceeded the warning threshold for nulls or zeros.")


def impute_missing_values(df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    df = df.copy()

    # Count features: missing -> 0
    for col in COUNT_COLUMNS:
        null_count = int(df[col].isna().sum())
        if null_count > 0:
            df[col] = df[col].fillna(0)
        metadata[col] = {
            **metadata.get(col, {}),
            "missing_value_strategy": "fillna_0",
            "nulls_filled": null_count,
        }

    # Salary features: missing -> median
    for col in SALARY_COLUMNS:
        null_count = int(df[col].isna().sum())
        median_val = float(df[col].median()) if not df[col].dropna().empty else 0.0
        if null_count > 0:
            df[col] = df[col].fillna(median_val)
        metadata[col] = {
            **metadata.get(col, {}),
            "missing_value_strategy": "fillna_median",
            "nulls_filled": null_count,
            "fill_value": median_val,
        }

    # Distance features: missing -> column max
    for col in DISTANCE_COLUMNS:
        null_count = int(df[col].isna().sum())
        max_val = float(df[col].max()) if not df[col].dropna().empty else 0.0
        if null_count > 0:
            df[col] = df[col].fillna(max_val)
        metadata[col] = {
            **metadata.get(col, {}),
            "missing_value_strategy": "fillna_column_max",
            "nulls_filled": null_count,
            "fill_value": max_val,
        }

    # Other inverted/direct features not covered above
    other_cols = set(INVERTED_MINMAX_COLUMNS + DIRECT_MINMAX_COLUMNS) - set(
        COUNT_COLUMNS + SALARY_COLUMNS + DISTANCE_COLUMNS
    )

    for col in other_cols:
        null_count = int(df[col].isna().sum())
        if null_count > 0:
            median_val = float(df[col].median()) if not df[col].dropna().empty else 0.0
            df[col] = df[col].fillna(median_val)
            metadata[col] = {
                **metadata.get(col, {}),
                "missing_value_strategy": "fillna_median",
                "nulls_filled": null_count,
                "fill_value": median_val,
            }
        else:
            metadata[col] = {
                **metadata.get(col, {}),
                "missing_value_strategy": "no_missing_values",
                "nulls_filled": 0,
            }

    return df


def apply_log1p_transforms(df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    df = df.copy()

    for col in LOG1P_COLUMNS:
        negative_count = int((df[col] < 0).sum())
        if negative_count > 0:
            raise ValueError(
                f"Column '{col}' contains negative values. log1p cannot be applied safely."
            )

        df[col] = np.log1p(df[col])

        metadata[col] = {
            **metadata.get(col, {}),
            "transform": "log1p",
        }

    return df


def build_normalized_table(df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    normalized_df = df[ANCHOR_COLUMNS].copy()

    for col in DIRECT_MINMAX_COLUMNS:
        norm_col = f"{col}_norm"
        normalized_df[norm_col] = minmax_normalize(df[col])

        metadata[col] = {
            **metadata.get(col, {}),
            "normalization": "direct_minmax",
            "direction": "higher_is_better",
            "raw_min_after_processing": float(df[col].min()),
            "raw_max_after_processing": float(df[col].max()),
        }

    for col in INVERTED_MINMAX_COLUMNS:
        norm_col = f"{col}_norm"
        normalized_df[norm_col] = inverted_minmax_normalize(df[col])

        metadata[col] = {
            **metadata.get(col, {}),
            "normalization": "inverted_minmax",
            "direction": "higher_is_better",
            "raw_min_after_processing": float(df[col].min()),
            "raw_max_after_processing": float(df[col].max()),
        }

    return normalized_df


def main():
    print("Loading dataset...")
    df = pd.read_csv(INPUT_PATH)

    validate_required_columns(df)

    metadata = {}

    # 1. Diagnostics before imputation
    distance_diag_df = build_distance_diagnostics(df)
    print_distance_warnings(distance_diag_df)

    # 2. Missing value handling
    df_processed = impute_missing_values(df, metadata)

    # 3. Log transforms
    df_processed = apply_log1p_transforms(df_processed, metadata)

    # 4. Normalize
    normalized_df = build_normalized_table(df_processed, metadata)

    # 5. Save outputs
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    normalized_df.to_csv(OUTPUT_PATH, index=False)
    distance_diag_df.to_csv(DISTANCE_DIAGNOSTICS_PATH, index=False)

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)

    # 6. Console summary
    print("\n=== OUTPUT SUMMARY ===")
    print(f"Normalized feature table saved to: {OUTPUT_PATH}")
    print(f"Feature metadata saved to: {METADATA_PATH}")
    print(f"Distance diagnostics saved to: {DISTANCE_DIAGNOSTICS_PATH}")
    print(f"Normalized table shape: {normalized_df.shape}")

    print("\nNormalized columns:")
    print(normalized_df.columns.tolist())

    print("\nPreview:")
    print(normalized_df.head())


if __name__ == "__main__":
    main()
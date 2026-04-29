from pathlib import Path
import pandas as pd


# =========================================================
# CONFIG
# =========================================================
HOUSING_PATH = Path("data/clean/rentcast_housing_clean.csv")
CIVIC_AMENITIES_FEATURES_PATH = Path("data/enriched/housing_civic_amenities_features.csv")
JOB_FEATURES_PATH = Path("data/enriched/housing_job_features.csv")
OUTPUT_PATH = Path("data/enriched/final_recommendation_engine_dataset.csv")


# =========================================================
# DATA LOADING / VALIDATION
# =========================================================
def load_data():
    housing = pd.read_csv(HOUSING_PATH)
    civic_features = pd.read_csv(CIVIC_AMENITIES_FEATURES_PATH)
    job_features = pd.read_csv(JOB_FEATURES_PATH)

    return housing, civic_features, job_features


def validate_required_columns(housing, civic_features, job_features):
    required_housing_cols = {"home_id"}
    required_civic_cols = {"home_id"}
    required_job_cols = {"home_id"}

    missing_housing = required_housing_cols - set(housing.columns)
    missing_civic = required_civic_cols - set(civic_features.columns)
    missing_job = required_job_cols - set(job_features.columns)

    if missing_housing:
        raise ValueError(f"Missing housing columns: {missing_housing}")
    if missing_civic:
        raise ValueError(f"Missing civic/amenities feature columns: {missing_civic}")
    if missing_job:
        raise ValueError(f"Missing job feature columns: {missing_job}")


def validate_unique_home_ids(housing, civic_features, job_features):
    if housing["home_id"].duplicated().any():
        dupes = housing.loc[housing["home_id"].duplicated(), "home_id"].unique()
        raise ValueError(f"Duplicate home_id values found in housing data: {dupes[:10]}")

    if civic_features["home_id"].duplicated().any():
        dupes = civic_features.loc[civic_features["home_id"].duplicated(), "home_id"].unique()
        raise ValueError(f"Duplicate home_id values found in civic features: {dupes[:10]}")

    if job_features["home_id"].duplicated().any():
        dupes = job_features.loc[job_features["home_id"].duplicated(), "home_id"].unique()
        raise ValueError(f"Duplicate home_id values found in job features: {dupes[:10]}")


def validate_join_coverage(housing, civic_features, job_features):
    housing_ids = set(housing["home_id"])
    civic_ids = set(civic_features["home_id"])
    job_ids = set(job_features["home_id"])

    missing_from_civic = housing_ids - civic_ids
    missing_from_job = housing_ids - job_ids

    if missing_from_civic:
        raise ValueError(
            f"{len(missing_from_civic)} home_id values from housing are missing in civic features. "
            f"Examples: {list(sorted(missing_from_civic))[:10]}"
        )

    if missing_from_job:
        raise ValueError(
            f"{len(missing_from_job)} home_id values from housing are missing in job features. "
            f"Examples: {list(sorted(missing_from_job))[:10]}"
        )


# =========================================================
# COLUMN MANAGEMENT
# =========================================================
def drop_redundant_feature_columns(housing, civic_features, job_features):
    """
    Keep housing as the source of truth for shared descriptive fields.
    Drop overlapping non-key columns from feature tables before merging.
    """
    housing_cols = set(housing.columns)

    civic_drop_cols = [col for col in civic_features.columns if col != "home_id" and col in housing_cols]
    job_drop_cols = [col for col in job_features.columns if col != "home_id" and col in housing_cols]

    civic_features = civic_features.drop(columns=civic_drop_cols, errors="ignore").copy()
    job_features = job_features.drop(columns=job_drop_cols, errors="ignore").copy()

    return civic_features, job_features


# =========================================================
# MERGE PIPELINE
# =========================================================
def build_final_dataset():
    housing, civic_features, job_features = load_data()

    validate_required_columns(housing, civic_features, job_features)
    validate_unique_home_ids(housing, civic_features, job_features)
    validate_join_coverage(housing, civic_features, job_features)

    civic_features, job_features = drop_redundant_feature_columns(
        housing, civic_features, job_features
    )

    final_df = (
        housing
        .merge(civic_features, on="home_id", how="left", validate="one_to_one")
        .merge(job_features, on="home_id", how="left", validate="one_to_one")
    )

    return final_df


# =========================================================
# OPTIONAL VALIDATION / REPORTING
# =========================================================
def print_dataset_summary(df):
    print("\nFinal dataset summary")
    print("-" * 60)
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print("\nColumns:")
    print(list(df.columns))

    print("\nNull percentage by column:")
    null_pct = (df.isnull().mean() * 100).sort_values(ascending=False)
    print(null_pct)

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        print("\nNumeric summary:")
        print(df[numeric_cols].describe())


# =========================================================
# MAIN
# =========================================================
def main():
    final_df = build_final_dataset()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT_PATH, index=False)

    print("Created final_recommendation_engine_dataset.csv successfully.")
    print(f"Saved to: {OUTPUT_PATH}")

    print_dataset_summary(final_df)


if __name__ == "__main__":
    main()
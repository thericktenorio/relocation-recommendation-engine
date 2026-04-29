from pathlib import Path
import numpy as np
import pandas as pd


# =========================================================
# CONFIG
# =========================================================
HOUSING_PATH = Path("data/clean/rentcast_housing_clean.csv")
JOBS_PATH = Path("data/clean/adzuna_jobs_clean.csv")
OUTPUT_PATH = Path("data/enriched/housing_job_features.csv")

# All computed geospatial distances are in miles.
EARTH_RADIUS_MILES = 3958.8

# Small constant to stabilize salary / distance scoring
EPSILON_DISTANCE = 0.1


# =========================================================
# HAVERSINE DISTANCE
# =========================================================
def haversine_matrix_miles(home_lat, home_lon, job_lat, job_lon):
    """
    Compute pairwise Haversine distances in miles between:
      - homes: arrays of shape (n_homes,)
      - jobs:  arrays of shape (n_jobs,)

    Returns:
      ndarray of shape (n_homes, n_jobs)
    """
    if len(job_lat) == 0:
        return np.empty((len(home_lat), 0))

    home_lat_rad = np.radians(home_lat)[:, np.newaxis]
    home_lon_rad = np.radians(home_lon)[:, np.newaxis]
    job_lat_rad = np.radians(job_lat)[np.newaxis, :]
    job_lon_rad = np.radians(job_lon)[np.newaxis, :]

    dlat = job_lat_rad - home_lat_rad
    dlon = job_lon_rad - home_lon_rad

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(home_lat_rad) * np.cos(job_lat_rad) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))

    return EARTH_RADIUS_MILES * c


# =========================================================
# FEATURE HELPERS
# =========================================================
def nearest_distance(dist_matrix):
    """
    Return the nearest job distance for each home.
    If there are no jobs, return NaN.
    """
    if dist_matrix.shape[1] == 0:
        return np.full(dist_matrix.shape[0], np.nan)
    return np.min(dist_matrix, axis=1)


def avg_nearest_k_distance(dist_matrix, k):
    """
    Return the average distance to the top-k nearest jobs for each home.
    If fewer than k jobs exist, use however many are available.
    If none exist, return NaN.
    """
    if dist_matrix.shape[1] == 0:
        return np.full(dist_matrix.shape[0], np.nan)

    k_eff = min(k, dist_matrix.shape[1])
    nearest_k = np.partition(dist_matrix, kth=k_eff - 1, axis=1)[:, :k_eff]
    return np.mean(nearest_k, axis=1)


def count_within_radius(dist_matrix, radius_miles):
    """
    Return the number of jobs within radius_miles for each home.
    If there are no jobs, return 0.
    """
    if dist_matrix.shape[1] == 0:
        return np.zeros(dist_matrix.shape[0], dtype=int)
    return np.sum(dist_matrix <= radius_miles, axis=1)


def avg_salary_of_nearest_k_nearest_jobs(dist_matrix, salaries, k):
    """
    For each home, find the k nearest jobs and return the average salary.
    If fewer than k jobs exist, use however many are available.
    If none exist, return NaN.
    """
    if dist_matrix.shape[1] == 0:
        return np.full(dist_matrix.shape[0], np.nan)

    k_eff = min(k, dist_matrix.shape[1])
    nearest_idx = np.argpartition(dist_matrix, kth=k_eff - 1, axis=1)[:, :k_eff]
    nearest_salaries = salaries[nearest_idx]
    return np.mean(nearest_salaries, axis=1)


def max_salary_within_radius(dist_matrix, salaries, radius_miles):
    """
    For each home, return the maximum salary among jobs within radius_miles.
    If no jobs fall within the radius, return NaN.
    """
    if dist_matrix.shape[1] == 0:
        return np.full(dist_matrix.shape[0], np.nan)

    result = np.full(dist_matrix.shape[0], np.nan)
    within_mask = dist_matrix <= radius_miles

    for i in range(dist_matrix.shape[0]):
        row_salaries = salaries[within_mask[i]]
        if row_salaries.size > 0:
            result[i] = np.max(row_salaries)

    return result


def log_opportunity_score_within_radius(
    dist_matrix,
    salaries,
    radius_miles,
    epsilon=EPSILON_DISTANCE
):
    """
    For each home, compute a radius-constrained log opportunity score:

        sum(log1p(annual_salary) / (distance + epsilon))

    using only jobs within radius_miles.

    Why log1p?
    - reduces sensitivity to salary outliers
    - preserves ordering
    - safely handles zero salaries if ever present

    If no jobs fall within the radius for a home, return NaN.
    """
    if dist_matrix.shape[1] == 0:
        return np.full(dist_matrix.shape[0], np.nan)

    within_mask = dist_matrix <= radius_miles
    log_salaries = np.log1p(salaries)

    score_matrix = np.where(
        within_mask,
        log_salaries[np.newaxis, :] / (dist_matrix + epsilon),
        0.0
    )

    scores = np.sum(score_matrix, axis=1)

    # If a home has no jobs within the radius, return NaN instead of 0
    has_any_jobs_within_radius = np.any(within_mask, axis=1)
    scores = np.where(has_any_jobs_within_radius, scores, np.nan)

    return scores


# =========================================================
# DATA LOADING / VALIDATION
# =========================================================
def load_and_validate():
    housing = pd.read_csv(HOUSING_PATH)
    jobs = pd.read_csv(JOBS_PATH)

    required_housing_cols = {
        "home_id",
        "latitude",
        "longitude",
    }

    required_job_cols = {
        "job_id",
        "annual_salary",
        "latitude",
        "longitude",
    }

    missing_housing = required_housing_cols - set(housing.columns)
    missing_jobs = required_job_cols - set(jobs.columns)

    if missing_housing:
        raise ValueError(f"Missing housing columns: {missing_housing}")
    if missing_jobs:
        raise ValueError(f"Missing job columns: {missing_jobs}")

    housing = housing.dropna(subset=["home_id", "latitude", "longitude"]).copy()
    jobs = jobs.dropna(subset=["job_id", "annual_salary", "latitude", "longitude"]).copy()

    housing["latitude"] = pd.to_numeric(housing["latitude"], errors="coerce")
    housing["longitude"] = pd.to_numeric(housing["longitude"], errors="coerce")
    jobs["latitude"] = pd.to_numeric(jobs["latitude"], errors="coerce")
    jobs["longitude"] = pd.to_numeric(jobs["longitude"], errors="coerce")
    jobs["annual_salary"] = pd.to_numeric(jobs["annual_salary"], errors="coerce")

    housing = housing.dropna(subset=["latitude", "longitude"]).copy()
    jobs = jobs.dropna(subset=["latitude", "longitude", "annual_salary"]).copy()

    housing = housing.drop_duplicates(subset=["home_id"]).copy()
    jobs = jobs.drop_duplicates(subset=["job_id"]).copy()

    return housing, jobs


# =========================================================
# OPTIONAL VALIDATION SUMMARY
# =========================================================
def print_feature_summary(features):
    print("\nFeature summary:")
    for col in features.columns:
        if col in {"home_id", "region_id", "city"}:
            continue

        if pd.api.types.is_numeric_dtype(features[col]):
            print(f"\n{col}")
            print(features[col].describe())


# =========================================================
# MAIN PIPELINE
# =========================================================
def build_housing_job_features():
    housing, jobs = load_and_validate()

    home_lat = housing["latitude"].to_numpy()
    home_lon = housing["longitude"].to_numpy()
    job_lat = jobs["latitude"].to_numpy()
    job_lon = jobs["longitude"].to_numpy()
    salaries = jobs["annual_salary"].to_numpy(dtype=float)

    dist_matrix = haversine_matrix_miles(home_lat, home_lon, job_lat, job_lon)

    features = housing[["home_id", "region_id", "city"]].copy()

    # -----------------------------------------------------
    # Distance features (miles)
    # -----------------------------------------------------
    features["dist_nearest_job"] = nearest_distance(dist_matrix)
    features["avg_dist_nearest5_jobs"] = avg_nearest_k_distance(dist_matrix, 5)
    features["avg_dist_nearest10_jobs"] = avg_nearest_k_distance(dist_matrix, 10)

    # -----------------------------------------------------
    # Density features
    # -----------------------------------------------------
    features["jobs_within_5mi"] = count_within_radius(dist_matrix, 5.0)
    features["jobs_within_10mi"] = count_within_radius(dist_matrix, 10.0)

    # -----------------------------------------------------
    # Salary / opportunity features
    # -----------------------------------------------------
    features["avg_salary_nearest5_jobs"] = avg_salary_of_nearest_k_nearest_jobs(
        dist_matrix, salaries, 5
    )

    features["max_salary_within_10mi"] = max_salary_within_radius(
    dist_matrix, salaries, 10.0
    )

    features["log_opportunity_score_10mi"] = log_opportunity_score_within_radius(
        dist_matrix=dist_matrix,
        salaries=salaries,
        radius_miles=10.0,
        epsilon=EPSILON_DISTANCE
    )

    # -----------------------------------------------------
    # Save output
    # -----------------------------------------------------
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUTPUT_PATH, index=False)

    print("Created housing_job_features.csv successfully.")
    print(f"Rows: {len(features)}")
    print(f"Columns: {list(features.columns)}")
    print(f"Saved to: {OUTPUT_PATH}")

    print_feature_summary(features)


if __name__ == "__main__":
    build_housing_job_features()
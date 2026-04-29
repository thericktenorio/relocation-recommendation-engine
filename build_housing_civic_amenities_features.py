from pathlib import Path
import numpy as np
import pandas as pd


# =========================================================
# CONFIG
# =========================================================
HOUSING_PATH = Path("data/clean/rentcast_housing_clean.csv")
AMENITIES_PATH = Path("data/clean/overpass_civic_amenities_clean.csv")
OUTPUT_PATH = Path("data/enriched/housing_civic_amenities_features.csv")

# All computed geospatial distances are in miles.
EARTH_RADIUS_MILES = 3958.8


# =========================================================
# HAVERSINE DISTANCE
# =========================================================
def haversine_matrix_miles(home_lat, home_lon, amenity_lat, amenity_lon):
    """
    Compute pairwise Haversine distances in miles between:
      - homes: arrays of shape (n_homes,)
      - amenities: arrays of shape (n_amenities,)

    Returns:
      ndarray of shape (n_homes, n_amenities)
    """
    if len(amenity_lat) == 0:
        return np.empty((len(home_lat), 0))

    home_lat_rad = np.radians(home_lat)[:, np.newaxis]
    home_lon_rad = np.radians(home_lon)[:, np.newaxis]
    amenity_lat_rad = np.radians(amenity_lat)[np.newaxis, :]
    amenity_lon_rad = np.radians(amenity_lon)[np.newaxis, :]

    dlat = amenity_lat_rad - home_lat_rad
    dlon = amenity_lon_rad - home_lon_rad

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(home_lat_rad) * np.cos(amenity_lat_rad) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))

    return EARTH_RADIUS_MILES * c


# =========================================================
# FEATURE HELPERS
# =========================================================
def nearest_distance(dist_matrix):
    """
    Return the nearest distance for each home.
    If no amenities exist for the category, return NaN.
    """
    if dist_matrix.shape[1] == 0:
        return np.full(dist_matrix.shape[0], np.nan)
    return np.min(dist_matrix, axis=1)


def avg_top_k_distance(dist_matrix, k):
    """
    Return the average distance to the top-k nearest amenities for each home.
    If fewer than k amenities exist, use however many are available.
    If none exist, return NaN.
    """
    if dist_matrix.shape[1] == 0:
        return np.full(dist_matrix.shape[0], np.nan)

    k_eff = min(k, dist_matrix.shape[1])
    nearest_k = np.partition(dist_matrix, kth=k_eff - 1, axis=1)[:, :k_eff]
    return np.mean(nearest_k, axis=1)


def count_within_radius(dist_matrix, radius_miles):
    """
    Return the count of amenities within the given radius for each home.
    If no amenities exist for the category, return 0.
    """
    if dist_matrix.shape[1] == 0:
        return np.zeros(dist_matrix.shape[0], dtype=int)
    return np.sum(dist_matrix <= radius_miles, axis=1)


# =========================================================
# DATA LOADING / VALIDATION
# =========================================================
def load_and_validate():
    housing = pd.read_csv(HOUSING_PATH)
    amenities = pd.read_csv(AMENITIES_PATH)

    required_housing_cols = {
        "home_id",
        "latitude",
        "longitude",
    }

    required_amenity_cols = {
        "clean_amenity_id",
        "amenity_type",
        "latitude",
        "longitude",
    }

    missing_housing = required_housing_cols - set(housing.columns)
    missing_amenities = required_amenity_cols - set(amenities.columns)

    if missing_housing:
        raise ValueError(f"Missing housing columns: {missing_housing}")
    if missing_amenities:
        raise ValueError(f"Missing amenities columns: {missing_amenities}")

    housing = housing.dropna(subset=["home_id", "latitude", "longitude"]).copy()
    amenities = amenities.dropna(
        subset=["clean_amenity_id", "amenity_type", "latitude", "longitude"]
    ).copy()

    housing["latitude"] = pd.to_numeric(housing["latitude"], errors="coerce")
    housing["longitude"] = pd.to_numeric(housing["longitude"], errors="coerce")
    amenities["latitude"] = pd.to_numeric(amenities["latitude"], errors="coerce")
    amenities["longitude"] = pd.to_numeric(amenities["longitude"], errors="coerce")

    housing = housing.dropna(subset=["latitude", "longitude"]).copy()
    amenities = amenities.dropna(subset=["latitude", "longitude"]).copy()

    housing = housing.drop_duplicates(subset=["home_id"]).copy()
    amenities["amenity_type"] = amenities["amenity_type"].astype(str).str.strip().str.lower()

    return housing, amenities


# =========================================================
# AMENITY SUBSETTING
# =========================================================
def get_amenity_subset(amenities, amenity_type_value):
    """
    Return amenities filtered to a single amenity_type value.
    """
    return amenities.loc[amenities["amenity_type"] == amenity_type_value].copy()


# =========================================================
# FEATURE BUILDERS
# =========================================================
def build_distance_feature(housing, amenities, amenity_type_value):
    subset = get_amenity_subset(amenities, amenity_type_value)

    dist_matrix = haversine_matrix_miles(
        housing["latitude"].to_numpy(),
        housing["longitude"].to_numpy(),
        subset["latitude"].to_numpy(),
        subset["longitude"].to_numpy(),
    )
    return nearest_distance(dist_matrix)


def build_count_feature(housing, amenities, amenity_type_value, radius_miles):
    subset = get_amenity_subset(amenities, amenity_type_value)

    dist_matrix = haversine_matrix_miles(
        housing["latitude"].to_numpy(),
        housing["longitude"].to_numpy(),
        subset["latitude"].to_numpy(),
        subset["longitude"].to_numpy(),
    )
    return count_within_radius(dist_matrix, radius_miles)


def build_avg_top_k_feature(housing, amenities, amenity_type_value, k):
    subset = get_amenity_subset(amenities, amenity_type_value)

    dist_matrix = haversine_matrix_miles(
        housing["latitude"].to_numpy(),
        housing["longitude"].to_numpy(),
        subset["latitude"].to_numpy(),
        subset["longitude"].to_numpy(),
    )
    return avg_top_k_distance(dist_matrix, k)


# =========================================================
# OPTIONAL VALIDATION SUMMARY
# =========================================================
def print_feature_summary(features):
    print("\nFeature summary:")
    for col in features.columns:
        if col == "home_id":
            continue

        if pd.api.types.is_numeric_dtype(features[col]):
            print(f"\n{col}")
            print(features[col].describe())


# =========================================================
# MAIN PIPELINE
# =========================================================
def build_housing_civic_amenities_features():
    housing, amenities = load_and_validate()

    features = pd.DataFrame({
        "home_id": housing["home_id"].values
    })

    # -----------------------------------------------------
    # Nearest-distance features (miles)
    # -----------------------------------------------------
    features["dist_nearest_park"] = build_distance_feature(
        housing, amenities, "park"
    )

    features["dist_nearest_grocery_store"] = build_distance_feature(
        housing, amenities, "grocery_store"
    )

    features["dist_nearest_hospital"] = build_distance_feature(
        housing, amenities, "hospital"
    )

    features["dist_nearest_school"] = build_distance_feature(
        housing, amenities, "school"
    )

    features["dist_nearest_waterfront"] = build_distance_feature(
        housing, amenities, "waterfront"
    )

    # -----------------------------------------------------
    # Density/count features
    # -----------------------------------------------------
    features["parks_within_1mi"] = build_count_feature(
        housing, amenities, "park", 1.0
    )

    features["restaurants_within_2mi"] = build_count_feature(
        housing, amenities, "restaurant", 2.0
    )

    features["grocery_stores_within_2mi"] = build_count_feature(
        housing, amenities, "grocery_store", 2.0
    )

    features["schools_within_2mi"] = build_count_feature(
        housing, amenities, "school", 2.0
    )

    # -----------------------------------------------------
    # Richness / optional features
    # -----------------------------------------------------
    features["avg_dist_top3_parks"] = build_avg_top_k_feature(
        housing, amenities, "park", 3
    )

    # -----------------------------------------------------
    # Save output
    # -----------------------------------------------------
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUTPUT_PATH, index=False)

    print("Created housing_civic_amenities_features.csv successfully.")
    print(f"Rows: {len(features)}")
    print(f"Columns: {list(features.columns)}")
    print(f"Saved to: {OUTPUT_PATH}")

    print_feature_summary(features)


if __name__ == "__main__":
    build_housing_civic_amenities_features()
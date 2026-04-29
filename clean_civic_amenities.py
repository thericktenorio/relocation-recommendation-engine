# clean_civic_amenities.py

from pathlib import Path
import math
import re
import unicodedata
import pandas as pd


# =========================
# CONFIG
# =========================
RAW_PATH = Path("data/raw/civic_amenities_raw.csv")
CLEAN_DIR = Path("data/clean")
OUTPUT_PATH = CLEAN_DIR / "overpass_civic_amenities_clean.csv"

ALLOWED_AMENITY_TYPES = {
    "grocery_store",
    "hospital",
    "park",
    "restaurant",
    "school",
    "waterfront",
}

# Conservative thresholds.
# You can tune these later after inspecting outputs.
NAMED_DEDUPE_RADIUS_METERS = 75
UNNAMED_DEDUPE_RADIUS_METERS = 40

ELEMENT_PRIORITY = {
    "relation": 3,
    "way": 2,
    "node": 1,
}


# =========================
# HELPERS
# =========================
def normalize_name(value):
    """
    Normalize amenity_name for grouping.
    """
    if pd.isna(value):
        return None

    value = str(value).strip()
    if not value:
        return None

    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", "ignore").decode("ascii")
    value = value.lower().strip()

    # collapse punctuation to spaces
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()

    return value if value else None


def haversine_meters(lat1, lon1, lat2, lon2):
    """
    Distance in meters between two lat/lon points.
    """
    r = 6371000  # Earth radius in meters

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def resolve_column(df, candidates, required=True):
    """
    Return the first matching column from a list of candidates.
    """
    for col in candidates:
        if col in df.columns:
            return col

    if required:
        raise ValueError(f"Missing required column. Tried: {candidates}")
    return None


def choose_best_record(group_df):
    """
    Pick the highest-priority representative row.
    Priority:
      1) relation > way > node
      2) named over unnamed
      3) lower amenity_id as stable tie-breaker if present
    """
    work = group_df.copy()

    work["_element_priority"] = (
        work["osm_element_type"]
        .astype(str)
        .str.lower()
        .map(ELEMENT_PRIORITY)
        .fillna(0)
    )

    work["_has_name"] = work["amenity_name"].notna().astype(int)

    sort_cols = ["_element_priority", "_has_name"]
    ascending = [False, False]

    if "amenity_id" in work.columns:
        sort_cols.append("amenity_id")
        ascending.append(True)

    work = work.sort_values(sort_cols, ascending=ascending)
    chosen = work.iloc[0].copy()

    source_ids = group_df["amenity_id"].astype(str).tolist() if "amenity_id" in group_df.columns else []
    chosen["source_row_count"] = len(group_df)
    chosen["source_amenity_ids"] = "|".join(source_ids)

    return chosen


def cluster_rows(group_df, radius_meters):
    """
    Very small-group clustering using pairwise connectivity.
    This is intentionally simple and dependency-light.

    Returns a list of representative rows, one per cluster.
    """
    if len(group_df) == 1:
        rep = choose_best_record(group_df)
        return [rep]

    records = group_df.reset_index(drop=True)
    n = len(records)

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # pairwise clustering
    for i in range(n):
        lat1 = records.at[i, "latitude"]
        lon1 = records.at[i, "longitude"]
        for j in range(i + 1, n):
            lat2 = records.at[j, "latitude"]
            lon2 = records.at[j, "longitude"]

            dist = haversine_meters(lat1, lon1, lat2, lon2)
            if dist <= radius_meters:
                union(i, j)

    clusters = {}
    for idx in range(n):
        root = find(idx)
        clusters.setdefault(root, []).append(idx)

    representatives = []
    for member_indexes in clusters.values():
        cluster_df = records.iloc[member_indexes].copy()
        rep = choose_best_record(cluster_df)
        representatives.append(rep)

    return representatives


# =========================
# MAIN
# =========================
def main():
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {RAW_PATH}")

    df = pd.read_csv(RAW_PATH)

    print(f"Loaded {len(df):,} raw rows.")

    # Resolve required columns with a little flexibility.
    lat_col = resolve_column(df, ["latitude", "lat", "y"])
    lon_col = resolve_column(df, ["longitude", "lon", "lng", "x"])
    amenity_type_col = resolve_column(df, ["amenity_type"])
    amenity_name_col = resolve_column(df, ["amenity_name"], required=False)
    osm_element_type_col = resolve_column(df, ["osm_element_type"], required=False)
    osm_element_id_col = resolve_column(df, ["osm_element_id"], required=False)
    amenity_id_col = resolve_column(df, ["amenity_id"], required=False)

    # Standardize column names used by the cleaner.
    rename_map = {
        lat_col: "latitude",
        lon_col: "longitude",
        amenity_type_col: "amenity_type",
    }

    if amenity_name_col:
        rename_map[amenity_name_col] = "amenity_name"
    if osm_element_type_col:
        rename_map[osm_element_type_col] = "osm_element_type"
    if osm_element_id_col:
        rename_map[osm_element_id_col] = "osm_element_id"
    if amenity_id_col:
        rename_map[amenity_id_col] = "amenity_id"

    df = df.rename(columns=rename_map)

    # Drop irrelevant fields.
    if "zip_code" in df.columns:
        df = df.drop(columns=["zip_code"])

    # Ensure required standard columns exist.
    for col in ["amenity_name", "osm_element_type", "osm_element_id", "amenity_id"]:
        if col not in df.columns:
            df[col] = pd.NA

    # Strip text fields
    for col in ["amenity_type", "amenity_name", "osm_element_type", "osm_element_id", "amenity_id"]:
        df[col] = df[col].astype("string").str.strip()

    # Normalize amenity_type
    df["amenity_type"] = df["amenity_type"].str.lower()

    # Keep only target amenity categories
    before = len(df)
    df = df[df["amenity_type"].isin(ALLOWED_AMENITY_TYPES)].copy()
    print(f"Removed {before - len(df):,} rows outside allowed amenity types.")

    # Numeric coordinates
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    # Drop blank coords
    before = len(df)
    df = df.dropna(subset=["latitude", "longitude"]).copy()
    print(f"Dropped {before - len(df):,} rows with missing latitude/longitude.")

    # Basic coordinate sanity
    before = len(df)
    df = df[
        df["latitude"].between(-90, 90)
        & df["longitude"].between(-180, 180)
    ].copy()
    print(f"Dropped {before - len(df):,} rows with invalid coordinate ranges.")

    # Normalize name
    df["normalized_amenity_name"] = df["amenity_name"].apply(normalize_name)

    # osm_key
    df["osm_key"] = (
        df["osm_element_type"].fillna("unknown").astype(str)
        + "_"
        + df["osm_element_id"].fillna("missing").astype(str)
    )

    # Remove exact duplicate rows on strongest identifying subset
    exact_subset = ["amenity_type", "normalized_amenity_name", "latitude", "longitude", "osm_key"]
    before = len(df)
    df = df.drop_duplicates(subset=exact_subset).copy()
    print(f"Removed {before - len(df):,} exact duplicates.")

    # Split named vs unnamed
    named_df = df[df["normalized_amenity_name"].notna()].copy()
    unnamed_df = df[df["normalized_amenity_name"].isna()].copy()

    # -------------------------
    # DEDUPLICATE NAMED
    # -------------------------
    named_reps = []

    if not named_df.empty:
        for (amenity_type, norm_name), group in named_df.groupby(
            ["amenity_type", "normalized_amenity_name"],
            dropna=False,
        ):
            reps = cluster_rows(group, NAMED_DEDUPE_RADIUS_METERS)
            named_reps.extend(reps)

    named_clean = pd.DataFrame(named_reps) if named_reps else pd.DataFrame(columns=df.columns)

    # -------------------------
    # DEDUPLICATE UNNAMED
    # -------------------------
    unnamed_reps = []

    if not unnamed_df.empty:
        for amenity_type, group in unnamed_df.groupby("amenity_type", dropna=False):
            reps = cluster_rows(group, UNNAMED_DEDUPE_RADIUS_METERS)
            unnamed_reps.extend(reps)

    unnamed_clean = pd.DataFrame(unnamed_reps) if unnamed_reps else pd.DataFrame(columns=df.columns)

    # Combine
    clean_df = pd.concat([named_clean, unnamed_clean], ignore_index=True)

    # Final cleaned ID
    clean_df = clean_df.reset_index(drop=True)
    clean_df["clean_amenity_id"] = [
        f"CA_{i:06d}" for i in range(1, len(clean_df) + 1)
    ]

    # Reorder columns
    preferred_order = [
        "clean_amenity_id",
        "amenity_id",
        "osm_element_id",
        "osm_element_type",
        "osm_key",
        "amenity_type",
        "amenity_name",
        "normalized_amenity_name",
        "latitude",
        "longitude",
        "source_row_count",
        "source_amenity_ids",
    ]

    existing_cols = [c for c in preferred_order if c in clean_df.columns]
    other_cols = [c for c in clean_df.columns if c not in existing_cols]
    clean_df = clean_df[existing_cols + other_cols]

    # Save
    clean_df.to_csv(OUTPUT_PATH, index=False)

    print("\nCleaning complete.")
    print(f"Final cleaned rows: {len(clean_df):,}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
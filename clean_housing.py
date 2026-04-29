import pandas as pd
import numpy as np
import re
from pathlib import Path


'''
# 1. Load Raw Data
'''
INPUT_FILE = "data/raw/rentcast_housing_raw.csv"
OUTPUT_FILE = "data/clean/rentcast_housing_clean.csv"

# Ensure output directory exists
Path("data/clean").mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE)
original_count = len(df)


'''
# 2. Standardize Column Names
'''
df.columns = [col.strip().lower() for col in df.columns]


'''
# 3. Basic Cleaning
'''
# Standardize text columns
text_columns = [
    "source_platform",
    "source_listing_id",
    "listing_type",
    "property_type",
    "city",
    "state",
    "zip_code",
    "listing_status",
    "address"
]

for col in text_columns:
    if col in df.columns:
        df[col] = df[col].astype("string").str.strip()

# Replace common string nulls with actual NaN
df = df.replace({
    "": np.nan,
    "nan": np.nan,
    "None": np.nan,
    "none": np.nan,
    "null": np.nan,
    "NULL": np.nan
})

# Numeric columns
numeric_columns = [
    "housing_cost",
    "beds",
    "baths",
    "sqft",
    "latitude",
    "longitude",
    "days_on_market"
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Datetime columns
if "listed_date" in df.columns:
    df["listed_date"] = pd.to_datetime(df["listed_date"], errors="coerce")


'''
# 4. Standardize Text Values
'''
# Normalize city/state formatting
if "city" in df.columns:
    df["city"] = df["city"].str.title()

if "state" in df.columns:
    df["state"] = df["state"].str.upper()

# Normalize ZIP codes
if "zip_code" in df.columns:
    df["zip_code"] = (
        df["zip_code"]
        .astype("string")
        .str.extract(r"(\d{5})", expand=False)
    )

# Normalize source platform
if "source_platform" in df.columns:
    df["source_platform"] = "rentcast"

# Normalize listing type
if "listing_type" in df.columns:
    df["listing_type"] = df["listing_type"].str.lower()

# Normalize listing status
if "listing_status" in df.columns:
    df["listing_status"] = df["listing_status"].str.title()

# Normalize property_type values into approved categories
def normalize_property_type(value):
    if pd.isna(value):
        return np.nan

    value = str(value).strip().lower()

    if "single" in value:
        return "single-family"
    elif "town" in value:
        return "townhome"
    elif "apart" in value:
        return "apartment"
    elif "condo" in value:
        return "condo"
    else:
        return value

df["property_type"] = df["property_type"].apply(normalize_property_type)


'''
# 5. Deduplication
'''
# Keep strongest row if duplicates exist:
# priority = fewest missing values, then most recent listed_date, then lower days_on_market
df["missing_count"] = df.isna().sum(axis=1)

sort_cols = ["missing_count"]
ascending_order = [True]

if "listed_date" in df.columns:
    sort_cols.append("listed_date")
    ascending_order.append(False)

if "days_on_market" in df.columns:
    sort_cols.append("days_on_market")
    ascending_order.append(True)

df = df.sort_values(by=sort_cols, ascending=ascending_order).copy()

# Primary deduplication by source listing ID
if "source_listing_id" in df.columns:
    df = df.drop_duplicates(subset=["source_listing_id"], keep="first")

# Secondary deduplication by normalized address
if "address" in df.columns:
    df["address_key"] = (
        df["address"]
        .astype("string")
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]", "", regex=True)
    )
    df = df.drop_duplicates(subset=["address_key"], keep="first")
else:
    df["address_key"] = np.nan


'''
# 6. Derived Features
'''
# region_id derived from city
def make_region_id(city_value):
    if pd.isna(city_value):
        return np.nan

    city_value = str(city_value).strip().lower()
    city_value = re.sub(r"[^a-z0-9]+", "_", city_value)
    city_value = re.sub(r"_+", "_", city_value).strip("_")
    return city_value

df["region_id"] = df["city"].apply(make_region_id)

# location_type derived from property_type
# simple first-pass rule set
def derive_location_type(property_type):
    if pd.isna(property_type):
        return "unknown"

    property_type = str(property_type).strip().lower()

    if property_type in ["apartment", "condo"]:
        return "urban"
    elif property_type in ["townhome", "single-family"]:
        return "suburban"
    else:
        return "unknown"

df["location_type"] = df["property_type"].apply(derive_location_type)

# Create home_id after deduplication
df = df.reset_index(drop=True)
df["home_id"] = df.index + 1
df["home_id"] = df["home_id"].apply(lambda x: f"HOME_{x:06d}")


'''
# 7. Final Column Order
'''
final_columns = [
    "home_id",
    "source_platform",
    "source_listing_id",
    "listing_type",
    "property_type",
    "housing_cost",
    "beds",
    "baths",
    "sqft",
    "city",
    "state",
    "zip_code",
    "latitude",
    "longitude",
    "listing_status",
    "days_on_market",
    "listed_date",
    "address",
    "region_id",
    "location_type"
]

final_df = df[final_columns].copy()


'''
# 8. Export Clean File
'''
final_df.to_csv(OUTPUT_FILE, index=False)

clean_count = len(final_df)
removed_count = original_count - clean_count

print("Housing cleaning complete.")
print(f"Original rows: {original_count}")
print(f"Clean rows: {clean_count}")
print(f"Duplicates/removed rows: {removed_count}")
print(f"Output saved to: {OUTPUT_FILE}")
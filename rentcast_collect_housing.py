import csv
import json
from pathlib import Path

import requests

import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("RENTCAST_API_KEY")

if not API_KEY:
    raise ValueError("Missing RENTCAST_API_KEY in .env file.")
BASE_URL = os.getenv("RENTCAST_BASE_URL")

TARGET_PER_CITY = 100
REQUEST_LIMIT = 150     #broad pull, but still conservative give 50API calls free via RentCast

CITIES = [
    {"city": "San Diego", "state": "CA"},
    {"city": "Dallas", "state": "TX"},
    {"city": "Raleigh", "state": "NC"},
]

OUTPUT_CSV = Path("data/raw/rentcast_housing_raw.csv")
RAW_JSON_DIR = Path("raw_json")
RAW_JSON_DIR.mkdir(exist_ok=True)

CSV_COLUMNS = [
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
]

HEADERS = {"X-Api-Key": API_KEY}


def ensure_csv_exists(csv_path: Path, columns: list[str]) -> None:
    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()


def load_existing_ids(csv_path: Path) -> set[str]:
    existing_ids = set()
    if not csv_path.exists():
        return existing_ids

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            listing_id = row.get("source_listing_id")
            if listing_id:
                existing_ids.add(listing_id)
    
    return existing_ids


def flatten_listing(listing: dict) -> dict:
    return {
        "source_platform": "rentcast",
        "source_listing_id": listing.get("id"),
        "listing_type": "rent",
        "property_type": listing.get("propertyType"),
        "housing_cost": listing.get("price"),
        "beds": listing.get("bedrooms"),
        "baths": listing.get("bathrooms"),
        "sqft": listing.get("squareFootage"),
        "city": listing.get("city"),
        "state": listing.get("state"),
        "zip_code": listing.get("zipCode"),
        "latitude": listing.get("latitude"),
        "longitude": listing.get("longitude"),
        "listing_status": listing.get("status"),
        "days_on_market": listing.get("daysOnMarket"),
        "listed_date": listing.get("listedDate"),
        "address": listing.get("formattedAddress"),
    }


def is_valid_row(row: dict) -> bool:
    required_fields = [
        "source_listing_id",
        "housing_cost",
        "beds",
        "baths",
        "city",
        "state",
    ]
    return all(row.get(field) not in (None, "") for field in required_fields)


def fetch_city_page(city: str, state: str, offset: int = 0) -> list[dict]:
    params = {
        "city": city,
        "state": state,
        "propertyType": "Single Family,Condo,Townhouse,Apartment",
        "status": "Active",
        "limit": REQUEST_LIMIT,
        "offset": offset,
        "includeTotalCount": "true",
    }

    response = requests.get(BASE_URL, headers=HEADERS, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    json_filename = f"{city.lower().replace(' ', '_')}_{state.lower()}_offset_{offset}.json"
    with (RAW_JSON_DIR / json_filename).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    
    if isinstance(data, list):
        return data
    
    return data.get("listings", data.get("results", []))


def append_rows(csv_path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerows(rows)


def pull_city(city: str, state: str, existing_ids: set[str]) -> int:
    city_count = 0
    offset = 0

    while city_count < TARGET_PER_CITY:
        listings = fetch_city_page(city, state, offset=offset)
    
        if not listings:
            print(f"No more listings returned for {city}, {state}.")
            break

        rows_to_append = []

        for listing in listings:
            row = flatten_listing(listing)
            listing_id = row["source_listing_id"]

            if not listing_id or listing_id in existing_ids:
                continue

            if not is_valid_row(row):
                continue

            rows_to_append.append(row)
            existing_ids.add(listing_id)
            city_count += 1

            if city_count >= TARGET_PER_CITY:
                break
        
        append_rows(OUTPUT_CSV, rows_to_append)

        print(
            f"{city}, {state} | offset={offset}  | "
            f"appended={len(rows_to_append)} | city_total={city_count}"
        )

        if city_count >= TARGET_PER_CITY:
            break

        # If API returns less than request limit, there may be no more pages to pull from
        if len(listings) < REQUEST_LIMIT:
            break

        offset += REQUEST_LIMIT

    return city_count


def main():
    ensure_csv_exists(OUTPUT_CSV, CSV_COLUMNS)
    existing_ids = load_existing_ids(OUTPUT_CSV)

    total_added = 0

    for location in CITIES:
        city = location["city"]
        state = location["state"]

        print(f"\nPulling listings for {city}, {state}...")
        added_for_city = pull_city(city, state, existing_ids)
        total_added += added_for_city

        print(f"Finished {city}, {state}: {added_for_city} listings added.")
    
    print(f"\nDone. Total new listings added: {total_added}")
    print(f"CSV saved to: {OUTPUT_CSV}")
    print(f"Raw JSON saved in: {RAW_JSON_DIR.resolve()}")


if __name__ == "__main__":
    main()
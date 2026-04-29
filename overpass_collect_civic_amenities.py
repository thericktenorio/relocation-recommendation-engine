"""
overpass_collect_civic_amenities.py

Purpose:
    Collect civic / amenities POI data from OpenStreetMap via Overpass API
    and export a normalized raw CSV for downstream enrichment.

Output schema:
    amenity_id
    region_id
    city
    state
    zip_code
    amenity_type
    amenity_name
    latitude
    longitude
    osm_element_type
    osm_element_id
    source

Notes:
    - Uses bounding boxes for reliability in Python automation.
    - Tier 1 amenities are enabled by default.
    - Optional amenities can be turned on later.
    - For ways / relations, uses Overpass `out center` so every record has lat/lon.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests


# =========================
# CONFIG
# =========================

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OUTPUT_CSV = Path("data/raw/civic_amenities_raw.csv")
RAW_JSON_DIR = Path("data/raw/osm_raw_json")
RAW_JSON_DIR.mkdir(parents=True, exist_ok=True)

REQUEST_TIMEOUT = 120
SLEEP_BETWEEN_REQUESTS_SEC = 2.0
MAX_RETRIES = 3
OVERPASS_TIMEOUT_SEC = 90

# Enable only the categories you want to acquire now.
INCLUDE_OPTIONAL_WATERFRONT = True
INCLUDE_OPTIONAL_NATURE_PRESERVE = False
INCLUDE_OPTIONAL_TRAIL = False

# Bounding boxes:
# Format = (south, west, north, east)
# Adjust as needed.
REGIONS = [
    {
        "region_id": "san_diego_ca",
        "city": "San Diego",
        "state": "CA",
        "bbox": (32.5343, -117.5967, 33.1142, -116.9050),
    },
    {
        "region_id": "dallas_tx",
        "city": "Dallas",
        "state": "TX",
        "bbox": (32.5456, -97.0404, 33.0239, -96.4637),
    },
    {
        "region_id": "raleigh_nc",
        "city": "Raleigh",
        "state": "NC",
        "bbox": (35.6300, -78.8400, 35.9500, -78.4500),
    },
]

# Output columns
CSV_COLUMNS = [
    "amenity_id",
    "region_id",
    "city",
    "state",
    "zip_code",
    "amenity_type",
    "amenity_name",
    "latitude",
    "longitude",
    "osm_element_type",
    "osm_element_id",
    "source",
]

# Amenity mapping:
# Each amenity_type maps to one or more OSM tag filters.
# Each tuple = (key, value)
#
# Tier 1 defaults:
AMENITY_TAGS: Dict[str, List[Tuple[str, str]]] = {
    "park": [
        ("leisure", "park"),
    ],
    "school": [
        ("amenity", "school"),
    ],
    "grocery_store": [
        ("shop", "supermarket"),
        ("shop", "grocery"),
    ],
    "hospital": [
        ("amenity", "hospital"),
    ],
    "restaurant": [
        ("amenity", "restaurant"),
    ],
    "bar": [
        ("amenity", "bar"),
    ],
}

if INCLUDE_OPTIONAL_WATERFRONT:
    # Project-defined rollup category, derived from multiple OSM tags.
    AMENITY_TAGS["waterfront"] = [
        ("natural", "water"),
        ("waterway", "river"),
        ("natural", "beach"),
    ]

if INCLUDE_OPTIONAL_NATURE_PRESERVE:
    AMENITY_TAGS["nature_preserve"] = [
        ("leisure", "nature_reserve"),
    ]

if INCLUDE_OPTIONAL_TRAIL:
    # Very rough first-pass trail proxy.
    # Trail extraction can get more complex later.
    AMENITY_TAGS["trail"] = [
        ("highway", "path"),
        ("route", "hiking"),
    ]


# =========================
# HELPERS
# =========================

def bbox_to_overpass_coords(bbox: Tuple[float, float, float, float]) -> str:
    """
    Overpass bbox format:
        (south,west,north,east)
    """
    south, west, north, east = bbox
    return f"({south},{west},{north},{east})"


def build_query_for_tag(
    key: str,
    value: str,
    bbox: Tuple[float, float, float, float],
    timeout_sec: int = OVERPASS_TIMEOUT_SEC,
) -> str:
    """
    Build an Overpass QL query for one OSM tag within one bounding box.
    Uses out center so ways/relations include a derived point.
    """
    bbox_str = bbox_to_overpass_coords(bbox)

    query = f"""
    [out:json][timeout:{timeout_sec}];
    (
      node["{key}"="{value}"]{bbox_str};
      way["{key}"="{value}"]{bbox_str};
      relation["{key}"="{value}"]{bbox_str};
    );
    out center tags;
    """
    return query.strip()

def overpass_post(query: str, session: requests.Session) -> dict:
    """
    Execute an Overpass query with simple retry logic.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.post(
                OVERPASS_URL,
                data={"data": query},
                timeout=REQUEST_TIMEOUT,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "recommendation_engine/1.0 (civic amenities collection)",
                },
            )

            if response.status_code == 429:
                wait = 5 * attempt
                print(f"Rate limited (429). Sleeping {wait}s before retry...")
                time.sleep(wait)
                continue

            response.raise_for_status()
            return response.json()

        except (requests.RequestException, json.JSONDecodeError) as exc:
            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"Overpass request failed after {MAX_RETRIES} attempts: {exc}"
                ) from exc

            wait = 5 * attempt
            print(f"Request failed on attempt {attempt}: {exc}. Retrying in {wait}s...")
            time.sleep(wait)

    raise RuntimeError("Unexpected Overpass retry failure.")


def extract_lat_lon(element: dict) -> Tuple[Optional[float], Optional[float]]:
    """
    Nodes usually have lat/lon.
    Ways and relations can provide center.lat / center.lon when queried with `out center`.
    """
    if "lat" in element and "lon" in element:
        return element.get("lat"), element.get("lon")

    center = element.get("center", {})
    return center.get("lat"), center.get("lon")


def get_name(tags: dict) -> str:
    """
    Best-effort human-readable amenity name.
    """
    return (
        tags.get("name")
        or tags.get("official_name")
        or tags.get("brand")
        or tags.get("operator")
        or ""
    )


def get_zip_code(tags: dict) -> str:
    """
    Best-effort ZIP extraction.
    Often absent in OSM POIs, so blank is acceptable.
    """
    return (
        tags.get("addr:postcode")
        or tags.get("postal_code")
        or ""
    )


def normalize_element(
    element: dict,
    amenity_type: str,
    region: dict,
) -> Optional[dict]:
    """
    Convert one Overpass element to the raw civic amenities schema.
    """
    lat, lon = extract_lat_lon(element)
    if lat is None or lon is None:
        return None

    tags = element.get("tags", {})

    osm_type = element.get("type", "")
    osm_id = element.get("id", "")
    amenity_id = f"osm_{osm_type}_{osm_id}"

    record = {
        "amenity_id": amenity_id,
        "region_id": region["region_id"],
        "city": region["city"],
        "state": region["state"],
        "zip_code": get_zip_code(tags),
        "amenity_type": amenity_type,
        "amenity_name": get_name(tags),
        "latitude": lat,
        "longitude": lon,
        "osm_element_type": osm_type,
        "osm_element_id": osm_id,
        "source": "openstreetmap_overpass",
    }
    return record


def dedupe_records(records: Iterable[dict]) -> List[dict]:
    """
    De-duplicate by (amenity_type, osm_element_type, osm_element_id).
    This is enough for first-pass normalization.
    """
    seen = set()
    deduped = []

    for rec in records:
        key = (
            rec["amenity_type"],
            rec["osm_element_type"],
            rec["osm_element_id"],
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(rec)

    return deduped


def save_json(data: dict, region_id: str, amenity_type: str, tag_key: str, tag_value: str) -> None:
    """
    Save raw Overpass response for debugging / auditability.
    """
    safe = f"{region_id}__{amenity_type}__{tag_key}_{tag_value}".replace(":", "_")
    path = RAW_JSON_DIR / f"{safe}.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def collect_region_amenities(region: dict, session: requests.Session) -> List[dict]:
    """
    Collect all configured amenities for one region.
    """
    all_records: List[dict] = []

    print(f"\nCollecting amenities for {region['city']}, {region['state']} ({region['region_id']})")

    for amenity_type, tag_filters in AMENITY_TAGS.items():
        print(f"  -> amenity_type: {amenity_type}")

        for tag_key, tag_value in tag_filters:
            print(f"     querying tag: {tag_key}={tag_value}")
            query = build_query_for_tag(tag_key, tag_value, region["bbox"])
            data = overpass_post(query, session)
            save_json(data, region["region_id"], amenity_type, tag_key, tag_value)

            elements = data.get("elements", [])
            print(f"       returned {len(elements)} elements")

            for element in elements:
                record = normalize_element(
                    element=element,
                    amenity_type=amenity_type,
                    region=region,
                )
                if record:
                    all_records.append(record)

            time.sleep(SLEEP_BETWEEN_REQUESTS_SEC)

    all_records = dedupe_records(all_records)
    print(f"  deduped region total: {len(all_records)}")
    return all_records


def write_csv(records: List[dict], output_csv: Path) -> None:
    """
    Write normalized records to CSV.
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    session = requests.Session()
    all_records: List[dict] = []

    for region in REGIONS:
        region_records = collect_region_amenities(region, session)
        all_records.extend(region_records)

    # Final global dedupe just in case
    all_records = dedupe_records(all_records)

    write_csv(all_records, OUTPUT_CSV)

    print("\nDone.")
    print(f"Total rows written: {len(all_records)}")
    print(f"Output file: {OUTPUT_CSV.resolve()}")


if __name__ == "__main__":
    main()
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv


load_dotenv()

APP_ID = os.getenv("ADZUNA_APP_ID")
APP_KEY = os.getenv("ADZUNA_APP_KEY")

if not APP_ID or not APP_KEY:
    raise ValueError("Missing ADZUNA_APP_ID or ADZUNA_APP_KEY in .env file.")

BASE_URL = "https://api.adzuna.com/v1/api/jobs/us/search"

OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents = True, exist_ok = True)

CITY_CONFIG = {
    "San Diego, CA": "SAN_DIEGO_CA",
    "Dallas, TX": "DALLAS_TX",
    "Raleigh, NC": "RALEIGH_NC",
}

SEARCH_TERMS = [
    "software engineer",
    "data analyst",
    "registered nurse",
    "accountant",
]

RESULTS_PER_PAGE = 20
PAGES_PER_TERM = 1  # just starting small here

def fetch_jobs(term: str, city: str, page: int) -> dict[str, Any]:
    """Fetch one page of Adzuna job results."""
    params = {
        "app_id": APP_ID,
        "app_key": APP_KEY,
        "results_per_page": RESULTS_PER_PAGE,
        "what": term,
        "where": city,
        "content-type": "application/json",
    }

    url = f"{BASE_URL}/{page}"
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def flatten_job(job: dict[str, Any], city: str, region_id: str, query_term: str) -> dict[str, Any]:
    '''Flaten raw Adzuna JSON into a CSV-friendly row.'''
    location = job.get("location", {}) or {}
    company = job.get("company", {}) or {}
    category = job.get("category", {}) or {}

    return {
        "job_id": job.get("id"),
        "region_id": region_id,
        "source_city": city,
        "query_term": query_term,
        "raw_job_title": job.get("title"),
        "employer": company.get("display_name"),
        "location_display_name": location.get("display_name"),
        "category_label": category.get("label"),
        "category_tag": category.get("tag"),
        "salary_min": job.get("salary_min"),
        "salary_max": job.get("salary_max"),
        "salary_is_predicted": job.get("salary_is_predicted"),
        "contract_type": job.get("contract_type"),
        "contract_time": job.get("contract_time"),
        "latitude": job.get("latitude"),
        "longitude": job.get("longitude"),
        "created": job.get("created"),
        "redirect_url": job.get("redirect_url"),
        "description": job.get("description"),
    }


def collect_all_jobs() -> pd.DataFrame:
    """Collect jobs for all configured cities and search terms"""
    rows: list[dict[str, Any]] = []

    for city, region_id in CITY_CONFIG.items():
        for term in SEARCH_TERMS:
            for page in range(1, PAGES_PER_TERM + 1):
                print(f"Fetching city={city} | term={term} | page={page}")
                payload = fetch_jobs(term=term, city=city, page=page)

                results = payload.get("results", [])
                for job in results:
                    rows.append(flatten_job(job, city=city, region_id=region_id, query_term=term))
                
                # polite pause to avoid hammering the API
                time.sleep(3)

    df = pd.DataFrame(rows)

    # de-duplicate by Adzuna job ID
    if not df.empty:
        df = df.drop_duplicates(subset=["job_id"]).reset_index(drop=True)
    
    return df


def main() -> None:
    df = collect_all_jobs()
    output_path = OUTPUT_DIR / "adzuna_jobs_raw.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
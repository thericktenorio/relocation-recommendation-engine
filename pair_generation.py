from __future__ import annotations

from typing import List, Dict, Any

import pandas as pd

from exact_job_house_pair_features import compute_haversine_distance_miles


def generate_candidate_pairs(
    housing_df: pd.DataFrame,
    jobs_df: pd.DataFrame,
    max_jobs_per_home: int = 10,
    max_commute_miles: float = 25.0,
) -> List[Dict[str, Any]]:
    """
    Build a manageable set of exact job-house candidate pairs.

    Strategy:
    - iterate through housing rows
    - compute exact distance to each available job
    - keep only jobs within max_commute_miles
    - sort by nearest jobs first
    - keep up to max_jobs_per_home jobs per house

    Returns a list of dictionaries:
        {
            "housing_row": {...},
            "job_row": {...},
            "commute_distance_miles": ...
        }
    """
    candidate_pairs: List[Dict[str, Any]] = []

    if housing_df.empty or jobs_df.empty:
        return candidate_pairs

    for _, housing in housing_df.iterrows():
        housing_row = housing.to_dict()
        local_matches = []

        for _, job in jobs_df.iterrows():
            job_row = job.to_dict()

            distance_miles = compute_haversine_distance_miles(
                home_latitude=housing_row.get("home_latitude"),
                home_longitude=housing_row.get("home_longitude"),
                job_latitude=job_row.get("job_latitude"),
                job_longitude=job_row.get("job_longitude"),
            )

            if distance_miles <= max_commute_miles:
                local_matches.append(
                    {
                        "housing_row": housing_row,
                        "job_row": job_row,
                        "commute_distance_miles": distance_miles,
                    }
                )

        local_matches.sort(key=lambda x: x["commute_distance_miles"])
        candidate_pairs.extend(local_matches[:max_jobs_per_home])

    return candidate_pairs
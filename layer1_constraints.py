from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd


VALID_CITIES = ["Dallas, TX", "Raleigh, NC", "San Diego, CA"]
VALID_INDUSTRIES = ["All Available Industries", "Finance", "Healthcare", "Technology"]
VALID_CAREER_STAGES = ["All", "Entry", "Mid", "Senior"]
VALID_HOUSING_PREFERENCES = [
    "Lowest rent possible",
    "Fairly priced",
    "Luxury",
    "Open",
]


@dataclass
class ConstraintInputs:
    selected_cities: List[str]
    selected_industry: str
    selected_career_stage: str
    housing_preference: str
    household_size: Optional[int]
    ignore_household_size: bool = False


def normalize_region_id(series: pd.Series) -> pd.Series:
    region_map = {
        "dallas": "dallas",
        "DALLAS": "dallas",
        "DALLAS_TX": "dallas",
        "dallas_tx": "dallas",
        "Dallas, TX": "dallas",

        "raleigh": "raleigh",
        "RALEIGH": "raleigh",
        "RALEIGH_NC": "raleigh",
        "raleigh_nc": "raleigh",
        "Raleigh, NC": "raleigh",

        "san_diego": "san_diego",
        "SAN_DIEGO": "san_diego",
        "SAN_DIEGO_CA": "san_diego",
        "san_diego_ca": "san_diego",
        "San Diego, CA": "san_diego",
    }

    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(r"[\s,]+", "_", regex=True)
    )

    return cleaned.map(region_map).fillna(cleaned.str.lower())


def normalize_city_name(series: pd.Series) -> pd.Series:
    """
    Converts city values such as 'Dallas' into 'Dallas, TX' so the filter logic
    can use one consistent city label format.
    """
    city_map = {
        "Dallas": "Dallas, TX",
        "dallas": "Dallas, TX",
        "dallas, tx": "Dallas, TX",
        "Dallas, TX": "Dallas, TX",

        "Raleigh": "Raleigh, NC",
        "raleigh": "Raleigh, NC",
        "raleigh, nc": "Raleigh, NC",
        "Raleigh, NC": "Raleigh, NC",

        "San Diego": "San Diego, CA",
        "san diego": "San Diego, CA",
        "san diego, ca": "San Diego, CA",
        "San Diego, CA": "San Diego, CA",
    }
    return series.astype(str).str.strip().map(city_map).fillna(series.astype(str).str.strip())


def validate_inputs(inputs: ConstraintInputs) -> None:
    if not set(inputs.selected_cities).issubset(set(VALID_CITIES)):
        raise ValueError("One or more selected cities are invalid.")

    if inputs.selected_industry not in VALID_INDUSTRIES:
        raise ValueError("Selected industry is invalid.")

    if inputs.selected_career_stage not in VALID_CAREER_STAGES:
        raise ValueError("Selected career stage is invalid.")

    if inputs.housing_preference not in VALID_HOUSING_PREFERENCES:
        raise ValueError("Selected housing preference is invalid.")

    if not inputs.ignore_household_size:
        if inputs.household_size is None or inputs.household_size < 1:
            raise ValueError("Household size must be an integer greater than 0 unless ignored.")


def standardize_jobs_df(jobs_df: pd.DataFrame) -> pd.DataFrame:
    df = jobs_df.copy()

    if "region_id" not in df.columns:
        raise ValueError("Jobs dataframe must contain a 'region_id' column.")
    if "industry" not in df.columns:
        raise ValueError("Jobs dataframe must contain an 'industry' column.")
    if "career_stage" not in df.columns:
        raise ValueError("Jobs dataframe must contain a 'career_stage' column.")

    df["region_id"] = normalize_region_id(df["region_id"])
    df["industry"] = df["industry"].astype(str).str.strip().str.lower()
    df["career_stage"] = df["career_stage"].astype(str).str.strip().str.lower()

    df = df.dropna(subset=["region_id"])
    return df


def standardize_housing_df(housing_df: pd.DataFrame) -> pd.DataFrame:
    df = housing_df.copy()

    required_columns = ["city", "housing_cost", "beds", "region_id"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Housing dataframe is missing required columns: {missing}")

    df["city"] = normalize_city_name(df["city"])
    df["housing_cost"] = pd.to_numeric(df["housing_cost"], errors="coerce")
    df["beds"] = pd.to_numeric(df["beds"], errors="coerce")
    df["region_id"] = normalize_region_id(df["region_id"])

    df = df.dropna(subset=["housing_cost", "beds"])
    return df


def apply_city_filter(df: pd.DataFrame, selected_cities: List[str]) -> pd.DataFrame:
    if not selected_cities:
        return df.copy()
    return df[df["city"].isin(selected_cities)].copy()


def apply_industry_filter(jobs_df: pd.DataFrame, selected_industry: str) -> pd.DataFrame:
    if selected_industry == "All Available Industries":
        return jobs_df.copy()
    
    industry_map = {
        "Finance": "finance",
        "Healthcare": "healthcare",
        "Technology": "tech",
    }

    target = industry_map[selected_industry]
    return jobs_df[jobs_df["industry"] == target].copy()


def apply_career_stage_filter(jobs_df: pd.DataFrame, selected_career_stage: str) -> pd.DataFrame:
    if selected_career_stage == "All":
        return jobs_df.copy()

    target = selected_career_stage.strip().lower()
    return jobs_df[jobs_df["career_stage"] == target].copy()


def apply_job_region_filter(jobs_df: pd.DataFrame, selected_region_ids: List[int]) -> pd.DataFrame:
    if not selected_region_ids:
        return jobs_df.copy()
    return jobs_df[jobs_df["region_id"].isin(selected_region_ids)].copy()


def apply_housing_preference_filter(
    housing_df: pd.DataFrame,
    housing_preference: str,
) -> pd.DataFrame:
    if housing_preference == "Open":
        return housing_df.copy()

    filtered_parts = []

    for city_name, city_group in housing_df.groupby("city"):
        if city_group.empty:
            continue

        p35 = city_group["housing_cost"].quantile(0.35)
        p40 = city_group["housing_cost"].quantile(0.40)
        p60 = city_group["housing_cost"].quantile(0.60)
        p65 = city_group["housing_cost"].quantile(0.65)

        if housing_preference == "Lowest rent possible":
            city_filtered = city_group[city_group["housing_cost"] <= p40]

        elif housing_preference == "Fairly priced":
            city_filtered = city_group[
                (city_group["housing_cost"] >= p35) &
                (city_group["housing_cost"] <= p65)
            ]

        elif housing_preference == "Luxury":
            city_filtered = city_group[city_group["housing_cost"] >= p60]

        else:
            raise ValueError("Invalid housing preference received.")

        filtered_parts.append(city_filtered)

    if not filtered_parts:
        return housing_df.iloc[0:0].copy()

    return pd.concat(filtered_parts, ignore_index=True)

def get_region_ids_for_selected_cities(
        housing_df: pd.DataFrame,
        selected_cities: List[str],
) -> List[int]:
    if not selected_cities:
        return sorted(housing_df["region_id"].dropna().unique().tolist())
    
    filtered = housing_df[housing_df["city"].isin(selected_cities)].copy()
    return sorted(filtered["region_id"].dropna().unique().tolist())


def get_allowed_bedroom_values(
    household_size: int,
    housing_preference: str,
) -> Optional[Tuple[float, float]]:
    """
    Returns a min/max bedroom range for the selected household size and housing preference.
    None means 'no filter'.
    """

    if housing_preference == "Open":
        return None

    if household_size in (1, 2):
        if housing_preference in ("Lowest rent possible", "Fairly priced"):
            return (0, 1)
        if housing_preference == "Luxury":
            return (0, 2)

    elif household_size == 3:
        if housing_preference == "Lowest rent possible":
            return (1, 2)
        if housing_preference == "Fairly priced":
            return (2, 2)
        if housing_preference == "Luxury":
            return (2, float("inf"))

    elif household_size == 4:
        if housing_preference == "Lowest rent possible":
            return (2, 2)
        if housing_preference == "Fairly priced":
            return (2, 3)
        if housing_preference == "Luxury":
            return (3, float("inf"))

    elif household_size >= 5:
        if housing_preference == "Lowest rent possible":
            return (2, 3)
        if housing_preference in ("Fairly priced", "Luxury"):
            return (3, float("inf"))

    return None


def apply_household_size_filter(
    housing_df: pd.DataFrame,
    household_size: Optional[int],
    housing_preference: str,
    ignore_household_size: bool,
) -> pd.DataFrame:
    if ignore_household_size:
        return housing_df.copy()

    if household_size is None:
        return housing_df.copy()

    bedroom_range = get_allowed_bedroom_values(household_size, housing_preference)

    if bedroom_range is None:
        return housing_df.copy()

    min_beds, max_beds = bedroom_range

    if max_beds == float("inf"):
        return housing_df[housing_df["beds"] >= min_beds].copy()

    return housing_df[
        (housing_df["beds"] >= min_beds) &
        (housing_df["beds"] <= max_beds)
    ].copy()


def run_constraint_layer(
    housing_df: pd.DataFrame,
    jobs_df: pd.DataFrame,
    inputs: ConstraintInputs,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applies Layer 1 constraint rules and returns:
    - filtered housing dataframe
    - filtered jobs dataframe
    """
    validate_inputs(inputs)

    housing = standardize_housing_df(housing_df)
    jobs = standardize_jobs_df(jobs_df)

    # 1. City filtering for housing
    housing = apply_city_filter(housing, inputs.selected_cities)
    
    # 1b. Region-based filtering for jobs
    selected_region_ids = get_region_ids_for_selected_cities(housing, inputs.selected_cities)
    jobs = apply_job_region_filter(jobs, selected_region_ids)

    # 2. Employment filtering
    jobs = apply_industry_filter(jobs, inputs.selected_industry)
    jobs = apply_career_stage_filter(jobs, inputs.selected_career_stage)

    # 3. Housing cost preference filtering
    housing = apply_housing_preference_filter(housing, inputs.housing_preference)

    # 4. Household-size / bedroom filtering
    housing = apply_household_size_filter(
        housing_df=housing,
        household_size=inputs.household_size,
        housing_preference=inputs.housing_preference,
        ignore_household_size=inputs.ignore_household_size,
    )

    return housing.reset_index(drop=True), jobs.reset_index(drop=True)


def build_constraint_diagnostics(
    housing_df: pd.DataFrame,
    jobs_df: pd.DataFrame,
    inputs: ConstraintInputs,
) -> dict:
    """
    Applies the Layer 1 filters step-by-step and returns row-count diagnostics
    for both housing and jobs.
    """
    validate_inputs(inputs)

    housing = standardize_housing_df(housing_df)
    jobs = standardize_jobs_df(jobs_df)

    diagnostics = {
        "housing": {},
        "jobs": {},
    }

    # Initial counts
    diagnostics["housing"]["initial"] = len(housing)
    diagnostics["jobs"]["initial"] = len(jobs)

    # 1. City filter for housing
    housing = apply_city_filter(housing, inputs.selected_cities)
    diagnostics["housing"]["after_city_filter"] = len(housing)
    
    # 1b. Region-based filter for jobs
    selected_region_ids = get_region_ids_for_selected_cities(housing, inputs.selected_cities)
    jobs = apply_job_region_filter(jobs, selected_region_ids)
    diagnostics["jobs"]["after_city_filter"] = len(jobs)

    # 2. Job filters
    jobs = apply_industry_filter(jobs, inputs.selected_industry)
    diagnostics["jobs"]["after_industry_filter"] = len(jobs)

    jobs = apply_career_stage_filter(jobs, inputs.selected_career_stage)
    diagnostics["jobs"]["after_career_stage_filter"] = len(jobs)

    # 3. Housing preference filter
    housing = apply_housing_preference_filter(housing, inputs.housing_preference)
    diagnostics["housing"]["after_housing_preference_filter"] = len(housing)

    # 4. Household size filter
    housing = apply_household_size_filter(
        housing_df=housing,
        household_size=inputs.household_size,
        housing_preference=inputs.housing_preference,
        ignore_household_size=inputs.ignore_household_size,
    )
    diagnostics["housing"]["after_household_size_filter"] = len(housing)

    # Final counts
    diagnostics["housing"]["final"] = len(housing)
    diagnostics["jobs"]["final"] = len(jobs)

    return diagnostics



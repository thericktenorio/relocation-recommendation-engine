import pandas as pd


def load_housing_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    df["home_latitude"] = df["latitude"]
    df["home_longitude"] = df["longitude"]
    df["home_region_id"] = df["region_id"]

    return df


def load_job_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    df["job_latitude"] = df["latitude"]
    df["job_longitude"] = df["longitude"]
    df["job_region_id"] = df["region_id"]

    return df
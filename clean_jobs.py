import pandas as pd
import numpy as np

# Load raw data
df = pd.read_csv("data/raw/adzuna_jobs_raw.csv")

# Drop rows with no salary info (optional but recommended)
df = df.dropna(subset=["salary_min", "salary_max"])


'''
# 1. AVERAGE ANNUAL SALARY
'''
# Clean salary data
# ensure numeric types (essential for scrapped data)
df["salary_min"] = pd.to_numeric(df["salary_min"], errors = "coerce")
df["salary_max"] = pd.to_numeric(df["salary_max"], errors = "coerce")

# 1. If salary_min is 0 or NaN, then use value from salary_max (and vis versa)
df.loc[
    (df["salary_min"].isna()) | (df["salary_min"] == 0),
    "salary_min"
] = df["salary_max"]

df.loc[
    (df["salary_max"].isna()) | (df["salary_max"] == 0),
    "salary_max"
] = df["salary_min"]

# 2. Convert hourly wages to annual (assume < $100 = hourly)
def convert_to_annual(x):
    if pd.notna(x) and x < 999:
        return x * 2080
    return x

df["salary_min"] = df["salary_min"].apply(convert_to_annual)
df["salary_max"] = df["salary_max"].apply(convert_to_annual)

# 3. Compute annual salary as an average of min and max salary
df["annual_salary"] = ((df["salary_min"] + df["salary_max"]) / 2).round().astype("Int64")

# 4. Drop all rows where salary values couldn't be resolved
df = df.dropna(subset=["salary_min", "salary_max"])


'''
# 2. INDUSTRY MAPPING
'''
def map_industry(category, title):
    category = str(category).lower()
    title = str(title).lower()
    text = f"{category} {title}"

    if any(x in text for x in ["software", "engineer", "developer", "data", "tech"]):
        return "tech"
    elif any(x in text for x in ["nurse", "medical", "clinical", "health", "hospital"]):
        return "healthcare"
    elif any(x in text for x in ["account", "finance", "financial"]):
        return "finance"
    elif any(x in text for x in ["sales", "business", "business analyst"]):
        return "business"
    else:
        return "other"

df["industry"] = df.apply(lambda row: map_industry(row["category_label"], row["raw_job_title"]), axis = 1)


'''
# 3. ROLE FAMILY MAPPING
'''
def map_role(title):
    title = str(title).lower()

    if any(x in title for x in ["software", "engineer", "developer"]):
        return "software"
    elif any(x in title for x in ["data", " data analyst", "scientist"]):
        return "data"
    
    elif any(x in title for x in ["nurse", "medical", "clinical"]):
        return "nurse"
    elif any(x in title for x in ["technician", "lab"]):
        return "technician"
    
    elif any(x in title for x in ["account", "finance"]):
        return "accountant"
    elif any(x in title for x in ["sales", "business analyst", "marketing"]):
        return "business"
    
    else:
        return "other"

df["role_family"] = df["raw_job_title"].apply(map_role)


'''
# 4. OFFICE MODE ... on-site positions are assumed because API derived job descriptions are too truncated to allow proper office_mode ascertaining
'''
df["office_mode"] = "on-site"


'''
# 5. Career Stage (Salary derived values)
'''
def map_career_stage(salary):
    if pd.isna(salary):
        return "unknown"
    elif salary <= 70000:
        return "entry"
    elif salary <= 105000:
        return "mid"
    else:
        return "senior"

df["career_stage"] = df["annual_salary"].apply(map_career_stage)


'''
# 6. PLACEHOLDER JOB FIT SCORE
'''
df["job_fit_score"] = 0.7   # temp placeholder


'''
# 7. FINAL COLUMN SELECTION
'''
final_df = df[[
    "job_id",
    "region_id",
    "industry",
    "role_family",
    "raw_job_title",
    "career_stage",
    "annual_salary",
    "office_mode",
    "job_fit_score",
    "employer",
    "latitude",
    "longitude"
]]

# Drop duplicates again just in case
final_df = final_df.drop_duplicates(subset=["job_id"])

# Save cleaned dataset
final_df.to_csv("data/clean/adzuna_jobs_clean.csv", index=False)

print("Cleaned jobs saved.")
print(final_df.head())
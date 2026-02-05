import pandas as pd
import numpy as np


raw_data_path = "data/raw/kaggle_raw.csv"
output_path = "data/processed/employees_v1.csv"

random_seed = 42
np.random.seed(random_seed)

# experience (text -> years)
experience_map = {
    "Entry-level": 1,
    "Mid-level": 4,
    "Senior": 8,
    "Executive": 12
}

education_levels = ["Bachelor", "Master", "PhD"]


def map_experience_level(level: str) -> int:
    
    return experience_map.get(level, 3)  # fallback = 3 years


def generate_age(experience_years: int) -> int:
    
    base_age = np.random.randint(18, 26)
    return experience_years + base_age


def generate_certifications(job_category: str) -> int:
    
    if isinstance(job_category, str) and "Data" in job_category:
        return np.random.randint(1, 4)  # 1 to 3
    return np.random.randint(0, 3)  # 0 to 2


def generate_education_level() -> str:
    
    return np.random.choice(education_levels, p=[0.5, 0.4, 0.1])


def main():
    df = pd.read_csv(raw_data_path)

    # selecting the columns we want
    df = df[
        [
            "job_title",
            "job_category",
            "salary_in_usd",
            "experience_level"
        ]
    ].copy()

    # rename columns for project
    df.rename(
        columns={
            "job_title": "role",
            "job_category": "department",
            "salary_in_usd": "salary"
        },
        inplace=True
    )

    # Map experience_level -> experience_years
    df["experience_years"] = df["experience_level"].apply(map_experience_level)
    df.drop(columns=["experience_level"], inplace=True)

    df["age"] = df["experience_years"].apply(generate_age)
    df["certifications_count"] = df["department"].apply(generate_certifications)
    df["education_level"] = [generate_education_level() for _ in range(len(df))]

    # reorder columns
    df = df[
        [
            "age",
            "salary",
            "role",
            "department",
            "experience_years",
            "education_level",
            "certifications_count"
        ]
    ]

    df.to_csv(output_path, index=False)

    print("Dataset normalization complete!")
    print(f"dataset normalized, saved to: {output_path}")
    print("Sample rows:")
    print(df.head())

if __name__ == "__main__":
    main()
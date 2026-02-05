from pathlib import Path
import numpy as np
import pandas as pd

# Paths
input_path = Path("data/processed/employees_v1.csv")
output_path = Path("data/processed/employees_labeled_v1.csv")


def assign_missing_skill(row, role_salary_avg):
    experience = row["experience_years"]
    salary = row["salary"]
    role = row["role"]

    if experience < 2:
        return "SQL"

    if 2 <= experience <= 5 and salary < role_salary_avg.get(role, salary):
        return "ML"

    if experience > 5 and role not in ["Data Engineer", "ML Engineer"]:
        return "Cloud"

    return "No Major Gap"


def inject_label_noise(df, noise_ratio=0.15):
    np.random.seed(42)
    mask = np.random.rand(len(df)) < noise_ratio

    possible_skills = df["missing_skill"].unique()

    df.loc[mask, "missing_skill"] = np.random.choice(
        possible_skills, size=mask.sum()
    )

    return df


def label_dataset():
    df = pd.read_csv(input_path)

    # average salary per role
    role_salary_avg = df.groupby("role")["salary"].mean().to_dict()

    # assign labels
    df["missing_skill"] = df.apply(
        lambda row: assign_missing_skill(row, role_salary_avg),
        axis=1
    )

    # inject
    df = inject_label_noise(df, noise_ratio=0.15)

    df.to_csv(output_path, index=False)
    print(f"Labeled dataset saved to {output_path}")


if __name__ == "__main__":
    label_dataset()

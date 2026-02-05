from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

import joblib


# Paths
data_path = Path("data/processed/employees_labeled_v1.csv")
model_dir = Path("models")
model_path = model_dir / "skill_gap_model_v1.joblib"


# Training function
def train_model():
    df = pd.read_csv(data_path)

    # Features & target
    X = df.drop(columns=["missing_skill"])
    y = df["missing_skill"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Identify column types
    numeric_features = [
        "salary",
        "experience_years",
        "certifications_count",
    ]

    categorical_features = [
        "role",
        "department",
        "education_level",
    ]

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    # Model
    model = LogisticRegression(max_iter=1000)

    # Pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    # Train
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("Model trained successfully")
    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save model
    model_dir.mkdir(exist_ok=True)
    joblib.dump(pipeline, model_path)

    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    train_model()
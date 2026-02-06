# SkillBridge AI

SkillBridge AI is a **machine-learning powered FastAPI backend** that predicts the *most likely skill gap* for an employee based on their role, experience, salary, and education-related attributes.

This project is intentionally designed as a **full ML engineering pipeline**, not just a model notebook. It demonstrates how raw data is transformed, labeled, trained, and finally exposed via a production-ready API.


## What This Project Does (End-to-End)

1. Takes a raw dataset (CSV) with employee/job information
2. Cleans and normalizes the data
3. Applies **rule-based labeling logic** to infer a missing skill
4. Trains a classification model using Scikit-learn
5. Saves a versioned model artifact (`.joblib`)
6. Serves predictions through a FastAPI backend
7. Exposes interactive API documentation via Swagger (`/docs`)


## Problem Statement

In many organizations, skill gaps are identified informally or too late.
This project simulates a system that can:
* Analyze employee attributes
* Predict areas where a skill gap might exist
* Provide structured insights that could later be extended to training or upskilling recommendations

**Note:** The dataset and labels are synthetic / heuristic-driven and are used to demonstrate engineering design and ML workflow — not to claim real-world accuracy.


## Tech Stack

* **Python 3.12+**
* **FastAPI** – API framework
* **Scikit-learn** – ML modeling
* **Pandas & NumPy** – Data processing
* **Pydantic** – Input validation
* **Joblib** – Model persistence
* **Uvicorn** – ASGI server


## Project Structure

```
skillbridge-ai/
│
├── api/
│   ├── __init__.py          # Marks API as a Python package
│   └── main.py              # FastAPI application
│
├── data/
│   ├── raw/
│   │   └── kaggle_raw.csv   # Original dataset
│   └── processed/
│       ├── employees_v1.csv
│       └── employees_labeled_v1.csv
│
├── labeling/
│   ├── normalize_dataset.py # Cleans & normalizes raw data
│   └── label_skills.py      # Applies labeling logic
│
├── modeling/
│   └── train_model.py       # Trains & saves ML model
│
├── models/
│   └── skill_gap_model_v1.joblib  # Trained model artifact
│
├── requirements.txt
└── README.md
```

## Dataset

This project uses a synthetic employee dataset for learning and demonstration purposes. The dataset is not included in this repository due to size, privacy, and reproducibility reasons.
The data generation approach is based on examples from Kaggle. Only the data processing, labeling, and model training code are provided. Anyone cloning this repository can recreate the dataset using the included scripts.

## Data Pipeline

### 1️⃣ Normalization (`labeling/normalize_dataset.py`)

This script:

* Loads the raw CSV dataset
* Cleans column names
* Standardizes numeric columns (salary, experience, etc.)
* Outputs a clean dataset (`employees_v1.csv`)

Purpose: **Ensure consistent, model-ready data**


### 2️⃣ Labeling Logic (`labeling/label_skills.py`)

Because no real "missing skill" labels exist, a **rule-based heuristic** is used.

Example logic:

* Low experience → likely missing foundational skills (e.g., SQL)
* Mid experience + underpaid for role → likely ML gap
* High experience but non-engineering role → likely Cloud gap

The output is a new column:

```
missing_skill
```

This produces:

```
employees_labeled_v1.csv
```

This step intentionally simulates **real-world weak supervision**.


## Machine Learning Model

### Training (`modeling/train_model.py`)

* Features are split into:
  * Numeric columns (salary, experience, certifications)
  * Categorical columns (role, department, education)
* A Scikit-learn **Pipeline** is used
* Encoding + scaling + model training happen together
* Classifier: Logistic Regression

Why Logistic Regression?
* Interpretable
* Stable
* Suitable for baseline ML systems

The trained model is saved as:

```
models/skill_gap_model_v1.joblib
```


## FastAPI Backend

### Main Application (`api/main.py`)

The API:
* Loads the trained model at startup
* Validates inputs using Pydantic
* Converts request data into a Pandas DataFrame
* Returns predictions + confidence score


### Available Endpoints

#### Health Check

```
GET /health
```

Response:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

Purpose:
* Used for service monitoring
* Confirms that the API is running
* Commonly required in production deployments


#### Prediction Endpoint

```
POST /predict
```

**Request Body:**

```json
{
  "salary": 85000,
  "role": "Data Analyst",
  "department": "Analytics",
  "experience_years": 3,
  "education_level": "Bachelor",
  "certifications_count": 1
}
```

**Response:**

```json
{
  "missing_skill": "ML",
  "confidence": 0.82
}
```

* `missing_skill` → model prediction
* `confidence` → highest class probability


## Run Locally

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start API Server

```bash
uvicorn api.main:app --reload
```

### 3. Open Documentation

```
http://127.0.0.1:8000/docs
```


## Design Philosophy

This project emphasizes:
* Separation of concerns
* Reproducible data pipelines
* Explicit model versioning
* API-first ML deployment
* Clear, inspectable logic over black-box accuracy


## Notes on Accuracy

High accuracy on this dataset **does not imply real-world performance**.

Reasons:
* Labels are heuristic-based
* Data patterns are simplified
* No real human-annotated ground truth

This is intentional and aligns with the goal of demonstrating **engineering workflow**, not claims of predictive superiority.


Future Improvements
* Multi-skill prediction
* Course recommendation engine
* Persistent database (PostgreSQL)
* Model version tracking
* CI/CD pipeline
* Docker-based deployment


Author
Built as a portfolio project to demonstrate applied machine learning engineering, backend system design, and deployment readiness.

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
from pathlib import Path


app = FastAPI(
    title="SkillBridge AI",
    description="Predict employee skill gaps using ML"
)

model_path = Path("models/skill_gap_model_v1.joblib")

try:
    model = joblib.load(model_path)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")


class EmployeeInput(BaseModel):
    salary: float = Field(..., gt=0)
    role: str
    department: str
    experience_years: float = Field(..., ge=0)
    education_level: str
    certifications_count: int = Field(..., ge=0)


@app.post("/predict")
def predict_skill_gap(employee: EmployeeInput):
    try:
        df = pd.DataFrame([employee.model_dump()])

        prediction = model.predict(df)[0]
        confidence = model.predict_proba(df).max()

        return {
            "missing_skill": prediction,
            "confidence": round(float(confidence), 3)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": True
    }
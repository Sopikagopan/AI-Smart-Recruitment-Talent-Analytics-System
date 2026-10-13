from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from Backend.schemas import RecruitmentRequest


# Project paths
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "recruitment_model.keras"
SCALER_PATH = ROOT / "models" / "scaler.pkl"


# Load saved artifacts
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not SCALER_PATH.exists():
    raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


# Create API
app = FastAPI(
    title="AI Smart Recruitment and Talent Analytics API",
    description="Predicts candidate-job matching using text and experience features.",
    version="1.0.0",
)


class PredictionResponse(BaseModel):
    match_probability: float
    prediction: int
    result: str


@app.get("/")
def home():
    return {
        "message": "Smart Recruitment API is running",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: RecruitmentRequest):
    try:
        numeric_values = np.array([[
            request.candidate_experience_years,
            request.job_required_experience,
            request.experience_gap,
        ]], dtype=np.float32)

        scaled_values = scaler.transform(numeric_values).astype(
            np.float32
        )

        text_values = tf.constant(
            [request.combined_text],
            dtype=tf.string
        )

        probabilities = model.predict(
            {
                "combined_text": text_values,
                "numeric_features": scaled_values,
            },
            verbose=0,
        ).ravel()

        probability = float(probabilities[0])
        prediction = int(probability >= 0.5)

        return PredictionResponse(
            match_probability=round(probability, 4),
            prediction=prediction,
            result=(
                "Suitable Match"
                if prediction == 1
                else "Potential Mismatch"
            ),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        ) from exc

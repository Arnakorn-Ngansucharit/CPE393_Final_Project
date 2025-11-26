from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field # Pydantic import is correct

# ---------- Model discovery ----------
BASE_DIR = Path(__file__).resolve().parent
BEST_MODELS_DIR = BASE_DIR / "best_models"


def _resolve_model_path() -> Path:
    # Automatically select the most recent model
    candidates = sorted(BEST_MODELS_DIR.glob("*.pkl"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(
            f"No serialized models found under {BEST_MODELS_DIR}. "
            "Run train.py first to generate a best model."
        )
    return candidates[0] 

# Load model and determine feature names BEFORE Pydantic class definition
MODEL_PATH = _resolve_model_path()
model = joblib.load(MODEL_PATH)
FEATURE_NAMES: Optional[pd.Index] = getattr(model, "feature_names_in_", None)

# Create a dictionary for the required features to be used as a Pydantic example
REQUIRED_FEATURE_EXAMPLE = {feature: 0.0 for feature in FEATURE_NAMES} if FEATURE_NAMES is not None else {
    "lat": 0.0, "lon": 0.0, "aqi": 0.0, "pm25": 0.0, "pm10": 0.0,
    "o3": 0.0, "no2": 0.0, "so2": 0.0, "co": 0.0, "t": 0.0,
    "h": 0.0, "p": 0.0, "w": 0.0, "aqi_lag1": 0.0, "aqi_lag3": 0.0,
    "pm25_lag1": 0.0, "pm10_lag1": 0.0, "t_lag1": 0.0, "h_lag1": 0.0
}


# ---------- FastAPI wiring ----------
app = FastAPI(
    title="AQI Forecast Service",
    version="1.0.0",
    description="Serves the latest .pkl from best_models.",
)


class PredictionRequest(BaseModel):
    # Use the dynamically created example dictionary in the Field definition
    features: Dict[str, float] = Field(
        ...,  # Required field
        example=REQUIRED_FEATURE_EXAMPLE
    )

class PredictionResponse(BaseModel):
    model_file: str
    aqi_next1h: float


@app.get("/")
def root():
    return {
        "message": "AQI Forecast API is running",
        "model_file": MODEL_PATH.name,
        "required_features": list(FEATURE_NAMES) if FEATURE_NAMES is not None else "free-form",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    df = pd.DataFrame([payload.features])

    if FEATURE_NAMES is not None:
        # **RESTORED FEATURE VALIDATION LOGIC**
        missing = [col for col in FEATURE_NAMES if col not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing required features: {missing}")
        
        # Ensure feature order for prediction
        df = df[FEATURE_NAMES]

    try:
        pred = float(model.predict(df)[0])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return PredictionResponse(model_file=MODEL_PATH.name, aqi_next1h=pred)


# The section that manually updated the OpenAPI schema has been REMOVED
# because defining the 'example' directly on the Field is the correct, cleaner way
# and it successfully sets the example for the 'features' dictionary.

if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("deploy:app", host="0.0.0.0", port=port, reload=False)
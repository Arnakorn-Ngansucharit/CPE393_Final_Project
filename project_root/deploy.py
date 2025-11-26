from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---------- Model discovery ----------
BASE_DIR = Path(__file__).resolve().parent
BEST_MODELS_DIR = BASE_DIR / "best_models"
PREFERRED_MODEL = "RandomForestRegressor_20251124_220310.pkl"


def _resolve_model_path() -> Path:
    preferred_path = BEST_MODELS_DIR / PREFERRED_MODEL
    if preferred_path.exists():
        return preferred_path

    candidates = sorted(BEST_MODELS_DIR.glob("*.pkl"))
    if not candidates:
        raise FileNotFoundError(
            f"No serialized models found under {BEST_MODELS_DIR}. "
            "Run train.py first to generate a best model."
        )
    return candidates[-1]


MODEL_PATH = _resolve_model_path()
model = joblib.load(MODEL_PATH)
FEATURE_NAMES: Optional[pd.Index] = getattr(model, "feature_names_in_", None)

# ---------- FastAPI wiring ----------
app = FastAPI(
    title="AQI Forecast Service",
    version="1.0.0",
    description="Serves the latest RandomForestRegressor .pkl from best_models.",
)


class PredictionRequest(BaseModel):
    features: Dict[str, float]


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
        missing = [col for col in FEATURE_NAMES if col not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing required features: {missing}")
        df = df[FEATURE_NAMES]

    try:
        pred = float(model.predict(df)[0])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return PredictionResponse(model_file=MODEL_PATH.name, aqi_next1h=pred)


if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("deploy:app", host="0.0.0.0", port=port, reload=False)


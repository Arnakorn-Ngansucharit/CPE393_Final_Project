from fastapi import FastAPI
import mlflow
import pandas as pd

app = FastAPI()

MODEL_URI = "models:/aqi_best_model/Production"
model = mlflow.pyfunc.load_model(MODEL_URI)

@app.get("/")
def root():
    return {"message": "AQI Forecast API is running"}

@app.post("/predict")
def predict(features: dict):
    df = pd.DataFrame([features])
    pred = model.predict(df)
    return {"aqi_next1h": float(pred[0])}

"""
Check model performance by comparing predictions to actual within a tolerance.

This script computes the fraction of predictions whose absolute difference
to the true target (aqi_next1h) is <= tolerance. If the fraction is below
`--min-accuracy` the script will print `SHOULD_TRAIN=true` and write the
same to `check_model_performance_output.txt` so GitHub Actions can use it.

Usage: python project_root/check_model_performance.py [--tolerance 5] [--min-accuracy 0.7]
"""

import argparse
import sys
from pathlib import Path
import os

import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed" / "aqi_lagged_SEA.csv"
MODEL_NAME = "aqi_best_model"


def get_latest_model_from_best_models() -> str:
    best_models_dir = BASE_DIR / "best_models"
    if not best_models_dir.exists():
        raise ValueError("best_models directory does not exist")

    model_files = list(best_models_dir.glob("*.pkl"))
    if not model_files:
        raise ValueError("No model files found in best_models directory")

    latest_model = max(model_files, key=os.path.getmtime)
    return str(latest_model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tolerance", type=float, default=5.0, help="absolute tolerance on AQI (±)")
    parser.add_argument("--min-accuracy", type=float, default=0.7, help="minimum fraction within tolerance to skip retrain")
    args = parser.parse_args()

    out_file = BASE_DIR / "check_model_performance_output.txt"

    if not DATA_PATH.exists():
        print(f"DATA file not found: {DATA_PATH}")
        out_file.write_text("SHOULD_TRAIN=false\n")
        sys.exit(0)

    df = pd.read_csv(DATA_PATH)

    if "aqi_next1h" not in df.columns:
        print("target column 'aqi_next1h' not found in dataset")
        out_file.write_text("SHOULD_TRAIN=false\n")
        sys.exit(0)

    # prepare data similarly to train.py
    df = df.drop_duplicates()
    df = df.dropna(subset=["aqi_next1h"]) 
    df = df.dropna()

    y = df["aqi_next1h"].to_numpy()
    cols_to_drop = {"aqi_next1h", "station_idx", "station_name", "station_time"}
    cols_to_use = [c for c in df.columns if c not in cols_to_drop]
    X = df[cols_to_use]
    X = X.select_dtypes(include=[np.number])

    try:
        model_path = get_latest_model_from_best_models()
        print(f"Using model from: {model_path}")
    except Exception as e:
        print(f"Could not find latest model in best_models: {e}")
        out_file.write_text("SHOULD_TRAIN=true\n")
        sys.exit(0)

    try:
        import joblib
        model = joblib.load(model_path)
    except Exception as e:
        print(f"Could not load model from best_models: {e}")
        out_file.write_text("SHOULD_TRAIN=false\n")
        sys.exit(0)

    try:
        preds = model.predict(X)
    except Exception as e:
        print(f"Model prediction failed: {e}")
        out_file.write_text("SHOULD_TRAIN=false\n")
        sys.exit(0)

    preds = np.asarray(preds).reshape(-1)

    if preds.shape[0] != y.shape[0]:
        print(f"Prediction length {preds.shape[0]} != target length {y.shape[0]}")
        out_file.write_text("SHOULD_TRAIN=false\n")
        sys.exit(0)

    abs_diff = np.abs(y - preds)
    within = np.sum(abs_diff <= args.tolerance)
    frac_within = within / len(y) if len(y) > 0 else 0.0

    should_train = frac_within < args.min_accuracy

    out_lines = [f"FractionWithinTolerance={frac_within:.4f}", f"SHOULD_TRAIN={'true' if should_train else 'false'}"]
    print("\n".join(out_lines))
    out_file.write_text("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    main()

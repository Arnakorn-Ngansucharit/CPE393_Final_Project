# predict_all_stations.py

import sys
import subprocess
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"   # เปลี่ยนจาก DATA_PATH เป็น DATA_DIR
PRED_DIR = BASE_DIR / "data" / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "aqi_best_model"
TARGET_COL = "aqi_next1h"


def get_latest_model_uri(model_name: str) -> str:
    """ดึง model version ล่าสุดจาก Model Registry"""
    client = MlflowClient()

    # list versions ของโมเดลนี้ทั้งหมด
    versions = client.search_model_versions(f"name='{model_name}'")

    if not versions:
        raise ValueError(f"ไม่พบโมเดลชื่อ '{model_name}' ใน Model Registry")

    # sort ตาม version number (string → int)
    versions_sorted = sorted(versions, key=lambda v: int(v.version), reverse=True)

    latest_version = versions_sorted[0].version
    print(f"✔ ใช้ model version ล่าสุด = {latest_version}")

    return f"models:/{model_name}/{latest_version}"


def find_latest_dataset_path() -> Path:
    """
    เลือกไฟล์ training/feature dataset ล่าสุดจาก DATA_DIR
    pattern: aqi_lagged_SEA_YYYYMMDD_HHMMSS.csv

    ถ้าไม่เจอไฟล์แบบมี timestamp จะลองหาไฟล์เก่า aqi_lagged_SEA.csv เป็น fallback
    """
    pattern = "aqi_lagged_SEA_*.csv"
    files = sorted(DATA_DIR.glob(pattern))

    if files:
        latest = files[-1]  # YYYYMMDD_HHMMSS ทำให้ sort ตามเวลาได้ตรงอยู่แล้ว
        print(f"🔍 [PREDICT] พบไฟล์ dataset {len(files)} ไฟล์, ใช้ไฟล์ล่าสุด: {latest.name}")
        return latest

    legacy = DATA_DIR / "aqi_lagged_SEA.csv"
    if legacy.exists():
        print(f"⚠️ [PREDICT] ไม่พบไฟล์แบบมี timestamp ใช้ไฟล์ legacy แทน: {legacy.name}")
        return legacy

    raise FileNotFoundError(
        f"ไม่พบไฟล์ dataset ใน {DATA_DIR} "
        f"(ทั้ง pattern aqi_lagged_SEA_*.csv และ aqi_lagged_SEA.csv)"
    )


def load_latest_per_station(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ processed data ที่ {path}")

    df = pd.read_csv(path)
    print(f"โหลดข้อมูลจาก {path}, shape = {df.shape}")

    # แปลงเวลา
    if "station_time" in df.columns:
        df["station_time"] = pd.to_datetime(df["station_time"], errors="coerce")

    # ลบ NaN
    df = df.dropna()
    print(f"หลัง dropna แล้ว shape = {df.shape}")

    # sort แล้วเลือกแถวล่าสุดต่อ station
    df = df.sort_values(["station_idx", "station_time"])
    latest = df.groupby("station_idx", as_index=False).tail(1).reset_index(drop=True)
    print(f"เลือกเฉพาะแถวล่าสุดของแต่ละสถานี เหลือ {latest.shape[0]} แถว")
    return latest


def make_feature_matrix(df: pd.DataFrame):
    cols_to_drop = {
        TARGET_COL,
        "station_idx",
        "station_name",
        "station_time",
    }

    cols_to_use = [c for c in df.columns if c not in cols_to_drop]
    X = df[cols_to_use]

    # ใช้เฉพาะค่าตัวเลข
    X = X.select_dtypes(include=[np.number])

    print(f"ใช้ features จำนวน {X.shape[1]} คอลัมน์: {list(X.columns)}")
    return X


def get_latest_model_path() -> Path:
    """Find the latest model file in the best_models directory."""
    model_files = sorted((BASE_DIR / "best_models").glob("*.pkl"))

    if not model_files:
        raise FileNotFoundError("No model files found in the best_models directory.")

    latest_model = model_files[-1]  # The most recent model based on naming convention
    print(f"✔ Using the latest model: {latest_model.name}")
    return latest_model


def main():
    # 0) หาไฟล์โมเดลล่าสุด (ถ้าไม่มี ให้เทรนก่อนด้วย train.py)
    try:
        model_path = get_latest_model_path()
    except FileNotFoundError:
        print("⚠ No models found in 'best_models' -> Running train.py to create the first model...")
        subprocess.run([sys.executable, str(BASE_DIR / "train.py")], check=True)
        # ลองดึงโมเดลอีกครั้ง
        model_path = get_latest_model_path()

    # 1) หาไฟล์ dataset ล่าสุด แล้วโหลดข้อมูลล่าสุดของแต่ละสถานี
    data_path = find_latest_dataset_path()
    df_latest = load_latest_per_station(data_path)

    # 2) เตรียม feature matrix
    X = make_feature_matrix(df_latest)

    # 3) โหลดโมเดลจาก best_models directory
    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)

    # 4) predict
    preds = model.predict(X)

    # 5) Export ผลลัพธ์
    out = df_latest[["station_idx", "station_name", "lat", "lon", "station_time"]].copy()
    out["pred_aqi_next1h"] = preds

    out_path = PRED_DIR / "aqi_next1h_latest_stations.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\nบันทึกผลการพยากรณ์ AQI ชั่วโมงถัดไปสำหรับแต่ละสถานีที่: {out_path}")
    print("ตัวอย่างหัวตาราง:")
    print(out.head())


if __name__ == "__main__":
    main()
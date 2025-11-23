# predict_all_stations.py

from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed" / "aqi_lagged_SEA.csv"
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


def main():
    # 0) ดึง URI ของ model version ล่าสุด
    model_uri = get_latest_model_uri(MODEL_NAME)

    # 1) โหลดข้อมูลล่าสุดของแต่ละสถานี
    df_latest = load_latest_per_station(DATA_PATH)

    # 2) เตรียม feature matrix
    X = make_feature_matrix(df_latest)

    # 3) โหลดโมเดลจาก MLflow
    print(f"โหลดโมเดลจาก MLflow URI = {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)

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
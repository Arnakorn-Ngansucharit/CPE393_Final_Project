import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from mlflow.tracking import MlflowClient

# ================= CONFIG =================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"   # เปลี่ยนจาก DATA_PATH เป็น DATA_DIR

TARGET_COL = "aqi_next1h"
EXPERIMENT_NAME = "aqi_forecasting"

RANDOM_STATE = 42
TEST_SIZE = 0.2


# ================= DATA LOADING =================

def find_latest_dataset_path() -> Path:
    """
    เลือกไฟล์ training dataset ล่าสุดจาก DATA_DIR
    pattern: aqi_lagged_SEA_YYYYMMDD_HHMMSS.csv

    ถ้าไม่เจอไฟล์แบบมี timestamp จะลองหาไฟล์เก่า aqi_lagged_SEA.csv เป็น fallback
    """
    pattern = "aqi_lagged_SEA_*.csv"
    files = sorted(DATA_DIR.glob(pattern))

    if files:
        latest = files[-1]  # เพราะ YYYYMMDD_HHMMSS ทำให้ sort ตามเวลาพอดี
        print(f"🔍 พบไฟล์ training dataset {len(files)} ไฟล์, ใช้ไฟล์ล่าสุด: {latest.name}")
        return latest

    # fallback: ใช้ชื่อเก่าแบบ fix
    legacy = DATA_DIR / "aqi_lagged_SEA.csv"
    if legacy.exists():
        print(f"⚠️ ไม่พบไฟล์แบบมี timestamp ใช้ไฟล์ legacy แทน: {legacy.name}")
        return legacy

    raise FileNotFoundError(
        f"ไม่พบไฟล์ training dataset ใน {DATA_DIR} "
        f"(ทั้ง pattern aqi_lagged_SEA_*.csv และ aqi_lagged_SEA.csv)"
    )


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ data ที่ {path}")
    df = pd.read_csv(path)
    print(f"โหลดข้อมูลจาก {path}, shape = {df.shape}")
    return df


def preprocess(df: pd.DataFrame):
    # ลบแถวซ้ำทั้งหมด
    df = df.drop_duplicates()
    print(f"หลังลบแถวซ้ำแล้ว shape = {df.shape}")
    
    # ลบแถวที่ target หาย
    if TARGET_COL not in df.columns:
        raise ValueError(f"ไม่พบคอลัมน์ target '{TARGET_COL}' ใน dataset")

    df = df.dropna(subset=[TARGET_COL])

    # ลบแถวที่ feature มี NaN (แบบง่าย ๆ ตามที่ต้องการ)
    df = df.dropna()

    print(f"หลัง dropna แล้ว shape = {df.shape}")

    # แยก X, y
    y = df[TARGET_COL]

    # ตัดคอลัมน์ที่ไม่อยากใช้เป็น feature
    cols_to_drop = {
        TARGET_COL,
        "station_idx",
        "station_name",
        "station_time",
    }
    cols_to_use = [c for c in df.columns if c not in cols_to_drop]

    X = df[cols_to_use]

    # เลือกเฉพาะคอลัมน์ตัวเลขกันเหนียว
    X = X.select_dtypes(include=[np.number])

    print(f"ใช้ features จำนวน {X.shape[1]} คอลัมน์: {list(X.columns)}")

    return X, y


# ================= TRAINING =================

def train_and_log_model(model_name: str, model, X_train, X_test, y_train, y_test):
    """
    เทรน model แล้ว log เข้า MLflow 1 run
    """
    with mlflow.start_run(run_name=model_name):
        # log ชื่อโมเดล
        mlflow.set_tag("model_name", model_name)

        # log parameters (เท่าที่ช่วยได้)
        if hasattr(model, "get_params"):
            mlflow.log_params(model.get_params())

        # train
        model.fit(X_train, y_train)

        # predict
        y_pred = model.predict(X_test)

        # metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        r2 = r2_score(y_test, y_pred)

        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)

        print(f"[{model_name}] MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")

        # log model
        mlflow.sklearn.log_model(model, artifact_path="model")

        # คืน metrics ไว้เอาไปเลือก best model ต่อ
        run_id = mlflow.active_run().info.run_id
        return {
            "model_name": model_name,
            "run_id": run_id,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        }


def main():
    # ---------- เตรียม MLflow experiment ----------
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"ใช้ MLflow experiment: {EXPERIMENT_NAME}")

    # ---------- หาไฟล์ล่าสุด & โหลด & preprocess data ----------
    data_path = find_latest_dataset_path()
    df = load_dataset(data_path)
    X, y = preprocess(df)

    if len(X) < 50:
        print("⚠ จำนวน sample น้อย (< 50) ระวังโมเดล overfit เทรนให้สำหรับลอง pipeline ก่อน")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print(f"Train size = {X_train.shape[0]} rows, Test size = {X_test.shape[0]} rows")

    # ---------- สร้างโมเดลทั้ง 3 ----------
    models = [
        ("LinearRegression", LinearRegression()),
        ("RandomForestRegressor", RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
        ("GradientBoostingRegressor", GradientBoostingRegressor(
            random_state=RANDOM_STATE
        )),
    ]

    results = []
    for name, model in models:
        print(f"\n===== Train {name} =====")
        res = train_and_log_model(name, model, X_train, X_test, y_train, y_test)
        results.append(res)

    # ---------- สรุปผล และหา best model ----------
    print("\n===== Summary (sorted by RMSE) =====")
    results_sorted = sorted(results, key=lambda r: r["rmse"])
    for r in results_sorted:
        print(
            f"{r['model_name']:25s} | RMSE={r['rmse']:.4f} | MAE={r['mae']:.4f} | R2={r['r2']:.4f} | run_id={r['run_id']}"
        )

    best = results_sorted[0]
    print("\nBest model (RMSE ต่ำสุด):")
    print(
        f"{best['model_name']} (run_id={best['run_id']}) "
        f"RMSE={best['rmse']:.4f}, MAE={best['mae']:.4f}, R2={best['r2']:.4f}"
    )

    # ---------- Auto-register best model เข้า Model Registry ----------
    model_name = "aqi_best_model"
    model_uri = f"runs:/{best['run_id']}/model"

    print(f"\nRegister best model เข้า Model Registry ชื่อ '{model_name}' ...")
    registered = mlflow.register_model(model_uri=model_uri, name=model_name)
    version = registered.version
    print(f"   -> registered version = {version}")

    print("\nเสร็จแล้ว: best model ถูก register เรียบร้อย")
    print(f"   Model URI สำหรับใช้ deploy: models:/{model_name}/{version}")
    print("   ถ้าอยากดูใน UI ให้รัน: mlflow ui")


if __name__ == "__main__":
    main()
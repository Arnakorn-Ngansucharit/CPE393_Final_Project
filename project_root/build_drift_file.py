# build_drift_file.py
# สร้างไฟล์สำหรับ data drift detection จาก clean daily data

from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent

DAILY_DIR = BASE_DIR / "data" / "clean" / "daily"
DRIFT_DIR = BASE_DIR / "data" / "drift"
DRIFT_DIR.mkdir(parents=True, exist_ok=True)


def load_all_daily() -> pd.DataFrame:
    """โหลด daily data ทั้งหมด"""
    files = sorted(DAILY_DIR.glob("waqi_daily_SEA_*.csv"))
    if not files:
        raise FileNotFoundError(f"ไม่พบ daily files ใน {DAILY_DIR}")

    dfs = []
    for f in files:
        print(f"[DRIFT] โหลด {f.name}")
        df = pd.read_csv(f)
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    print(f"รวม daily data ได้ทั้งหมด {len(df_all)} แถว")
    return df_all


def prepare_drift_features(df: pd.DataFrame) -> pd.DataFrame:
    """เตรียม features สำหรับ drift detection"""
    df = df.copy()

    # แปลง datetime columns
    if "station_time" in df.columns:
        df["station_time"] = pd.to_datetime(df["station_time"], errors="coerce")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # เลือก numeric features ที่ใช้ในโมเดล (ตาม train.py)
    # ตัดคอลัมน์ที่ไม่ใช่ feature ออก
    cols_to_drop = {
        "station_idx",
        "station_name",
        "station_url",
        "station_time",
        "station_tz",
        "station_unix",
        "ingested_at_utc",
        "global_uid",
        "global_aqi_snapshot",
        "global_snapshot_utc",
        "dominentpol",  # categorical
    }

    # เลือกเฉพาะ numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # ลบ columns ที่ไม่ต้องการ
    feature_cols = [c for c in numeric_cols if c not in cols_to_drop]
    
    # แบ่ง features เป็น core (สำคัญ) และ optional (อาจมี missing)
    core_features = ["lat", "lon", "aqi", "pm25", "pm10", "t", "h", "p"]
    core_features = [c for c in core_features if c in feature_cols]
    
    optional_features = [c for c in feature_cols if c not in core_features]
    
    # เพิ่ม date สำหรับ tracking (ถ้ามี)
    metadata_cols = []
    if "date" in df.columns:
        metadata_cols.append("date")

    # สร้าง dataframe สำหรับ drift
    drift_df = df[feature_cols + metadata_cols].copy()

    # ลบแถวที่มี NaN ใน core features เท่านั้น (อนุญาตให้ optional features มี missing)
    drift_df = drift_df.dropna(subset=core_features)

    # สำหรับ optional features ที่มี missing มากเกินไป (>80%), ลบออก
    cols_to_keep = []
    for col in feature_cols:
        if col in core_features:
            cols_to_keep.append(col)
        else:
            missing_pct = drift_df[col].isnull().sum() / len(drift_df)
            if missing_pct < 0.8:  # ถ้า missing < 80% เก็บไว้
                cols_to_keep.append(col)
            else:
                print(f"   - ลบ {col} (missing {missing_pct*100:.1f}%)")

    # เลือกเฉพาะ columns ที่ต้องการ
    final_cols = [c for c in cols_to_keep + metadata_cols if c in drift_df.columns]
    drift_df = drift_df[final_cols].copy()

    print(f"ใช้ features จำนวน {len(cols_to_keep)} คอลัมน์")
    print(f"   Core features: {core_features}")
    print(f"   Optional features: {[c for c in cols_to_keep if c not in core_features]}")
    print(f"หลัง dropna (core features) แล้ว shape = {drift_df.shape}")

    return drift_df


def main():
    """สร้างไฟล์สำหรับ data drift detection"""
    print("=" * 60)
    print("สร้างไฟล์สำหรับ Data Drift Detection")
    print("=" * 60)

    # โหลด daily data
    df_all = load_all_daily()

    # เตรียม features สำหรับ drift
    df_drift = prepare_drift_features(df_all)

    # บันทึกไฟล์
    output_path = DRIFT_DIR / "waqi_drift_reference.csv"
    df_drift.to_csv(output_path, index=False)
    
    print(f"\n🎉 บันทึกไฟล์ drift reference ที่: {output_path}")
    print(f"   Shape: {df_drift.shape}")
    print(f"   Columns: {list(df_drift.columns)}")
    print(f"\n💡 ใช้ไฟล์นี้เป็น reference dataset สำหรับ Evidently AI")


if __name__ == "__main__":
    main()


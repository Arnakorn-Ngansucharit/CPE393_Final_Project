# clean_hourly_timeseries.py

from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent

RAW_TS_DIR = BASE_DIR / "data" / "raw" / "waqi_timeseries"
CLEAN_HOURLY_DIR = BASE_DIR / "data" / "clean" / "hourly"
CLEAN_HOURLY_DIR.mkdir(parents=True, exist_ok=True)


def clean_single_file(path: Path):
    print(f"[CLEAN] โหลดไฟล์ดิบ: {path}")
    df = pd.read_csv(path)

    # ต้องมี station_time
    if "station_time" not in df.columns:
        print("⚠ ไม่มีคอลัมน์ station_time ข้ามไฟล์นี้ไป")
        return None

    # แปลงเวลา
    df["station_time"] = pd.to_datetime(df["station_time"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["station_time"])
    print(f"   - ลบแถวที่เวลาเป็น NaT: {before} → {len(df)}")

    # ลบ duplicate ตาม station_idx + station_time (ถ้ามี)
    if "station_idx" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["station_idx", "station_time"])
        print(f"   - ลบ duplicate: {before} → {len(df)}")

    # ลบค่าติดลบในค่ามลพิษ
    numeric_cols = ["aqi", "pm25", "pm10", "o3", "no2", "so2", "co"]
    for col in numeric_cols:
        if col in df.columns:
            before = len(df)
            df = df[(df[col].isna()) | (df[col] >= 0)]
            print(f"   - กรอง {col} >= 0: {before} → {len(df)}")

    # sort ตาม station_idx + station_time
    sort_cols = [c for c in ["station_idx", "station_time"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    # เซฟเป็นไฟล์ clean/hourly ด้วยชื่อเดิม
    out_path = CLEAN_HOURLY_DIR / path.name
    df.to_csv(out_path, index=False)
    print(f"   → บันทึกไฟล์คลีนแล้ว: {out_path}\n")
    return out_path


def main():
    files = sorted(RAW_TS_DIR.glob("waqi_timeseries_SEA_*.csv"))
    if not files:
        print(f"ไม่พบไฟล์ใน {RAW_TS_DIR}")
        return

    print(f"พบไฟล์ดิบทั้งหมด {len(files)} ไฟล์\n")
    for f in files:
        clean_single_file(f)

    print("🎉 ทำความสะอาดข้อมูลรายชั่วโมงเสร็จแล้ว")


if __name__ == "__main__":
    main()

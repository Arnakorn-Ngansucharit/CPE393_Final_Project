# process_aqi.py

import os
from pathlib import Path
import pandas as pd
from typing import List

import re
from datetime import datetime

# ========= CONFIG =========
RAW_FILENAME_PATTERN = r"waqi_timeseries_SEA_(\d{8})_(\d{6})\.csv"

BASE_DIR = Path(__file__).resolve().parent

RAW_DIR = BASE_DIR / "data" / "raw" / "waqi_timeseries"
PROCESSED_HOURLY_DIR = BASE_DIR / "data" / "clean" / "hourly"   # เป็น root ของโฟลเดอร์รายวัน
PROCESSED_DAILY_DIR = BASE_DIR / "data" / "clean" / "daily"

# ใช้ชื่อ column เดียวกับ clean_single_file
DATETIME_COL = "station_time"
LOCATION_COL = "station_idx"


# สร้างโฟลเดอร์ถ้ายังไม่มี
for d in [RAW_DIR, PROCESSED_HOURLY_DIR, PROCESSED_DAILY_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ========= CLEANING =========

def clean_aqi(df: pd.DataFrame) -> pd.DataFrame:
    """
    ฟังก์ชันคลีนใช้ร่วมกันทั้ง hourly และ daily
    Logic อิงจาก clean_single_file()
    """
    df = df.copy()

    # ต้องมี station_time
    if DATETIME_COL not in df.columns:
        print(f"⚠ ไม่มีคอลัมน์ {DATETIME_COL} ข้าม dataframe นี้ไป")
        return df.iloc[0:0].copy()  # คืน df ว่าง ๆ

    # แปลงเวลา
    df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL], errors="coerce")
    before = len(df)
    df = df.dropna(subset=[DATETIME_COL])
    print(f"   - ลบแถวที่เวลาเป็น NaT: {before} → {len(df)}")

    # ลบ duplicate ตาม station_idx + station_time (ถ้ามี LOCATION_COL)
    if LOCATION_COL in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=[LOCATION_COL, DATETIME_COL])
        print(f"   - ลบ duplicate ({LOCATION_COL}, {DATETIME_COL}): {before} → {len(df)}")

    # ลบค่าติดลบในค่ามลพิษ
    pollutant_cols = ["aqi", "pm25", "pm10", "o3", "no2", "so2", "co"]
    for col in pollutant_cols:
        if col in df.columns:
            before = len(df)
            df = df[(df[col].isna()) | (df[col] >= 0)]
            print(f"   - กรอง {col} >= 0: {before} → {len(df)}")

    # แปลง numeric type ให้แน่ใจ
    numeric_extra = ["temperature", "humidity"]
    for col in pollutant_cols + numeric_extra:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # sort ตาม station_idx + station_time (ถ้ามี)
    sort_cols = [c for c in [LOCATION_COL, DATETIME_COL] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    return df


# ========= HELPER =========

def extract_datetime_from_filename(filename: str) -> datetime:
    """
    Example: waqi_timeseries_SEA_20251124_163000.csv
    Returns datetime(2025, 11, 24, 16, 30, 0)
    """
    m = re.match(RAW_FILENAME_PATTERN, filename)
    if not m:
        raise ValueError(f"Filename does not match pattern: {filename}")

    date_part = m.group(1)   # 20251124
    time_part = m.group(2)   # 163000

    return datetime.strptime(date_part + time_part, "%Y%m%d%H%M%S")


def find_latest_raw_file() -> Path:
    """
    หาไฟล์ raw ล่าสุดจาก RAW_DIR (ตามชื่อ waqi_timeseries_SEA_YYYYMMDD_HHMMSS.csv)
    """
    files = sorted(RAW_DIR.glob("waqi_timeseries_SEA_*.csv"))
    if not files:
        raise FileNotFoundError(f"ไม่พบไฟล์ raw ใน {RAW_DIR}")

    latest = files[-1]
    try:
        rel = latest.relative_to(BASE_DIR)
    except ValueError:
        rel = latest
    print(f"[RAW] พบไฟล์ล่าสุด: {rel}")
    return latest


def save_hourly_clean(df: pd.DataFrame, run_time: datetime) -> Path:
    """
    เซฟไฟล์ hourly ที่คลีนแล้วไปไว้ในโฟลเดอร์รายวัน เช่น:
    data/clean/hourly/20251124/waqi_cleaned_20251124_093000.csv
    """
    date_str = run_time.strftime('%Y%m%d')
    day_dir = PROCESSED_HOURLY_DIR / date_str
    day_dir.mkdir(parents=True, exist_ok=True)

    fname = f"waqi_cleaned_{run_time.strftime('%Y%m%d_%H%M%S')}.csv"
    out_path = day_dir / fname
    df.to_csv(out_path, index=False)
    print(f"   → บันทึกไฟล์คลีนแล้ว (hourly): {out_path}")
    return out_path


def hourly_files_of(date: datetime) -> List[Path]:
    """
    ดึงไฟล์ hourly ทั้งหมดของวันนั้น จากโฟลเดอร์:
    data/clean/hourly/YYYYMMDD/waqi_cleaned_YYYYMMDD_*.csv
    """
    date_prefix = date.strftime("%Y%m%d")
    day_dir = PROCESSED_HOURLY_DIR / date_prefix
    if not day_dir.exists():
        return []

    pattern = f"waqi_cleaned_{date_prefix}_*.csv"
    return sorted(day_dir.glob(pattern))


def build_daily(date: datetime) -> Path:
    """
    รวมไฟล์ hourly ที่คลีนแล้วของวันนั้นทั้งหมดมาต่อกัน (concat)
    ไม่ทำ groupby, ไม่สรุปสถิติ
    และลบแถวซ้ำตาม station_idx + station_time
    """
    files = hourly_files_of(date)
    if not files:
        raise FileNotFoundError(f"No hourly files found for {date}")

    print(f"[DAILY] วันที่ {date.strftime('%Y-%m-%d')}")
    print(f"        พบไฟล์ hourly ที่จะนำมารวมทั้งหมด: {len(files)} ไฟล์")
    for f in files:
        try:
            rel = f.relative_to(BASE_DIR)
        except ValueError:
            rel = f
        print(f"        - {rel}")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL], errors="coerce")
        df = df.dropna(subset=[DATETIME_COL])
        dfs.append(df)

    # ต่อข้อมูลทุกชั่วโมงของวันนั้นเข้าเป็นก้อนเดียว
    df_day = pd.concat(dfs, ignore_index=True)

    # ลบ row ซ้ำ
    if LOCATION_COL in df_day.columns:
        before = len(df_day)
        df_day = df_day.drop_duplicates(subset=[LOCATION_COL, DATETIME_COL])
        after = len(df_day)
        print(f"        - ลบ row ซ้ำใน daily: {before} → {after}")
    else:
        print("        - ไม่มีคอลัมน์ station_idx → ข้ามขั้นตอนลบซ้ำ")

    # เรียงลำดับตามสถานี + เวลา
    sort_cols = [c for c in [LOCATION_COL, DATETIME_COL] if c in df_day.columns]
    if sort_cols:
        df_day = df_day.sort_values(sort_cols).reset_index(drop=True)

    out_path = PROCESSED_DAILY_DIR / f"waqi_daily_SEA_{date.strftime('%Y%m%d')}.csv"
    df_day.to_csv(out_path, index=False)
    print(f"   → บันทึกไฟล์ daily (concat hourly): {out_path}")
    return out_path


# ========= MAIN FUNCTION =========

def process_new_raw_file(raw_file_path: Path):
    """
    1. อ่าน raw hourly
    2. clean → save hourly cleaned
    3. update daily summary
    """
    print(f"[CLEAN] โหลดไฟล์ดิบ: {raw_file_path}")
    df_raw = pd.read_csv(raw_file_path)

    # clean ด้วย logic เดียวกับ clean_single_file
    df_clean = clean_aqi(df_raw)

    # ถ้า clean แล้วว่างเปล่า ไม่ต้องไปต่อ
    if df_clean.empty:
        print("⚠ dataframe หลังคลีนว่างเปล่า ไม่สร้างไฟล์ต่อ")
        return

    raw_filename = raw_file_path.name
    run_time = extract_datetime_from_filename(raw_filename)

    # save hourly cleaned -> โฟลเดอร์รายวัน
    save_hourly_clean(df_clean, run_time)

    # update daily
    affected_dates = df_clean[DATETIME_COL].dt.date.unique()
    for d in affected_dates:
        build_daily(pd.Timestamp(d))


# ========= CLI MODE =========

if __name__ == "__main__":
    import sys

    # ถ้า user ใส่ path มา -> ใช้ path นั้น
    # ถ้าไม่ใส่ argument -> หาไฟล์ raw ล่าสุดจาก RAW_DIR มาใช้เอง
    if len(sys.argv) == 2:
        raw_path = Path(sys.argv[1])
        # ถ้าเป็น path relative ให้ถือว่า relative กับ BASE_DIR
        if not raw_path.is_absolute():
            raw_path = (BASE_DIR / raw_path).resolve()
    elif len(sys.argv) == 1:
        raw_path = find_latest_raw_file()
    else:
        print("ใช้แบบนี้: python process_aqi.py [path/to/raw_file.csv]")
        sys.exit(1)

    process_new_raw_file(raw_path)
    print("✔️ เสร็จ: hourly + daily ถูกอัปเดตเรียบร้อย")
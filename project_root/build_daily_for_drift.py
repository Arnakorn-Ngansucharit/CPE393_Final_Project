# build_daily_for_drift.py

from pathlib import Path
import pandas as pd
import shutil

BASE_DIR = Path(__file__).resolve().parent

CLEAN_HOURLY_DIR = BASE_DIR / "data" / "clean" / "hourly"
DAILY_DIR = BASE_DIR / "data" / "clean" / "daily"
DAILY_DIR.mkdir(parents=True, exist_ok=True)


def load_all_clean_hourly() -> pd.DataFrame:
    files = sorted(CLEAN_HOURLY_DIR.glob("waqi_timeseries_SEA_*.csv"))
    if not files:
        raise FileNotFoundError(f"ไม่พบไฟล์ใน {CLEAN_HOURLY_DIR}")

    dfs = []
    for f in files:
        print(f"[DAILY] โหลด {f}")
        df = pd.read_csv(f)
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    if "station_time" not in df_all.columns:
        raise ValueError("ไม่พบคอลัมน์ station_time ใน hourly data")

    df_all["station_time"] = pd.to_datetime(df_all["station_time"], errors="coerce")
    df_all = df_all.dropna(subset=["station_time"])
    return df_all


def main():
    df_all = load_all_clean_hourly()
    print(f"รวม clean hourly data: {len(df_all)} แถว")

    # เพิ่มคอลัมน์ date
    df_all["date"] = df_all["station_time"].dt.date

    unique_dates = sorted(df_all["date"].unique())
    print(f"เจอวันที่ทั้งหมด {len(unique_dates)} วัน: {unique_dates}")

    for d in unique_dates:
        df_day = df_all[df_all["date"] == d].copy()
        date_str = str(d).replace("-", "")  # YYYYMMDD
        out_path = DAILY_DIR / f"waqi_daily_SEA_{date_str}.csv"
        df_day.to_csv(out_path, index=False)
        print(f"→ บันทึกไฟล์รายวัน {d}: {out_path} ({len(df_day)} แถว)")

    # ลบ hourly ทิ้งเพื่อประหยัดพื้นที่
    print("\n🧹 ลบข้อมูล hourly ทั้งหมดเพื่อประหยัดพื้นที่...")
    if CLEAN_HOURLY_DIR.exists():
        shutil.rmtree(CLEAN_HOURLY_DIR)
    CLEAN_HOURLY_DIR.mkdir(parents=True, exist_ok=True)
    print(f"   - เคลียร์โฟลเดอร์ {CLEAN_HOURLY_DIR} แล้ว")

    print("\n🎉 สร้างไฟล์รายวันสำหรับ data drift เสร็จแล้ว")
    print(f"ไฟล์อยู่ที่: {DAILY_DIR}")


if __name__ == "__main__":
    main()

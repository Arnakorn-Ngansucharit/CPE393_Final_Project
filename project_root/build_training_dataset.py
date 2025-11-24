from pathlib import Path
import pandas as pd
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent

HOURLY_DIR = BASE_DIR / "data" / "clean" / "hourly"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_all_daily() -> pd.DataFrame:
    """
    โหลดไฟล์ hourly ที่คลีนแล้วทั้งหมดจาก HOURLY_DIR
    (รองรับโฟลเดอร์รายวัน: data/clean/hourly/YYYYMMDD/waqi_cleaned_*.csv)
    แล้ว concat รวมเป็น DataFrame เดียว พร้อมลบ row ซ้ำ
    """
    # ใช้ rglob เพื่อไล่หาใน subfolder ทุกวัน
    files = sorted(HOURLY_DIR.rglob("waqi_cleaned_*.csv"))
    if not files:
        raise FileNotFoundError(f"ไม่พบ hourly files ใน {HOURLY_DIR}")

    print(f"[TRAIN-DATA] พบไฟล์ hourly ที่จะใช้สร้าง training dataset ทั้งหมด {len(files)} ไฟล์")
    for f in files:
        try:
            rel = f.relative_to(BASE_DIR)
        except ValueError:
            rel = f
        print(f"   - {rel}")

    dfs = []
    for f in files:
        print(f"[TRAIN-DATA] โหลด {f}")
        df = pd.read_csv(f)
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    # ลบ row ซ้ำ
    before = len(df_all)
    subset = [c for c in ["station_idx", "station_time"] if c in df_all.columns]
    if subset:
        df_all = df_all.drop_duplicates(subset=subset)
        after = len(df_all)
        print(f"[TRAIN-DATA] ลบ row ซ้ำตามคีย์ {subset}: {before} → {after}")
    else:
        df_all = df_all.drop_duplicates()
        after = len(df_all)
        print(f"[TRAIN-DATA] ลบ row ซ้ำ (ทุกคอลัมน์): {before} → {after}")

    if "station_time" not in df_all.columns:
        raise ValueError("ไม่พบคอลัมน์ station_time ใน hourly data")

    df_all["station_time"] = pd.to_datetime(df_all["station_time"], errors="coerce")
    df_all = df_all.dropna(subset=["station_time"])

    # sort ให้กลายเป็น time series ต่อเนื่อง
    sort_cols = [c for c in ["station_idx", "station_time"] if c in df_all.columns]
    df_all = df_all.sort_values(sort_cols).reset_index(drop=True)

    print(f"[TRAIN-DATA] รวม hourly data ได้ทั้งหมด {len(df_all)} แถว (หลัง sort แล้ว)")
    return df_all


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    cols_keep = [
        "station_idx", "station_name", "lat", "lon",
        "station_time", "aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
        "t", "h", "p", "w",
    ]
    cols_keep = [c for c in cols_keep if c in df.columns]
    df = df[cols_keep]

    def create_lags(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("station_time")

        # target: AQI ใน 1 ชั่วโมงถัดไป
        group["aqi_next1h"] = group["aqi"].shift(-1)

        # lag features หลัก ๆ
        group["aqi_lag1"] = group["aqi"].shift(1)
        group["aqi_lag3"] = group["aqi"].shift(3)

        if "pm25" in group.columns:
            group["pm25_lag1"] = group["pm25"].shift(1)
        if "pm10" in group.columns:
            group["pm10_lag1"] = group["pm10"].shift(1)
        if "t" in group.columns:
            group["t_lag1"] = group["t"].shift(1)
        if "h" in group.columns:
            group["h_lag1"] = group["h"].shift(1)

        return group

    if "station_idx" in df.columns:
        df = df.groupby("station_idx", group_keys=False).apply(create_lags)
    else:
        df = create_lags(df)

    must_have = [c for c in ["aqi_next1h", "aqi_lag1"] if c in df.columns]
    df = df.dropna(subset=must_have)

    print(f"[TRAIN-DATA] หลังทำ lag + target เหลือ {len(df)} แถว สำหรับเทรน")
    return df


def main():
    df_all = load_all_daily()
    df_lagged = add_lag_features(df_all)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = PROCESSED_DIR / f"aqi_lagged_SEA_{ts}.csv"

    df_lagged.to_csv(output_csv, index=False)
    print(f"🎉 บันทึก training dataset ที่: {output_csv}")


if __name__ == "__main__":
    main()
# build_training_dataset.py

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

DAILY_DIR = BASE_DIR / "data" / "clean" / "daily"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = PROCESSED_DIR / "aqi_lagged_SEA.csv"


def load_all_daily() -> pd.DataFrame:
    files = sorted(DAILY_DIR.glob("waqi_daily_SEA_*.csv"))
    if not files:
        raise FileNotFoundError(f"ไม่พบ daily files ใน {DAILY_DIR}")

    dfs = []
    for f in files:
        print(f"[TRAIN-DATA] โหลด {f}")
        df = pd.read_csv(f)
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    if "station_time" not in df_all.columns:
        raise ValueError("ไม่พบคอลัมน์ station_time ใน daily data")

    df_all["station_time"] = pd.to_datetime(df_all["station_time"], errors="coerce")
    df_all = df_all.dropna(subset=["station_time"])

    # sort ให้กลายเป็น time series ต่อเนื่อง
    sort_cols = [c for c in ["station_idx", "station_time"] if c in df_all.columns]
    df_all = df_all.sort_values(sort_cols).reset_index(drop=True)

    print(f"รวม daily data ได้ทั้งหมด {len(df_all)} แถว")
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

    # ลบแถวที่ target หรือ lag สำคัญเป็น NaN
    must_have = [c for c in ["aqi_next1h", "aqi_lag1"] if c in df.columns]
    df = df.dropna(subset=must_have)

    print(f"หลังทำ lag + target เหลือ {len(df)} แถว สำหรับเทรน")
    return df


def main():
    df_all = load_all_daily()
    df_lagged = add_lag_features(df_all)

    df_lagged.to_csv(OUTPUT_CSV, index=False)
    print(f"🎉 บันทึก training dataset ที่: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

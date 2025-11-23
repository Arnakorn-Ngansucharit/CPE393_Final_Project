# prepare_dataset.py

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_RAW_TS_DIR = BASE_DIR / "data" / "raw" / "waqi_timeseries"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = DATA_PROCESSED_DIR / "aqi_lagged_SEA.csv"


def load_all_timeseries():
    files = sorted(DATA_RAW_TS_DIR.glob("waqi_timeseries_SEA_*.csv"))
    if not files:
        raise FileNotFoundError("ไม่พบไฟล์ waqi_timeseries_SEA_*.csv ใน data/raw/waqi_timeseries")

    dfs = []
    for f in files:
        print(f"โหลด {f}")
        df = pd.read_csv(f)
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    # แปลงเวลา
    df_all["station_time"] = pd.to_datetime(df_all["station_time"], errors="coerce")

    # ลบแถวที่เวลาเป็น NaT
    df_all = df_all.dropna(subset=["station_time"])

    # sort ตาม station + เวลา
    df_all = df_all.sort_values(["station_idx", "station_time"]).reset_index(drop=True)
    return df_all


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    เพิ่ม lag features + target aqi_next1h สำหรับแต่ละ station_idx
    """
    df = df.copy()

    # เลือกเฉพาะคอลัมน์ที่สนใจ
    cols_keep = [
        "station_idx", "station_name", "lat", "lon",
        "station_time", "aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
        "t", "h", "p", "w"
    ]
    df = df[cols_keep]

    # สร้าง lag features ภายในแต่ละ station
    def create_lags(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("station_time")
        # target: AQI ในชั่วโมงถัดไป
        group["aqi_next1h"] = group["aqi"].shift(-1)

        # lag ของ aqi และ pm25
        group["aqi_lag1"] = group["aqi"].shift(1)
        group["aqi_lag3"] = group["aqi"].shift(3)

        group["pm25_lag1"] = group["pm25"].shift(1)
        group["pm10_lag1"] = group["pm10"].shift(1)

        # lag ของ weather
        group["t_lag1"] = group["t"].shift(1)
        group["h_lag1"] = group["h"].shift(1)

        return group

    df = df.groupby("station_idx", group_keys=False).apply(create_lags)

    # ลบแถวที่ target หรือ lag เป็น NaN (เช่น แถวแรกๆ ของแต่ละ station)
    df = df.dropna(subset=["aqi_next1h", "aqi_lag1", "pm25_lag1"])

    return df


def main():
    df_all = load_all_timeseries()
    print(f"รวมข้อมูลได้ {len(df_all)} แถว")

    df_lagged = add_lag_features(df_all)
    print(f"หลังทำ lag และ target เหลือ {len(df_lagged)} แถว")

    df_lagged.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"บันทึก dataset สำหรับ train model ที่: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

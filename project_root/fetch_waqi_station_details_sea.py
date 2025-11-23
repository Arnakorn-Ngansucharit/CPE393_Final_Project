import os
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
from dotenv import load_dotenv

# ========== CONFIG ==========
load_dotenv()
WAQI_TOKEN = os.getenv("WAQI_TOKEN")
if WAQI_TOKEN is None:
    raise ValueError("ไม่พบ WAQI_TOKEN ใน .env กรุณาเพิ่ม WAQI_TOKEN=<token>")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "raw"
GLOBAL_DIR = DATA_DIR / "waqi_global"
TS_DIR = DATA_DIR / "waqi_timeseries"
TS_DIR.mkdir(parents=True, exist_ok=True)

# SEA bounding box (คร่าว ๆ): -10 ถึง 25N, 90 ถึง 135E
LAT_MIN_SEA, LAT_MAX_SEA = -10, 25
LON_MIN_SEA, LON_MAX_SEA = 90, 135

# จำกัดจำนวน concurrent workers กันโดน rate limit
MAX_WORKERS = 8

# ========== HELPERS ==========

def get_latest_global_snapshot() -> Path:
    files = sorted(GLOBAL_DIR.glob("waqi_global_*.csv"))
    if not files:
        raise FileNotFoundError("ไม่พบไฟล์ waqi_global_*.csv ใน data/raw/waqi_global")
    return files[-1]


def fetch_station_detail_from_row(row: pd.Series) -> dict | None:
    """ดึงรายละเอียดของสถานี 1 แถว (ใช้ lat, lon จาก global snapshot)"""
    lat = row["lat"]
    lon = row["lon"]

    if pd.isna(lat) or pd.isna(lon):
        return None

    url = f"https://api.waqi.info/feed/geo:{lat};{lon}/"
    params = {"token": WAQI_TOKEN}

    try:
        resp = requests.get(url, params=params, timeout=10)
    except Exception as e:
        print(f"[ERROR] request exception lat={lat}, lon={lon}: {e}")
        return None

    if resp.status_code != 200:
        print(f"[ERROR] HTTP {resp.status_code} lat={lat}, lon={lon}")
        return None

    data = resp.json()
    if data.get("status") != "ok":
        print(f"[ERROR] status={data.get('status')} lat={lat}, lon={lon}")
        return None

    d = data["data"]
    iaqi = d.get("iaqi", {})

    def get_iaqi(key: str):
        v = iaqi.get(key)
        return v.get("v") if isinstance(v, dict) else None

    time_info = d.get("time", {})
    city_info = d.get("city", {})
    city_geo = city_info.get("geo", [None, None])

    result = {
        # map ระหว่าง global snapshot กับ station detail
        "global_uid": row.get("uid"),
        "global_aqi_snapshot": row.get("aqi"),
        "global_snapshot_utc": row.get("snapshot_utc"),

        # station-level meta
        "station_idx": d.get("idx"),
        "station_name": city_info.get("name"),
        "station_url": city_info.get("url"),
        "lat": city_geo[0],
        "lon": city_geo[1],

        # main AQI info
        "aqi": d.get("aqi"),
        "dominentpol": d.get("dominentpol"),

        # pollutants
        "pm25": get_iaqi("pm25"),
        "pm10": get_iaqi("pm10"),
        "o3": get_iaqi("o3"),
        "no2": get_iaqi("no2"),
        "so2": get_iaqi("so2"),
        "co": get_iaqi("co"),

        # weather
        "t": get_iaqi("t"),   # temperature
        "h": get_iaqi("h"),   # humidity
        "p": get_iaqi("p"),   # pressure
        "w": get_iaqi("w"),   # wind speed
        "dew": get_iaqi("dew"),
        "rain": get_iaqi("r"),
        "uv": get_iaqi("uv"),

        # time at station
        "station_time": time_info.get("s"),
        "station_tz": time_info.get("tz"),
        "station_unix": time_info.get("v"),

        # ingestion info (เวลาที่เราดึงจริง)
        "ingested_at_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    }

    return result


def main():
    snapshot_path = get_latest_global_snapshot()
    print(f"ใช้ global snapshot: {snapshot_path}")

    df_global = pd.read_csv(snapshot_path)

    # filter เฉพาะ SEA
    df_sea = df_global[
        (df_global["lat"] >= LAT_MIN_SEA) &
        (df_global["lat"] <= LAT_MAX_SEA) &
        (df_global["lon"] >= LON_MIN_SEA) &
        (df_global["lon"] <= LON_MAX_SEA)
    ].reset_index(drop=True)

    print(f"สถานีใน SEA ทั้งหมด: {len(df_sea)} แถว")

    if df_sea.empty:
        print("ไม่มีสถานีใน SEA เลย ตรวจสอบ bounding box อีกครั้ง")
        return

    results = []
    start_time = time.time()

    # ใช้ ThreadPoolExecutor ดึงข้อมูลแบบ concurrent
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(fetch_station_detail_from_row, row): i
            for i, row in df_sea.iterrows()
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                data = future.result()
                if data is not None:
                    results.append(data)
            except Exception as e:
                print(f"[ERROR] future idx={idx}: {e}")

            if (idx + 1) % 50 == 0:
                elapsed = time.time() - start_time
                print(f"ดึงเสร็จแล้ว {idx+1}/{len(df_sea)} ใช้เวลา {elapsed:.1f} วินาที")

    if not results:
        print("ไม่มีข้อมูล station detail ที่ดึงสำเร็จเลย")
        return

    df_ts = pd.DataFrame(results)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = TS_DIR / f"waqi_timeseries_SEA_{ts}.csv"
    df_ts.to_csv(out_path, index=False, encoding="utf-8-sig")

    total_elapsed = time.time() - start_time
    print(f"บันทึก station details SEA ที่: {out_path}")
    print(f"จำนวนแถว: {len(df_ts)} ใช้เวลาทั้งหมด {total_elapsed:.1f} วินาที")


if __name__ == "__main__":
    main()

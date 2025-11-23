import os
import time
from datetime import datetime
from pathlib import Path

import requests
import pandas as pd
from dotenv import load_dotenv

# =============== CONFIG ===============

load_dotenv()
WAQI_TOKEN = os.getenv("WAQI_TOKEN")

if WAQI_TOKEN is None:
    raise ValueError("ไม่พบ WAQI_TOKEN ใน .env กรุณาเพิ่ม WAQI_TOKEN=<token>")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# โฟลเดอร์เก็บ snapshot รายรอบ
SNAPSHOT_DIR = DATA_DIR / "waqi_global"
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

# กำหนดกริดครอบโลก (เปลี่ยนได้)
LAT_MIN, LAT_MAX = -80, 80
LON_MIN, LON_MAX = -180, 180
LAT_STEP = 20
LON_STEP = 20

# หน่วงเวลาแต่ละ request (ป้องกันโดน rate limit)
SLEEP_SEC = 1.0

MAP_BOUNDS_URL = "https://api.waqi.info/map/bounds/"


# =============== HELPER ===============

def generate_global_tiles():
    """สร้างกริด (lat1, lon1, lat2, lon2) ครอบโลกตาม step ที่ตั้งไว้"""
    tiles = []
    lat = LAT_MIN
    while lat < LAT_MAX:
        lon = LON_MIN
        lat2 = min(lat + LAT_STEP, LAT_MAX)
        while lon < LON_MAX:
            lon2 = min(lon + LON_STEP, LON_MAX)
            tiles.append((lat, lon, lat2, lon2))
            lon += LON_STEP
        lat += LAT_STEP
    return tiles


def fetch_tile(lat1, lon1, lat2, lon2):
    """ดึงข้อมูลสถานีในกรอบ lat/lon หนึ่งช่องด้วย map/bounds API"""
    params = {
        "token": WAQI_TOKEN,
        "latlng": f"{lat1},{lon1},{lat2},{lon2}",
    }
    resp = requests.get(MAP_BOUNDS_URL, params=params, timeout=15)
    if resp.status_code != 200:
        print(f"[ERROR] HTTP {resp.status_code} for tile {lat1},{lon1},{lat2},{lon2}")
        return []

    data = resp.json()
    if data.get("status") != "ok":
        print(f"[ERROR] status={data.get('status')} for tile {lat1},{lon1},{lat2},{lon2}")
        return []

    rows = []
    for st in data.get("data", []):
        # ตัวอย่าง st: {"lat": 51.4, "lon": 7.2, "uid": 6093, "aqi": "20"}
        rows.append(
            {
                "uid": st.get("uid"),
                "lat": st.get("lat"),
                "lon": st.get("lon"),
                "aqi": st.get("aqi"),
                "tile_lat1": lat1,
                "tile_lon1": lon1,
                "tile_lat2": lat2,
                "tile_lon2": lon2,
            }
        )
    return rows


def fetch_global_snapshot():
    """ดึง snapshot ทั้งโลก 1 รอบ คืนค่าเป็น DataFrame"""
    tiles = generate_global_tiles()
    all_rows = []

    print(f"จำนวน tiles ทั้งหมด: {len(tiles)}")
    for idx, (lat1, lon1, lat2, lon2) in enumerate(tiles, start=1):
        print(
            f"[{idx}/{len(tiles)}] ดึงข้อมูลกรอบ latlng=({lat1},{lon1},{lat2},{lon2}) ..."
        )   
        rows = fetch_tile(lat1, lon1, lat2, lon2)
        all_rows.extend(rows)

        # หน่วงนิดนึงกันโดน block
        time.sleep(SLEEP_SEC)

    if not all_rows:
        print("ไม่มีข้อมูลสถานีเลย (all_rows ว่างเปล่า)")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # aqi จาก API เป็น string บางครั้งเป็น "-" แปลงเป็น numeric
    df["aqi"] = pd.to_numeric(df["aqi"], errors="coerce")

    # deduplicate เผื่อสถานีหลุดซ้ำในขอบเขตติดกัน
    df = df.drop_duplicates(subset=["uid"]).reset_index(drop=True)

    # เพิ่ม column เวลา snapshot ตอนดึง
    snapshot_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    df["snapshot_utc"] = snapshot_time

    return df


def save_snapshot(df: pd.DataFrame):
    """เซฟ DataFrame เป็นไฟล์ CSV แยกตามเวลา"""
    if df.empty:
        print("DataFrame ว่าง ไม่เซฟไฟล์")
        return

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = SNAPSHOT_DIR / f"waqi_global_{ts}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"บันทึก snapshot แล้วที่: {out_path}")
    print(f"จำนวนสถานี: {len(df)}")


# =============== MAIN ===============

def main():
    df = fetch_global_snapshot()
    save_snapshot(df)


if __name__ == "__main__":
    main()

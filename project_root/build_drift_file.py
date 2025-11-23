# build_drift_file.py
# สร้างไฟล์สำหรับ data drift detection จาก trained data ล่าสุด

from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent

TRAINED_DATA_DIR = BASE_DIR / "data" / "trained_data"
DRIFT_DIR = BASE_DIR / "data" / "drift"
DRIFT_DIR.mkdir(parents=True, exist_ok=True)


def load_latest_trained_data() -> pd.DataFrame:
    """โหลด trained data ล่าสุด"""
    if not TRAINED_DATA_DIR.exists():
        raise FileNotFoundError(f"ไม่พบโฟลเดอร์ {TRAINED_DATA_DIR}")
        
    files = sorted(TRAINED_DATA_DIR.glob("trained_data_*.csv"), reverse=True)
    if not files:
        raise FileNotFoundError(f"ไม่พบไฟล์ trained_data ใน {TRAINED_DATA_DIR}")

    latest_file = files[0]
    print(f"[DRIFT] โหลดไฟล์ trained data ล่าสุด: {latest_file.name}")
    df = pd.read_csv(latest_file)
    return df


def main():
    """สร้างไฟล์สำหรับ data drift detection"""
    print("=" * 60)
    print("สร้างไฟล์สำหรับ Data Drift Detection จาก Trained Data")
    print("=" * 60)

    # โหลด trained data ล่าสุด
    try:
        df_drift = load_latest_trained_data()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # บันทึกไฟล์
    output_path = DRIFT_DIR / "waqi_drift_reference.csv"
    df_drift.to_csv(output_path, index=False)
    
    print(f"\n🎉 บันทึกไฟล์ drift reference ที่: {output_path}")
    print(f"   Shape: {df_drift.shape}")
    print(f"   Columns: {list(df_drift.columns)}")
    print(f"\n💡 ใช้ไฟล์นี้เป็น reference dataset สำหรับ Evidently AI")


if __name__ == "__main__":
    main()


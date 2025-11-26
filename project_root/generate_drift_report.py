# generate_drift_report.py
# สร้าง Evidently AI report สำหรับ data drift detection

from pathlib import Path
import pandas as pd
from datetime import datetime

try:
    from evidently import Report
    from evidently.presets import DataDriftPreset
except ImportError:
    print("⚠️  Evidently AI ยังไม่ได้ติดตั้ง")
    print("   กรุณาติดตั้งด้วย: pip install evidently")
    raise

BASE_DIR = Path(__file__).resolve().parent

DRIFT_DIR = BASE_DIR / "data" / "drift"
DRIFT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_DIR = BASE_DIR / "data" / "drift" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TRAINED_DATA_DIR = BASE_DIR / "data" / "trained_data"
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def load_reference_data() -> pd.DataFrame:
    """
    โหลด reference dataset จาก trained_data
    ค้นหาไฟล์ preprocessed_dataset_*.csv หรือ trained_data_*.csv ใน data/trained_data/
    ถ้าไม่มี ให้ใช้ training dataset จาก data/processed/aqi_lagged_SEA_*.csv
    """
    # ลองหาไฟล์ใน trained_data/ ก่อน
    reference_file = None
    
    if TRAINED_DATA_DIR.exists():
        # หาไฟล์ preprocessed_dataset_*.csv
        files_preprocessed = sorted(TRAINED_DATA_DIR.glob("preprocessed_dataset_*.csv"), reverse=True)
        # หาไฟล์ trained_data_*.csv
        files_trained = sorted(TRAINED_DATA_DIR.glob("trained_data_*.csv"), reverse=True)
        
        all_trained_files = files_preprocessed + files_trained
        if all_trained_files:
            reference_file = all_trained_files[0]
            print(f"[DRIFT] พบไฟล์ trained data: {reference_file.name}")
    
    # Fallback: ใช้ training dataset จาก processed/
    if reference_file is None:
        if PROCESSED_DIR.exists():
            files_processed = sorted(PROCESSED_DIR.glob("aqi_lagged_SEA_*.csv"), reverse=True)
            if files_processed:
                reference_file = files_processed[0]
                print(f"[DRIFT] ใช้ training dataset จาก processed/: {reference_file.name}")
    
    if reference_file is None:
        raise FileNotFoundError(
            f"ไม่พบไฟล์ trained data สำหรับใช้เป็น reference\n"
            f"   ค้นหาใน: {TRAINED_DATA_DIR} (preprocessed_dataset_*.csv หรือ trained_data_*.csv)\n"
            f"   หรือ: {PROCESSED_DIR} (aqi_lagged_SEA_*.csv)\n"
            f"กรุณารัน train.py หรือ build_training_dataset.py ก่อน"
        )
    
    print(f"[DRIFT] โหลดไฟล์ reference (trained data): {reference_file.relative_to(BASE_DIR)}")
    df = pd.read_csv(reference_file)
    
    # แปลง date column ถ้ามี
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "station_time" in df.columns:
        df["station_time"] = pd.to_datetime(df["station_time"], errors="coerce")
    
    print(f"โหลด reference data (trained data): {df.shape}")
    return df


def load_current_data() -> pd.DataFrame:
    """
    โหลด current/production data จากไฟล์ hourly ล่าสุด
    ค้นหาไฟล์ waqi_hourly_SEA_* และ waqi_cleaned_* ในทุก subfolder
    """
    hourly_dir = BASE_DIR / "data" / "clean" / "hourly"
    if not hourly_dir.exists():
        raise FileNotFoundError(f"ไม่พบโฟลเดอร์ {hourly_dir}")

    # ค้นหาไฟล์ทั้ง waqi_hourly_SEA_* และ waqi_cleaned_* ในทุก subfolder
    files_hourly = list(hourly_dir.rglob("waqi_hourly_SEA_*.csv"))
    files_cleaned = list(hourly_dir.rglob("waqi_cleaned_*.csv"))
    all_files = files_hourly + files_cleaned
    
    if not all_files:
        raise FileNotFoundError(f"ไม่พบไฟล์ hourly data ใน {hourly_dir}")

    # เรียงตามเวลาที่แก้ไขล่าสุด (mtime)
    all_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    latest_file = all_files[0]
    
    print(f"[DRIFT] โหลดไฟล์ hourly ล่าสุด: {latest_file.name}")
    print(f"       Path: {latest_file.relative_to(BASE_DIR)}")
    df_current = pd.read_csv(latest_file)

    # แปลง date column ถ้ามี
    if "date" in df_current.columns:
        df_current["date"] = pd.to_datetime(df_current["date"], errors="coerce")

    print(f"โหลด current data: {df_current.shape}")
    return df_current


def prepare_data_for_evidently(df_ref: pd.DataFrame, df_current: pd.DataFrame) -> tuple:
    """
    เตรียมข้อมูลสำหรับ Evidently
    Evidently ต้องการให้ columns ตรงกันและไม่มี date column (ถ้าไม่ใช้เป็น datetime feature)
    """
    # เลือกเฉพาะ numeric columns (ไม่รวม date)
    numeric_cols = df_ref.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # ลบ date ออกถ้ามี (เพราะเป็น metadata)
    if "date" in numeric_cols:
        numeric_cols.remove("date")
    
    # ตรวจสอบว่า columns ตรงกัน
    common_cols = [c for c in numeric_cols if c in df_current.columns]
    
    if len(common_cols) != len(numeric_cols):
        missing = set(numeric_cols) - set(common_cols)
        print(f"⚠️  Warning: columns ไม่ตรงกัน - ขาด: {missing}")
    
    df_ref_clean = df_ref[common_cols].copy()
    df_current_clean = df_current[common_cols].copy()
    
    # ลบแถวที่มี NaN (Evidently ต้องการข้อมูลที่ clean)
    df_ref_clean = df_ref_clean.dropna()
    df_current_clean = df_current_clean.dropna()
    
    print(f"Reference data (clean): {df_ref_clean.shape}")
    print(f"Current data (clean): {df_current_clean.shape}")
    print(f"Features: {list(common_cols)}")
    
    return df_ref_clean, df_current_clean


def generate_drift_report(df_ref: pd.DataFrame, df_current: pd.DataFrame):
    """สร้าง Evidently data drift report (Evidently 0.7.x API)"""
    print("\n" + "=" * 60)
    print("สร้าง Evidently Data Drift Report")
    print("=" * 60)
    
    # เตรียมข้อมูล
    df_ref_clean, df_current_clean = prepare_data_for_evidently(df_ref, df_current)
    
    if len(df_ref_clean) == 0 or len(df_current_clean) == 0:
        raise ValueError("ข้อมูลไม่เพียงพอสำหรับสร้าง report")
    
    # กำหนด schema (numeric features)
    numeric_features = [c for c in df_ref_clean.columns if df_ref_clean[c].dtype in ['float64', 'int64']]
    
    print(f"\n📊 กำลังสร้าง report...")
    print(f"   Numeric features: {len(numeric_features)}")
    
    # สร้าง report
    report = Report(metrics=[DataDriftPreset()])
    
    # รัน report และรับ snapshot (Evidently 0.7.x: column_mapping ส่งใน run(), และ run() คืนค่า snapshot)
    snapshot = report.run(
        reference_data=df_ref_clean,
        current_data=df_current_clean,
    )
    
    # บันทึก report (Evidently 0.7.x: save_html อยู่ใน snapshot)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORT_DIR / f"drift_report_{timestamp}.html"
    
    snapshot.save_html(str(report_path))
    
    print(f"\n✅ บันทึก report ที่: {report_path}")
    
    # แสดง summary
    print("\n" + "=" * 60)
    print("Report Summary")
    print("=" * 60)
    
    # ดึง metrics จาก snapshot dict
    try:
        metrics_dict = snapshot.dict()
        metrics = metrics_dict.get('metrics', [])

        drift_metric = next(
            (m for m in metrics if str(m.get('metric_name', '')).startswith('DriftedColumnsCount')),
            None,
        )

        if drift_metric:
            value = drift_metric.get('value', {})
            config = drift_metric.get('config', {})
            drift_share = float(value.get('share', 0))
            num_drifted = int(value.get('count', 0))
            total_features = len(numeric_features)
            threshold = float(config.get('drift_share', 0.5))
            drift_detected = drift_share >= threshold

            print(f"Dataset Drift Detected: {drift_detected}")
            print(f"Drift Share: {drift_share:.2f} (threshold {threshold:.2f})")
            print(f"Drifted Features: {num_drifted} / {total_features}")
        else:
            print("⚠️  ไม่พบ metric DriftedColumnsCount ใน snapshot.dict()")
    except Exception as e:
        print(f"⚠️  ไม่สามารถดึง summary ได้: {e}")
        print("   ดูรายละเอียดใน HTML report แทน")
    
    print(f"\n💡 เปิดไฟล์ {report_path} ใน browser เพื่อดูรายละเอียด")
    
    return report_path


def main():
    """Main function"""
    try:
        # โหลดข้อมูล
        print("=" * 60)
        print("Evidently AI - Data Drift Report Generator")
        print("=" * 60)
        
        df_ref = load_reference_data()
        df_current = load_current_data()
        
        # สร้าง report
        report_path = generate_drift_report(df_ref, df_current)
        
        print("\n🎉 สร้าง drift report เสร็จแล้ว!")
        print(f"   Report: {report_path}")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 วิธีแก้:")
        print("   1. รัน train.py เพื่อสร้าง trained data")
        print("   2. หรือรัน build_training_dataset.py เพื่อสร้าง training dataset")
        print("   3. หรือตรวจสอบ path ของไฟล์ข้อมูล")
    except ImportError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 วิธีแก้:")
        print("   pip install evidently")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


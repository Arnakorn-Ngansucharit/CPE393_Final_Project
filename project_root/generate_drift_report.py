# generate_drift_report.py
# สร้าง Evidently AI report สำหรับ data drift detection

from pathlib import Path
import pandas as pd
from datetime import datetime

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    from evidently import ColumnMapping
except ImportError:
    print("⚠️  Evidently AI ยังไม่ได้ติดตั้ง")
    print("   กรุณาติดตั้งด้วย: pip install evidently")
    raise

BASE_DIR = Path(__file__).resolve().parent

DRIFT_DIR = BASE_DIR / "data" / "drift"
DRIFT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_DIR = BASE_DIR / "data" / "drift" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_PATH = DRIFT_DIR / "waqi_drift_reference.csv"


def load_reference_data() -> pd.DataFrame:
    """โหลด reference dataset"""
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(
            f"ไม่พบไฟล์ reference ที่ {REFERENCE_PATH}\n"
            f"กรุณารัน build_drift_file.py ก่อน"
        )
    
    df = pd.read_csv(REFERENCE_PATH)
    
    # แปลง date column ถ้ามี
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    
    print(f"โหลด reference data: {df.shape}")
    return df


def load_current_data() -> pd.DataFrame:
    """
    โหลด current/production data จากไฟล์ hourly ล่าสุด
    """
    hourly_dir = BASE_DIR / "data" / "clean" / "hourly"
    if not hourly_dir.exists():
        raise FileNotFoundError(f"ไม่พบโฟลเดอร์ {hourly_dir}")

    files = sorted(hourly_dir.glob("waqi_hourly_SEA_*.csv"), reverse=True)
    if not files:
        raise FileNotFoundError(f"ไม่พบไฟล์ hourly data ใน {hourly_dir}")

    latest_file = files[0]
    print(f"[DRIFT] โหลดไฟล์ hourly ล่าสุด: {latest_file.name}")
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
    """สร้าง Evidently data drift report"""
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
    
    # สร้าง ColumnMapping
    column_mapping = ColumnMapping()
    column_mapping.numerical_features = numeric_features

    # สร้าง report
    report = Report(metrics=[DataDriftPreset()])
    
    # รัน report
    report.run(reference_data=df_ref_clean, current_data=df_current_clean, column_mapping=column_mapping)
    
    # บันทึก report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORT_DIR / f"drift_report_{timestamp}.html"
    
    report.save_html(str(report_path))
    
    print(f"\n✅ บันทึก report ที่: {report_path}")
    
    # แสดง summary
    print("\n" + "=" * 60)
    print("Report Summary")
    print("=" * 60)
    
    # ดึง metrics จาก json
    try:
        metrics_dict = report.as_dict()
        
        # หา data drift metrics
        if 'metrics' in metrics_dict:
            for metric_result in metrics_dict['metrics']:
                if metric_result['metric'] == 'DatasetDriftMetric':
                    result = metric_result.get('result', {})
                    drift_detected = result.get('dataset_drift', False)
                    drift_share = result.get('drift_share', 0)
                    print(f"Dataset Drift Detected: {drift_detected}")
                    print(f"Drift Share: {drift_share:.2f}")
                    
                    num_drifted = result.get('number_of_drifted_columns', 0)
                    total_features = result.get('number_of_columns', 0)
                    print(f"Drifted Features: {num_drifted} / {total_features}")
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
        print("   1. รัน build_drift_file.py เพื่อสร้าง reference dataset")
        print("   2. หรือตรวจสอบ path ของไฟล์ข้อมูล")
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


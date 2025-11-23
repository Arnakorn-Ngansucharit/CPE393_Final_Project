# CPE393_Final_Project
# CPE393 – AQI Forecasting MLOps Pipeline

โครงการนี้เป็นส่วนหนึ่งของรายวิชา **CPE393: Machine Learning Operations**  
เป้าหมายคือสร้าง MLOps pipeline สำหรับ **ทำนายค่า Air Quality Index (AQI) ล่วงหน้า** จากข้อมูลมลพิษอากาศในเขต **SEA (Southeast Asia)** โดยใช้ข้อมูลจาก **WAQI / AQICN API**

Pipeline ครอบคลุมตั้งแต่

- Data ingestion จาก WAQI API
- Data preprocessing + feature engineering (lag features)
- Model training (อย่างน้อย 3 โมเดล)
- Experiment tracking & model registry ด้วย MLflow
- EDA (Exploratory Data Analysis) สำหรับใส่ในรายงาน/สไลด์

> **หมายเหตุ:** Repo นี้โฟกัสฝั่ง Data Pipeline + Model Training  
> ส่วน Deployment, Monitoring และ Automation จะทำแยกในบทบาท MLOps Engineer

---

## 1. Project Structure

โครงสร้างหลักของโปรเจกต์ (ในโฟลเดอร์ `project_root/`):

```text
project_root/
├─ data/
│  ├─ raw/
│  │   ├─ waqi_global/            # global snapshot จาก WAQI map/bounds API
│  │   └─ waqi_timeseries/        # SEA station-level time series จาก feed/geo API
│  └─ processed/
│      └─ aqi_lagged_SEA.csv      # dataset หลังทำ feature engineering (ใช้ train model)
├─ .env                           # ไฟล์เก็บ WAQI API token (อย่าใส่ลง Git)
├─ fetch_waqi_global_snapshot.py  # ดึง global AQI snapshot ทั้งโลก
├─ fetch_waqi_station_details_sea.py  # ดึง station-level features เฉพาะ SEA
├─ prepare_dataset.py             # รวมหลายไฟล์ time series + ทำ lag features
├─ train.py                       # เทรน 3 โมเดล + log ด้วย MLflow + register best model
├─ eda.py                         # สร้าง EDA plots (ใช้ในรายงาน/สไลด์)
└─ README.md                      # (ไฟล์นี้)
```

## 2. การขอ WAQI API Token (จำเป็นต้องทำเอง)
```
- เนื่องจาก token เป็นของส่วนตัว ทุกคนในกลุ่มต้องไปขอ token เอง จาก WAQI / AQICN

- วิธีขอ token

- ไปที่เว็บไซต์ AQICN / WAQI (Air Quality Open Data Platform) https://aqicn.org/data-platform/api/H5773/ คลิกที่ data-platform token registration page

- สมัครหรือล็อกอิน (ใช้ email ของตัวเอง)

- ในหน้า user dashboard จะมี API Token แสดงอยู่ (string ยาว ๆ)

- คัดลอก token นั้นมาเก็บไว้

- แนะนำให้คนเดียวในทีมดูแล token แล้วแชร์ให้เฉพาะในกลุ่ม (ห้าม commit ลง GitHub แบบ public)
```

## 3. การวาง Token ไว้ตรงไหนของโฟลเดอร์
```
โปรเจกต์นี้ใช้ไฟล์ .env ที่ project_root/.env เพื่อเก็บค่า token

ที่ตำแหน่ง:
project_root/
├─ .env      ← สร้างไฟล์นี้เอง
├─ fetch_waqi_global_snapshot.py
├─ fetch_waqi_station_details_sea.py
...

ให้สร้างไฟล์ .env แล้วใส่

WAQI_TOKEN=YOUR_TOKEN_HERE

เปลี่ยน YOUR_TOKEN_HERE เป็น token จริงของคุณ

ห้ามมี quote (" " หรือ ' ') รอบ token

ควรเพิ่ม .env ลงใน .gitignore ถ้า push ขึ้น GitHub

สคริปต์ทุกไฟล์ที่เรียก WAQI API จะโหลดค่า WAQI_TOKEN จากไฟล์ .env นี้ผ่าน python-dotenv
```

## 4. การติดตั้ง Dependencies
```
แนะนำให้ใช้ Python 3.11
```

## 5. อธิบายแต่ละไฟล์ & วิธีรัน
5.1 fetch_waqi_global_snapshot.py – ดึง Global AQI Snapshot

หน้าที่:
ดึงข้อมูล AQI ทั่วโลกผ่าน map/bounds API ของ WAQI โดยแบ่งโลกเป็นกริด 20°x20°
แล้วรวมสถานีทั้งหมด (มี uid, lat, lon, aqi, snapshot_utc, tile_*)

Output:
```
data/raw/waqi_global/waqi_global_YYYYMMDD_HHMMSS.csv
```

วิธีรัน:
```
python fetch_waqi_global_snapshot.py
```

5.2 fetch_waqi_station_details_sea.py – ดึง Station-Level Features เฉพาะ SEA

หน้าที่:
```
อ่านไฟล์ล่าสุดจาก data/raw/waqi_global/

filter เฉพาะสถานีใน SEA (lat/lon ประมาณ -10 ถึง 25, 90 ถึง 135)

ใช้ ThreadPoolExecutor ดึงข้อมูลละเอียด station-level ด้วย feed/geo:lat;lon API
```
ดึงค่า:
```
AQI, dominent pollutant

PM2.5, PM10, O3, NO2, SO2, CO

temperature (t), humidity (h), pressure (p), wind (w), ฯลฯ

station time, timezone, geo, station name

บันทึกเป็น time series snapshot สำหรับ SEA
```
Output:
```
data/raw/waqi_timeseries/waqi_timeseries_SEA_YYYYMMDD_HHMMSS.csv
```

วิธีรัน:
```
python fetch_waqi_station_details_sea.py
```

แนะนำให้รันซ้ำหลายครั้ง (เช่น ทุก 30 นาที หรือ 1 ชั่วโมง) เพื่อให้ได้ time series หลายเวลา

5.3 prepare_dataset.py – รวม Time Series + สร้าง Lag Features

หน้าที่:
```
อ่านไฟล์ทั้งหมดจาก data/raw/waqi_timeseries/

แปลง station_time เป็น datetime

sort ตาม station_idx + station_time

สำหรับแต่ละ station:

สร้าง target: aqi_next1h = aqi.shift(-1)

สร้าง lag features:

aqi_lag1, aqi_lag3

pm25_lag1, pm10_lag1

t_lag1, h_lag1

ลบแถวที่มี NaN ใน target หรือ lag ที่สำคัญ
```
Output:
```
data/processed/aqi_lagged_SEA.csv
```

วิธีรัน:
```
python prepare_dataset.py
```

หลังรันเสร็จจะได้ training dataset ที่พร้อมใช้ใน train.py

5.4 train.py – Train 3 Models + MLflow Tracking + Register Best Model

หน้าที่:
```
โหลด data/processed/aqi_lagged_SEA.csv

ลบแถวที่มี NaN

แยก X และ target aqi_next1h

train/test split (80/20)

เทรนโมเดลอย่างน้อย 3 ตัว:

LinearRegression

RandomForestRegressor

GradientBoostingRegressor

คำนวณ metrics:

MAE

RMSE

R²

log parameters, metrics และ model เข้า MLflow ใน experiment aqi_forecasting

เลือก best model จาก RMSE ต่ำสุด

Register best model เข้า MLflow Model Registry ชื่อ aqi_best_model
```
Output:
```
MLflow runs ใน mlruns/ (local tracking)
```
Registered model: aqi_best_model (version 1, 2, … ตามจำนวนรอบการ train)

วิธีรัน:
```
python train.py
```

ดูผลผ่าน MLflow UI:
```
mlflow ui
```

แล้วเปิด browser ไปที่: http://127.0.0.1:5000

5.5 eda.py – Exploratory Data Analysis

หน้าที่:
```
โหลด data/processed/aqi_lagged_SEA.csv

วาดกราฟ EDA หลัก ๆ:

Missing values per column

Distribution ของ aqi, pm25, pm10, t, h

Correlation heatmap (numeric features)

Scatter: AQI vs PM2.5

Station bias (ค่า AQI เฉลี่ย top-20 สถานี)

Time trend (ค่า AQI เฉลี่ยรายวัน ตาม station_time ถ้า parse ได้)
```
Output:

รูปกราฟทั้งหมดจะถูกเก็บในโฟลเดอร์:
```
eda_output/
├─ missing_values.png
├─ dist_aqi.png
├─ dist_pm25.png
├─ dist_pm10.png
├─ dist_t.png
├─ dist_h.png
├─ corr_heatmap.png
├─ scatter_aqi_pm25.png
├─ station_bias.png
└─ aqi_time_trend.png   # ถ้ามีข้อมูลเวลาเพียงพอ
```

วิธีรัน:
```
python eda.py
```

รูปที่ได้สามารถนำไปใช้ใน รายงาน และ presentation ได้โดยตรง

## 6. Pipeline Summary (สำหรับใส่ในสไลด์)

ลำดับการรัน pipeline:

ดึง global snapshot:
```
python fetch_waqi_global_snapshot.py
```

ดึง station-level time series เฉพาะ SEA:
```
python fetch_waqi_station_details_sea.py
```

(ควรทำหลาย ๆ รอบต่างเวลา เพื่อให้ได้ time series)

เตรียม training dataset (lag features):
```
python prepare_dataset.py
```

เทรน 3 โมเดล + MLflow + Register best model:
```
python train.py
```

ทำ EDA เพื่อรายงาน/สไลด์:
```
python eda.py
```
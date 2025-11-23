import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------- CONFIG ----------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed" / "aqi_lagged_SEA.csv"

OUTPUT_DIR = BASE_DIR / "eda_output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------- LOAD DATA ----------------
df = pd.read_csv(DATA_PATH)
print(f"Loaded dataset: {df.shape}")

# ---------------- 1) Missing Values ----------------
missing = df.isna().sum()

plt.figure(figsize=(10,5))
sns.barplot(x=missing.index, y=missing.values)
plt.xticks(rotation=45)
plt.title("Missing Values Per Column")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "missing_values.png")
plt.close()

print("✔ saved: missing_values.png")


# ---------------- 2) Distribution (Histogram) ----------------
numeric_cols = df.select_dtypes(include=[np.number]).columns

for col in ["aqi", "pm25", "pm10", "t", "h"]:
    if col in numeric_cols:
        plt.figure(figsize=(6,4))
        sns.histplot(df[col], bins=40, kde=True)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"dist_{col}.png")
        plt.close()
        print(f"✔ saved: dist_{col}.png")


# ---------------- 3) Correlation Heatmap ----------------
plt.figure(figsize=(12,10))
corr = df[numeric_cols].corr()
sns.heatmap(corr, cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "corr_heatmap.png")
plt.close()

print("✔ saved: corr_heatmap.png")


# ---------------- 4) AQI vs PM2.5 Scatter ----------------
if "aqi" in df.columns and "pm25" in df.columns:
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=df["pm25"], y=df["aqi"], alpha=0.4)
    plt.title("AQI vs PM2.5")
    plt.xlabel("PM2.5")
    plt.ylabel("AQI")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "scatter_aqi_pm25.png")
    plt.close()

    print("✔ saved: scatter_aqi_pm25.png")


# ---------------- 5) Station Bias ----------------
# แสดงค่า AQI เฉลี่ยของแต่ละสถานี
if "station_idx" in df.columns:
    station_mean = df.groupby("station_idx")["aqi"].mean().sort_values(ascending=False).head(20)

    plt.figure(figsize=(8,6))
    sns.barplot(x=station_mean.values, y=station_mean.index)
    plt.title("Top 20 Stations with Highest Average AQI")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "station_bias.png")
    plt.close()

    print("✔ saved: station_bias.png")


# ---------------- 6) Time Trend (ถ้ามี timestamp) ----------------
if "station_time" in df.columns:
    try:
        df["station_time"] = pd.to_datetime(df["station_time"])

        # ดูแนวโน้ม AQI เฉลี่ยของวัน
        df_daily = df.groupby(df["station_time"].dt.date)["aqi"].mean()

        plt.figure(figsize=(10,5))
        plt.plot(df_daily.index, df_daily.values)
        plt.xticks(rotation=45)
        plt.title("Average AQI Over Time")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "aqi_time_trend.png")
        plt.close()

        print("✔ saved: aqi_time_trend.png")
    except:
        print("⚠ Cannot parse station_time into datetime")


print("\n🎉 All EDA plots saved to:", OUTPUT_DIR)

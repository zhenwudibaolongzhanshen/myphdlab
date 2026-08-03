"""下载 Bike Sharing 数据集并预处理为 .npz 格式

数据来源: UCI Bike Sharing Dataset
输出: bike_sharing.npz (X, y 已标准化的滑动窗口数据)
"""

import numpy as np
import os, sys, io, zipfile, urllib.request
from sklearn.preprocessing import StandardScaler

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bike_sharing", "data")
os.makedirs(DATA_DIR, exist_ok=True)

ZIP_PATH = os.path.join(DATA_DIR, "Bike-Sharing-Dataset.zip")
CSV_PATH = os.path.join(DATA_DIR, "hour.csv")
NPZ_PATH = os.path.join(DATA_DIR, "bike_sharing.npz")

# 下载
if not os.path.exists(CSV_PATH):
    if not os.path.exists(ZIP_PATH):
        print("Downloading Bike Sharing dataset...")
        urllib.request.urlretrieve(URL, ZIP_PATH)
    print("Extracting...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        zf.extract("hour.csv", DATA_DIR)

# 加载
import pandas as pd
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} records, {df.shape[1]} columns")

# 特征工程
feature_cols = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday',
                'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
target_col = 'cnt'

X_raw = df[feature_cols].values.astype(np.float32)
y_raw = df[target_col].values.astype(np.float32).reshape(-1, 1)

# 滑动窗口
window_size = 24
stride = 6

X_windows, y_windows = [], []
for i in range(0, len(X_raw) - window_size, stride):
    X_windows.append(X_raw[i:i + window_size])
    y_windows.append(y_raw[i + window_size - 1])  # 预测窗口最后一时刻

X_arr = np.stack(X_windows)  # [N, 24, 12]
y_arr = np.stack(y_windows)  # [N, 1]

# 标准化：对 X 的每个特征在时间维上全局标准化
N, T, F = X_arr.shape
X_flat = X_arr.reshape(-1, F)
x_scaler = StandardScaler().fit(X_flat)
X_scaled = x_scaler.transform(X_flat).reshape(N, T, F).astype(np.float32)

y_scaler = StandardScaler().fit(y_arr)
y_scaled = y_scaler.transform(y_arr).astype(np.float32)

np.savez(NPZ_PATH, X=X_scaled, y=y_scaled,
         x_mean=x_scaler.mean_, x_scale=x_scaler.scale_,
         y_mean=y_scaler.mean_, y_scale=y_scaler.scale_,
         window_size=window_size, stride=stride, n_features=F)

print(f"Saved {N} windows to {NPZ_PATH}")
print(f"  X shape: {X_scaled.shape}, y shape: {y_scaled.shape}")
print(f"  y range: [{y_scaled.min():.2f}, {y_scaled.max():.2f}]")

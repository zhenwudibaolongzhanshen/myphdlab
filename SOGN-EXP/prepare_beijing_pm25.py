"""下载并预处理 Beijing PM2.5 数据集，生成滑动窗口保存为 .npz

输出: experiments/beijing_pm25/data/windows.npz
  - X: (N, W, F)  float32, W=24h 滑动窗口
  - y: (N,)       float32, 对应窗口末尾时刻的 pm2.5 值
"""

import numpy as np
import pandas as pd
import os
import sys

# === 配置 ===
WINDOW = 24           # 24 小时窗口
STRIDE = 6            # 步长 6h，每 seed ~7300 样本，7方法×3seed 约 25min
TARGET = "pm2.5"
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_data_2010.1.1-2014.12.31.csv"

SAVE_DIR = os.path.join(os.path.dirname(__file__), "beijing_pm25", "data")
SAVE_PATH = os.path.join(SAVE_DIR, "windows.npz")


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ---- 1. 下载 ----
    print("Downloading Beijing PM2.5 dataset from UCI ...")
    try:
        df = pd.read_csv(URL)
    except Exception:
        print("ERROR: 下载失败，请手动下载并放到 beijing_pm25/data/ 下")
        sys.exit(1)
    print(f"  Loaded: {df.shape}")

    # ---- 2. 清洗 ----
    # 删除 No 列（行号），提取特征
    df = df.drop(columns=["No"], errors="ignore")

    # pm2.5 缺失值：前向填充
    missing_before = df[TARGET].isna().sum()
    df[TARGET] = df[TARGET].fillna(method="ffill")
    if df[TARGET].isna().sum() > 0:
        df[TARGET] = df[TARGET].fillna(method="bfill")
    print(f"  pm2.5 missing: {missing_before} → filled, remaining {df[TARGET].isna().sum()}")

    # cbwd 风向 → one-hot
    cbwd = pd.get_dummies(df["cbwd"], prefix="cbwd", drop_first=False)
    df = df.drop(columns=["cbwd"])
    df = pd.concat([df, cbwd], axis=1)

    # 验证无缺失
    assert df.isna().sum().sum() == 0, "Still have missing values"
    print(f"  Cleaned: {df.shape}")

    # ---- 3. 特征与目标 ----
    target_arr = df[TARGET].values.astype(np.float32)
    df = df.drop(columns=[TARGET])
    feature_cols = df.columns.tolist()
    feature_arr = df.values.astype(np.float32)
    print(f"  Features: {len(feature_cols)}")

    # ---- 4. 滑动窗口 ----
    N = len(feature_arr)
    windows = []
    targets = []
    for i in range(N - WINDOW):
        if i % STRIDE == 0:
            windows.append(feature_arr[i:i + WINDOW])
            targets.append(target_arr[i + WINDOW - 1])  # 窗口末尾的 pm2.5

    X = np.stack(windows, axis=0)  # (n_windows, W, F)
    y = np.array(targets, dtype=np.float32)

    print(f"  Windows: {X.shape}, Targets: {y.shape}")
    print(f"  y range: [{y.min():.2f}, {y.max():.2f}]")

    # ---- 5. 保存 ----
    np.savez_compressed(SAVE_PATH, X=X, y=y, feature_cols=np.array(feature_cols),
                        window=WINDOW, stride=STRIDE)
    print(f"Saved to {SAVE_PATH}")


if __name__ == "__main__":
    main()

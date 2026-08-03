"""共享数据加载、分割、分箱"""

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
import os


def load_california_housing(random_state=42, test_size=0.2, val_size=0.1):
    """加载 California Housing，返回 train/val/test 分割"""
    data = fetch_california_housing()
    X, y = data.data, data.target  # (20640, 8), (20640,)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return (torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32).view(-1, 1),
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32).view(-1, 1),
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32).view(-1, 1),
            scaler, X_train.shape[1])


def load_beijing_pm25(random_state=42, test_size=0.2, val_size=0.1):
    """加载预处理好的 Beijing PM2.5 数据，返回 train/val/test 分割"""
    data_dir = os.path.join(os.path.dirname(__file__), "..", "beijing_pm25", "data")
    data_file = os.path.join(data_dir, "windows.npz")
    if not os.path.exists(data_file):
        raise FileNotFoundError(
            f"{data_file} 不存在，请先运行 prepare_beijing_pm25.py")
    d = np.load(data_file)
    X, y = d["X"], d["y"]  # (N, W, F), (N,)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state)

    # 展平标准化（时序数据按特征维标准化）
    N_train, W, F = X_train.shape
    X_train_flat = X_train.reshape(-1, F)
    scaler = StandardScaler().fit(X_train_flat)
    X_train = scaler.transform(X_train_flat).reshape(N_train, W, F)
    X_val = scaler.transform(X_val.reshape(-1, F)).reshape(-1, W, F)
    X_test = scaler.transform(X_test.reshape(-1, F)).reshape(-1, W, F)

    return (torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32).view(-1, 1),
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32).view(-1, 1),
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32).view(-1, 1),
            scaler, (W, F))  # input_dim = (seq_len, n_features)


def load_bike_sharing(random_state=42, test_size=0.2, val_size=0.1):
    """加载预处理好的 Bike Sharing 数据，返回 train/val/test 分割"""
    data_dir = os.path.join(os.path.dirname(__file__), "..", "bike_sharing", "data")
    data_file = os.path.join(data_dir, "bike_sharing.npz")
    if not os.path.exists(data_file):
        raise FileNotFoundError(
            f"{data_file} 不存在，请先运行 prepare_bike_sharing.py")
    d = np.load(data_file)
    X, y = d["X"], d["y"]  # (N, W, F), (N, 1)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state)

    # 展平标准化（在训练集上拟合）
    N_train, W, F = X_train.shape
    X_train_flat = X_train.reshape(-1, F)
    scaler = StandardScaler().fit(X_train_flat)
    X_train = scaler.transform(X_train_flat).reshape(N_train, W, F)
    X_val = scaler.transform(X_val.reshape(-1, F)).reshape(-1, W, F)
    X_test = scaler.transform(X_test.reshape(-1, F)).reshape(-1, W, F)

    return (torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32).view(-1, 1),
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32).view(-1, 1),
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32).view(-1, 1),
            scaler, (W, F))


def quantile_bins(y_train_np, K=6):
    """等频分箱边界，返回 (K+1,) 的边界数组"""
    quantiles = np.linspace(0, 1, K + 1)[1:-1]
    bin_edges = np.quantile(y_train_np, quantiles)
    return np.concatenate([[y_train_np.min() - 1e-6], bin_edges, [y_train_np.max() + 1e-6]])


def to_bin_labels(y_arr, bin_edges):
    """将连续值映射到 0..K-1 的区间标签"""
    return np.searchsorted(bin_edges[1:-1], y_arr, side="right").astype(np.int64)


def build_mlp_encoder(input_dim, hidden_dims=[128, 64], dropout=0.0):
    """构建 MLP 编码器"""
    layers = []
    prev = input_dim
    for h in hidden_dims:
        layers.append(torch.nn.Linear(prev, h))
        layers.append(torch.nn.ReLU())
        if dropout > 0:
            layers.append(torch.nn.Dropout(dropout))
        prev = h
    return torch.nn.Sequential(*layers), prev


def build_lstm_encoder(input_dim, hidden_size=64, num_layers=1, dropout=0.0):
    """
    构建 LSTM 编码器，input_dim = (seq_len, n_features) 中的 n_features。
    返回最后一个 time step 的 hidden state 作为隐表示。
    """
    lstm = torch.nn.LSTM(input_dim, hidden_size, num_layers,
                         batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
    return lstm, hidden_size

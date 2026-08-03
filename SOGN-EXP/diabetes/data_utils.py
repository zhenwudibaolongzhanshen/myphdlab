"""共享数据加载 — diabetes 数据集"""
import numpy as np
import torch
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def prepare_diabetes_data(test_size=0.2, val_size=0.1, random_state=42):
    """所有方法使用相同的数据划分，确保公平对比"""
    data = load_diabetes()
    X, y = data.data, data.target

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    input_dim = X_train.shape[1]
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}, Features: {input_dim}")
    return (X_train_t, y_train_t, X_val_t, y_val_t, X_test_t, y_test_t, scaler, input_dim)


def build_encoder(input_dim, hidden_dims=[128, 64], dropout=0.0):
    """构建共享编码器，可选 dropout"""
    layers = []
    prev_dim = input_dim
    for h_dim in hidden_dims:
        layers.append(torch.nn.Linear(prev_dim, h_dim))
        layers.append(torch.nn.ReLU())
        if dropout > 0:
            layers.append(torch.nn.Dropout(dropout))
        prev_dim = h_dim
    return torch.nn.Sequential(*layers), prev_dim

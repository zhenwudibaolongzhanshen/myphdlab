"""6. Split Conformal — 校准集残差分位数 + k-NN局部不确定性"""
import numpy as np
import torch
import torch.nn as nn
import json
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from data_utils import prepare_diabetes_data, build_encoder
from sklearn.neighbors import KNeighborsRegressor

CONFIG = {
    "method": "conformal",
    "hidden_dims": [128, 64],
    "epochs": 200,
    "batch_size": 32,
    "lr": 1e-3,
    "cal_ratio": 0.3,    # 训练集中划出30%做校准集
    "k_neighbors": 20,
    "random_state": 42,
}
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "conformal")
RESULT_DIR = os.path.join(os.path.dirname(__file__), "results", "conformal")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


class ConformalRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        self.encoder, hidden_dim = build_encoder(input_dim, hidden_dims, dropout=0.0)
        self.reg_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        return self.reg_head(self.encoder(x))

    def get_features(self, x):
        """返回隐表示，用于kNN"""
        return self.encoder(x)


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    X_train, y_train, X_val, y_val, X_test, y_test, scaler, input_dim = prepare_diabetes_data()

    # 从训练集划出校准集
    n_train = len(X_train)
    n_cal = int(n_train * CONFIG["cal_ratio"])
    idx = np.random.RandomState(CONFIG["random_state"]).permutation(n_train)
    cal_idx, train_idx = idx[:n_cal], idx[n_cal:]
    X_cal, y_cal = X_train[cal_idx], y_train[cal_idx]
    X_train_sub, y_train_sub = X_train[train_idx], y_train[train_idx]

    print(f"Proper train: {len(X_train_sub)}, Calibration: {len(X_cal)}")

    train_ds = torch.utils.data.TensorDataset(X_train_sub, y_train_sub)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)

    model = ConformalRegressor(input_dim, CONFIG["hidden_dims"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    for epoch in range(CONFIG["epochs"]):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val.to(device)), y_val.to(device)).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "model.pt"))

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1:3d} | Val Loss: {val_loss:.4f}")

    # 在 calibration set 上计算残差 + 训练 kNN 局部不确定性
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "model.pt")))
    model.eval()
    with torch.no_grad():
        cal_pred = model(X_cal.to(device)).cpu().numpy().squeeze()
        cal_residuals = np.abs(y_cal.numpy().squeeze() - cal_pred)
        cal_features = model.get_features(X_cal.to(device)).cpu().numpy()

    # 训练 kNN：用隐表示预测残差（局部不确定性）
    knn = KNeighborsRegressor(n_neighbors=CONFIG["k_neighbors"])
    knn.fit(cal_features, cal_residuals)

    # 测试评估
    with torch.no_grad():
        test_pred = model(X_test.to(device)).cpu().numpy().squeeze()
        test_features = model.get_features(X_test.to(device)).cpu().numpy()
    scores = -knn.predict(test_features)  # 负残差：越高越好

    np.save(os.path.join(RESULT_DIR, "test_predictions.npy"), test_pred)
    np.save(os.path.join(RESULT_DIR, "test_scores.npy"), scores)
    with open(os.path.join(RESULT_DIR, "config.json"), "w") as f:
        json.dump(CONFIG, f, indent=2)

    print(f"Model saved. Pred residual mean={-scores.mean():.3f}, std={scores.std():.3f}")


if __name__ == "__main__":
    train()

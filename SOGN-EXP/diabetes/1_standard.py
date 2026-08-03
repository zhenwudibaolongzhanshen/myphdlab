"""1. 标准回归 — 无筛选基线"""
import numpy as np
import torch
import torch.nn as nn
import json
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from data_utils import prepare_diabetes_data, build_encoder

CONFIG = {
    "method": "standard",
    "hidden_dims": [128, 64],
    "epochs": 200,
    "batch_size": 32,
    "lr": 1e-3,
    "random_state": 42,
}
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "standard")
RESULT_DIR = os.path.join(os.path.dirname(__file__), "results", "standard")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


class StandardRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        self.encoder, hidden_dim = build_encoder(input_dim, hidden_dims, dropout=0.0)
        self.reg_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        return self.reg_head(self.encoder(x))


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    X_train, y_train, X_val, y_val, X_test, y_test, scaler, input_dim = prepare_diabetes_data()
    train_ds = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)

    model = StandardRegressor(input_dim, CONFIG["hidden_dims"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    for epoch in range(CONFIG["epochs"]):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_ds)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val.to(device)), y_val.to(device)).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "model.pt"))

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # 测试评估
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "model.pt")))
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test.to(device)).cpu().numpy().squeeze()
        # 标准回归无不确定性 → 用随机分数（coverage_eval 会在 coverage=1 用全量）
        scores = np.random.permutation(len(y_pred)).astype(np.float32)  # 仅保证格式一致

    np.save(os.path.join(RESULT_DIR, "test_predictions.npy"), y_pred)
    np.save(os.path.join(RESULT_DIR, "test_scores.npy"), scores)
    with open(os.path.join(RESULT_DIR, "config.json"), "w") as f:
        json.dump(CONFIG, f, indent=2)

    print(f"Model saved to {MODEL_DIR}")
    print(f"Predictions: {y_pred.shape}, Scores (random): {scores.shape}")


if __name__ == "__main__":
    train()

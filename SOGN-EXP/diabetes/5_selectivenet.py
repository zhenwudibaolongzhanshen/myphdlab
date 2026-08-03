"""5. SelectiveNet — 内嵌拒绝头，联合优化预测+筛选"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from data_utils import prepare_diabetes_data, build_encoder

CONFIG = {
    "method": "selectivenet",
    "hidden_dims": [128, 64],
    "epochs": 200,
    "batch_size": 32,
    "lr": 1e-3,
    "target_coverage": 0.7,
    "lambda_cov": 10.0,
    "random_state": 42,
}
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "selectivenet")
RESULT_DIR = os.path.join(os.path.dirname(__file__), "results", "selectivenet")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


class SelectiveNet(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        self.encoder, hidden_dim = build_encoder(input_dim, hidden_dims, dropout=0.0)
        self.pred_head = nn.Linear(hidden_dim, 1)   # 预测
        self.sel_head = nn.Linear(hidden_dim, 1)    # 选择分数

    def forward(self, x):
        h = self.encoder(x)
        y_hat = self.pred_head(h)
        s = torch.sigmoid(self.sel_head(h))  # s ∈ (0, 1)
        return y_hat, s


def selective_loss(y_true, y_pred, s, target_cov, lambda_cov):
    """选择性回归损失"""
    mse = (y_true - y_pred) ** 2
    # 覆盖损失：确保选择比例接近 target_cov
    cov_loss = lambda_cov * F.relu(target_cov - s.mean()) ** 2
    # 加权回归损失（仅对选定样本）
    if s.mean() > 1e-8:
        reg_loss = (s * mse).sum() / s.sum()
    else:
        reg_loss = mse.mean()
    return reg_loss + cov_loss


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    X_train, y_train, X_val, y_val, X_test, y_test, scaler, input_dim = prepare_diabetes_data()
    train_ds = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)

    model = SelectiveNet(input_dim, CONFIG["hidden_dims"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])

    best_val_loss = float('inf')
    for epoch in range(CONFIG["epochs"]):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            y_hat, s = model(xb)
            loss = selective_loss(yb, y_hat, s, CONFIG["target_coverage"], CONFIG["lambda_cov"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_ds)

        model.eval()
        with torch.no_grad():
            y_hat_v, s_v = model(X_val.to(device))
            val_loss = selective_loss(y_val.to(device), y_hat_v, s_v,
                                       CONFIG["target_coverage"], CONFIG["lambda_cov"]).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "model.pt"))

        if (epoch + 1) % 50 == 0:
            with torch.no_grad():
                s_mean = s_v.mean().item()
            print(f"Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | Val: {val_loss:.4f} | Coverage: {s_mean:.3f}")

    # 测试评估
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "model.pt")))
    model.eval()
    with torch.no_grad():
        y_pred, s = model(X_test.to(device))
        y_pred = y_pred.cpu().numpy().squeeze()
        scores = s.cpu().numpy().squeeze()  # s 越高越应该被选

    np.save(os.path.join(RESULT_DIR, "test_predictions.npy"), y_pred)
    np.save(os.path.join(RESULT_DIR, "test_scores.npy"), scores)
    with open(os.path.join(RESULT_DIR, "config.json"), "w") as f:
        json.dump(CONFIG, f, indent=2)

    print(f"Model saved. Selection score mean={scores.mean():.3f}, std={scores.std():.3f}")


if __name__ == "__main__":
    train()

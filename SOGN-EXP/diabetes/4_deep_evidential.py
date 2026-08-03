"""4. Deep Evidential Regression — NIG先验，输出(γ,ν,α,β)"""
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
    "method": "deep_evidential",
    "hidden_dims": [128, 64],
    "epochs": 200,
    "batch_size": 32,
    "lr": 1e-3,
    "lambda_reg": 0.01,
    "random_state": 42,
}
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "deep_evidential")
RESULT_DIR = os.path.join(os.path.dirname(__file__), "results", "deep_evidential")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


class EvidentialRegressor(nn.Module):
    """输出 NIG 分布四参数: gamma(均值), nu(证据), alpha, beta"""
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        self.encoder, hidden_dim = build_encoder(input_dim, hidden_dims, dropout=0.0)
        self.head = nn.Linear(hidden_dim, 4)

    def forward(self, x):
        out = self.head(self.encoder(x))
        gamma = out[:, 0:1]
        nu = F.softplus(out[:, 1:2]) + 1.0
        alpha = F.softplus(out[:, 2:3]) + 1.0
        beta = F.softplus(out[:, 3:4])
        return gamma, nu, alpha, beta


def nig_nll(y, gamma, nu, alpha, beta):
    """Normal-Inverse-Gamma 负对数似然"""
    omega = 2.0 * beta * (1.0 + nu)
    nll = (0.5 * (torch.pi / nu).log()
           - alpha * omega.log()
           + (alpha + 0.5) * ((y - gamma) ** 2 * nu + omega).log()
           + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5))
    return nll.mean()


def evidential_loss(y, gamma, nu, alpha, beta, lambda_reg=0.01):
    """证据回归总损失 = NLL + 正则项"""
    nll = nig_nll(y, gamma, nu, alpha, beta)
    reg = lambda_reg * (torch.abs(y - gamma) * (2.0 * nu + alpha)).mean()
    return nll + reg


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    X_train, y_train, X_val, y_val, X_test, y_test, scaler, input_dim = prepare_diabetes_data()
    train_ds = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)

    model = EvidentialRegressor(input_dim, CONFIG["hidden_dims"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])

    best_val_loss = float('inf')
    for epoch in range(CONFIG["epochs"]):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            gamma, nu, alpha, beta = model(xb)
            loss = evidential_loss(yb, gamma, nu, alpha, beta, CONFIG["lambda_reg"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_ds)

        model.eval()
        with torch.no_grad():
            gamma, nu, alpha, beta = model(X_val.to(device))
            val_loss = evidential_loss(y_val.to(device), gamma, nu, alpha, beta, CONFIG["lambda_reg"]).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "model.pt"))

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # 测试评估：ν (证据量) 越高 = 越确定
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "model.pt")))
    model.eval()
    with torch.no_grad():
        gamma, nu, alpha, beta = model(X_test.to(device))
        y_pred = gamma.cpu().numpy().squeeze()
        scores = nu.cpu().numpy().squeeze()  # ν 越高越有把握

    np.save(os.path.join(RESULT_DIR, "test_predictions.npy"), y_pred)
    np.save(os.path.join(RESULT_DIR, "test_scores.npy"), scores)
    with open(os.path.join(RESULT_DIR, "config.json"), "w") as f:
        json.dump(CONFIG, f, indent=2)

    print(f"Model saved. Nu mean={scores.mean():.3f}, std={scores.std():.3f}")


if __name__ == "__main__":
    train()

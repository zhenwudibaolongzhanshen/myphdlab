"""3. Deep Ensemble — 5个独立模型，预测方差 = 不确定性"""
import numpy as np
import torch
import torch.nn as nn
import json
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from data_utils import prepare_diabetes_data, build_encoder

CONFIG = {
    "method": "deep_ensemble",
    "hidden_dims": [128, 64],
    "n_models": 5,
    "epochs": 200,
    "batch_size": 32,
    "lr": 1e-3,
    "random_state": 42,
}
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "deep_ensemble")
RESULT_DIR = os.path.join(os.path.dirname(__file__), "results", "deep_ensemble")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


class EnsembleRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        self.encoder, hidden_dim = build_encoder(input_dim, hidden_dims, dropout=0.0)
        self.reg_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        return self.reg_head(self.encoder(x))


def train_single(seed, input_dim, device, train_loader, X_val, y_val, n_train):
    """训练单个模型"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = EnsembleRegressor(input_dim, CONFIG["hidden_dims"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
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
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return best_state, best_val_loss


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    X_train, y_train, X_val, y_val, X_test, y_test, scaler, input_dim = prepare_diabetes_data()
    train_ds = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)

    models = []
    for m in range(CONFIG["n_models"]):
        seed = CONFIG["random_state"] + m * 100
        state, val_loss = train_single(seed, input_dim, device, train_loader, X_val, y_val, len(train_ds))
        torch.save(state, os.path.join(MODEL_DIR, f"model_{m}.pt"))
        models.append(state)
        print(f"Model {m+1}/{CONFIG['n_models']} trained, val_loss={val_loss:.4f}")

    # 测试评估：所有模型预测
    all_preds = []
    for m in range(CONFIG["n_models"]):
        model = EnsembleRegressor(input_dim, CONFIG["hidden_dims"]).to(device)
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"model_{m}.pt")))
        model.eval()
        with torch.no_grad():
            pred = model(X_test.to(device)).cpu().numpy().squeeze()
        all_preds.append(pred)

    all_preds = np.array(all_preds)  # [n_models, n_test]
    y_mean = all_preds.mean(axis=0)
    y_var = all_preds.var(axis=0)
    scores = -y_var  # 负方差：越高越好

    np.save(os.path.join(RESULT_DIR, "test_predictions.npy"), y_mean)
    np.save(os.path.join(RESULT_DIR, "test_scores.npy"), scores)
    with open(os.path.join(RESULT_DIR, "config.json"), "w") as f:
        json.dump(CONFIG, f, indent=2)

    print(f"Ensemble trained. Uncertainty mean={y_var.mean():.4f}, std={y_var.std():.4f}")


if __name__ == "__main__":
    train()

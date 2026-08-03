"""0. SOGN — 序数门控网络（外部参照原则）"""
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
    "method": "sogn",
    "hidden_dims": [128, 64],
    "K": 6,
    "alpha": 5.0,
    "eps": 0.1,
    "gamma_softmin": 10.0,
    "beta_init": 1.0,
    "beta_max": 50.0,
    "anneal_steps": 100,
    "lambda_cov": 0.1,
    "target_coverage": 0.2,
    "pretrain_epochs": 30,
    "epochs": 150,
    "batch_size": 32,
    "lr": 1e-3,
    "random_state": 42,
}
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "sogn")
RESULT_DIR = os.path.join(os.path.dirname(__file__), "results", "sogn")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ============ 门控模块 ============
class SoftHC(nn.Module):
    def __init__(self, alpha=5.0, tau=0.5, beta_init=1.0, beta_max=50.0, anneal_steps=100):
        super().__init__()
        self.alpha = alpha
        self.tau = tau
        self.beta_init = beta_init
        self.beta_max = beta_max
        self.anneal_steps = anneal_steps
        self.current_step = 0

    def set_step(self, step):
        self.current_step = step

    def get_beta(self):
        if self.current_step >= self.anneal_steps:
            return self.beta_max
        ratio = self.current_step / self.anneal_steps
        return self.beta_init + ratio * (self.beta_max - self.beta_init)

    def forward(self, p):
        pow_mean = ((p ** self.alpha).mean(dim=-1)) ** (1.0 / self.alpha)
        beta = self.get_beta()
        return torch.sigmoid(beta * (pow_mean - self.tau))


class SoftLS(nn.Module):
    def __init__(self, eps=0.1, beta_init=1.0, beta_max=50.0, anneal_steps=100):
        super().__init__()
        self.eps = eps
        self.beta_init = beta_init
        self.beta_max = beta_max
        self.anneal_steps = anneal_steps
        self.current_step = 0

    def set_step(self, step):
        self.current_step = step

    def get_beta(self):
        if self.current_step >= self.anneal_steps:
            return self.beta_max
        ratio = self.current_step / self.anneal_steps
        return self.beta_init + ratio * (self.beta_max - self.beta_init)

    def forward(self, p):
        K = p.size(-1)
        kernel = torch.ones(1, 1, 3, device=p.device, dtype=p.dtype)
        p_unsqueezed = p.unsqueeze(1)
        window_sums = F.conv1d(p_unsqueezed, kernel, padding=0).squeeze(1)
        beta = self.get_beta()
        lse = torch.logsumexp(beta * window_sums, dim=-1) / beta
        return torch.sigmoid(beta * (lse - (1.0 - self.eps)))


class HCLSGate(nn.Module):
    def __init__(self, gamma=10.0, **kwargs):
        super().__init__()
        self.gamma = gamma
        self.hc = SoftHC(**{k: v for k, v in kwargs.items()
                            if k in ['alpha', 'tau', 'beta_init', 'beta_max', 'anneal_steps']})
        self.ls = SoftLS(**{k: v for k, v in kwargs.items()
                            if k in ['eps', 'beta_init', 'beta_max', 'anneal_steps']})

    def set_step(self, step):
        self.hc.set_step(step)
        self.ls.set_step(step)

    def forward(self, p):
        a, b = self.hc(p), self.ls(p)
        m = torch.min(a, b)
        return m - (1.0 / self.gamma) * torch.log(
            torch.exp(-self.gamma * (a - m)) + torch.exp(-self.gamma * (b - m)))


# ============ SOGN 模型 ============
class SOGN(nn.Module):
    def __init__(self, input_dim, hidden_dims, K, hc_ls_kwargs):
        super().__init__()
        self.encoder, hidden_dim = build_encoder(input_dim, hidden_dims, dropout=0.0)
        self.reg_head = nn.Linear(hidden_dim, 1)
        self.ord_head = nn.Linear(hidden_dim, K)
        self.gate = HCLSGate(**hc_ls_kwargs)
        self.tau_raw = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        h = self.encoder(x)
        y_hat = self.reg_head(h)
        p = F.softmax(self.ord_head(h), dim=-1)
        self.gate.hc.tau = torch.sigmoid(self.tau_raw)
        w = self.gate(p)
        return y_hat, w, p

    def set_step(self, step):
        self.gate.set_step(step)


def compute_confidence_score(p, eps=0.1, ls_weight=1.0):
    """加分排序得分：越高越可预测"""
    max_prob, _ = p.max(dim=-1)
    K = p.size(-1)
    kernel = torch.ones(1, 1, 3, device=p.device, dtype=p.dtype)
    window_sums = F.conv1d(p.unsqueeze(1), kernel, padding=0).squeeze(1)
    max_window, _ = window_sums.max(dim=-1)
    ls_satisfied = (max_window > (1.0 - eps)).float()
    return max_prob + ls_weight * ls_satisfied


# ============ 训练 ============
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    X_train, y_train, X_val, y_val, X_test, y_test, scaler, input_dim = prepare_diabetes_data()

    # 分箱
    K = CONFIG["K"]
    y_train_np = y_train.numpy().squeeze()
    quantiles = np.linspace(0, 1, K + 1)[1:-1]
    bin_edges = np.quantile(y_train_np, quantiles)
    bin_edges = np.concatenate([[-np.inf], bin_edges, [np.inf]])

    def to_bins(y_arr):
        return np.digitize(y_arr, bin_edges[1:-1], right=True).astype(np.int64)

    y_train_bin = torch.tensor(to_bins(y_train_np), dtype=torch.long)
    y_val_bin = torch.tensor(to_bins(y_val.numpy().squeeze()), dtype=torch.long)
    y_test_bin = torch.tensor(to_bins(y_test.numpy().squeeze()), dtype=torch.long)

    train_ds = torch.utils.data.TensorDataset(X_train, y_train, y_train_bin)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)

    hc_ls_kwargs = {
        'alpha': CONFIG["alpha"], 'tau': 0.5, 'eps': CONFIG["eps"],
        'beta_init': CONFIG["beta_init"], 'beta_max': CONFIG["beta_max"],
        'anneal_steps': CONFIG["anneal_steps"], 'gamma': CONFIG["gamma_softmin"],
    }
    model = SOGN(input_dim, CONFIG["hidden_dims"], K, hc_ls_kwargs).to(device)

    # ---- 阶段一：序数预训练 ----
    pretrain_params = (list(model.encoder.parameters()) +
                       list(model.ord_head.parameters()) + [model.tau_raw])
    opt_pre = torch.optim.Adam(pretrain_params, lr=CONFIG["lr"])

    for epoch in range(CONFIG["pretrain_epochs"]):
        model.train()
        model.set_step(epoch)
        for xb, yb, bb in train_loader:
            xb, bb = xb.to(device), bb.to(device)
            _, w, p = model(xb)
            loss = F.cross_entropy(p, bb) + CONFIG["lambda_cov"] * F.relu(CONFIG["target_coverage"] - w.mean())
            opt_pre.zero_grad()
            loss.backward()
            opt_pre.step()
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                acc = (model(X_val.to(device))[2].argmax(dim=1) == y_val_bin.to(device)).float().mean()
            print(f"Pre Epoch {epoch+1:2d} | Ord Acc: {acc.item():.3f}")

    # ---- 阶段二：联合训练 ----
    opt = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    best_val_loss = float('inf')
    best_state = None

    for epoch in range(CONFIG["epochs"]):
        model.train()
        model.set_step(epoch + CONFIG["pretrain_epochs"])
        for xb, yb, bb in train_loader:
            xb, yb, bb = xb.to(device), yb.to(device), bb.to(device)
            y_hat, w, p = model(xb)
            loss_ord = F.cross_entropy(p, bb)
            loss_reg = (w.detach() * (y_hat - yb).pow(2).squeeze()).mean()
            loss_cov = CONFIG["lambda_cov"] * F.relu(CONFIG["target_coverage"] - w.mean())
            loss = loss_ord + loss_reg + loss_cov
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            y_hat_v, w_v, p_v = model(X_val.to(device))
            val_loss = (F.cross_entropy(p_v, y_val_bin.to(device)) +
                        (w_v.detach() * (y_hat_v - y_val.to(device)).pow(2).squeeze()).mean()
                        + CONFIG["lambda_cov"] * F.relu(torch.tensor(CONFIG["target_coverage"]) - w_v.mean())).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 50 == 0:
            print(f"Joint Epoch {epoch+1:3d} | Loss: {val_loss:.4f} | "
                  f"w mean: {w_v.mean().item():.3f} | tau: {torch.sigmoid(model.tau_raw).item():.3f}")

    # 保存最优模型
    torch.save(best_state, os.path.join(MODEL_DIR, "model.pt"))
    model.load_state_dict(best_state)

    # 测试评估
    model.eval()
    with torch.no_grad():
        y_hat_t, w_t, p_t = model(X_test.to(device))
        y_pred = y_hat_t.cpu().numpy().squeeze()
        scores = compute_confidence_score(p_t, eps=CONFIG["eps"]).cpu().numpy()

    np.save(os.path.join(RESULT_DIR, "test_predictions.npy"), y_pred)
    np.save(os.path.join(RESULT_DIR, "test_scores.npy"), scores)
    with open(os.path.join(RESULT_DIR, "config.json"), "w") as f:
        json.dump({k: str(v) if isinstance(v, dict) else v for k, v in CONFIG.items()}, f, indent=2)

    print(f"\nSOGN saved. Tau final: {torch.sigmoid(model.tau_raw).item():.3f}")
    print(f"Score mean: {scores.mean():.3f}, std: {scores.std():.3f}")


if __name__ == "__main__":
    train()

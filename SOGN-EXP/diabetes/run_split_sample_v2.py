"""Diabetes Split-Sample v2: Full Data vs Half Data vs Split-Sample.

target_coverage=0.55, matching paper's official Diabetes config.
All three variants share the same random data split for fair comparison.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json, os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(__file__))
from data_utils import prepare_diabetes_data, build_encoder

CONFIG = {
    "hidden_dims": [128, 64],
    "K": 6,
    "alpha": 5.0,
    "eps": 0.1,
    "gamma_softmin": 10.0,
    "beta_init": 1.0,
    "beta_max": 50.0,
    "anneal_steps": 100,
    "lambda_cov": 0.1,
    "target_coverage": 0.55,
    "pretrain_epochs": 30,
    "epochs": 150,
    "batch_size": 32,
    "lr": 1e-3,
    "seeds": [42, 43, 44],
}

BASE_DIR = os.path.dirname(__file__)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============ 门控模块 ============
class SoftHC(nn.Module):
    def __init__(self, alpha=5.0, tau=0.5, beta_init=1.0, beta_max=50.0, anneal_steps=100):
        super().__init__()
        self.alpha = alpha; self.tau = tau
        self.beta_init = beta_init; self.beta_max = beta_max
        self.anneal_steps = anneal_steps; self.current_step = 0
    def set_step(self, step): self.current_step = step
    def get_beta(self):
        if self.current_step >= self.anneal_steps: return self.beta_max
        return self.beta_init + (self.current_step / self.anneal_steps) * (self.beta_max - self.beta_init)
    def forward(self, p):
        pow_mean = ((p ** self.alpha).mean(dim=-1)) ** (1.0 / self.alpha)
        return torch.sigmoid(self.get_beta() * (pow_mean - self.tau))

class SoftLS(nn.Module):
    def __init__(self, eps=0.1, beta_init=1.0, beta_max=50.0, anneal_steps=100):
        super().__init__()
        self.eps = eps; self.beta_init = beta_init; self.beta_max = beta_max
        self.anneal_steps = anneal_steps; self.current_step = 0
    def set_step(self, step): self.current_step = step
    def get_beta(self):
        if self.current_step >= self.anneal_steps: return self.beta_max
        return self.beta_init + (self.current_step / self.anneal_steps) * (self.beta_max - self.beta_init)
    def forward(self, p):
        K = p.size(-1)
        kernel = torch.ones(1, 1, 3, device=p.device, dtype=p.dtype)
        window_sums = F.conv1d(p.unsqueeze(1), kernel, padding=0).squeeze(1)
        beta = self.get_beta()
        lse = torch.logsumexp(beta * window_sums, dim=-1) / beta
        return torch.sigmoid(beta * (lse - (1.0 - self.eps)))

class HCLSGate(nn.Module):
    def __init__(self, gamma=10.0, **kwargs):
        super().__init__()
        self.gamma = gamma
        hc_keys = ['alpha', 'tau', 'beta_init', 'beta_max', 'anneal_steps']
        ls_keys = ['eps', 'beta_init', 'beta_max', 'anneal_steps']
        self.hc = SoftHC(**{k: v for k, v in kwargs.items() if k in hc_keys})
        self.ls = SoftLS(**{k: v for k, v in kwargs.items() if k in ls_keys})
    def set_step(self, step): self.hc.set_step(step); self.ls.set_step(step)
    def forward(self, p):
        a, b = self.hc(p), self.ls(p)
        m = torch.min(a, b)
        return m - (1.0 / self.gamma) * torch.log(
            torch.exp(-self.gamma * (a - m)) + torch.exp(-self.gamma * (b - m)))

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
    def set_step(self, step): self.gate.set_step(step)

def compute_confidence_score(p, eps=0.1, ls_weight=1.0):
    max_prob, _ = p.max(dim=-1)
    K = p.size(-1)
    kernel = torch.ones(1, 1, 3, device=p.device, dtype=p.dtype)
    window_sums = F.conv1d(p.unsqueeze(1), kernel, padding=0).squeeze(1)
    max_window, _ = window_sums.max(dim=-1)
    ls_satisfied = (max_window > (1.0 - eps)).float()
    return max_prob + ls_weight * ls_satisfied


def make_model(input_dim):
    hc_ls_kwargs = {
        'alpha': CONFIG["alpha"], 'tau': 0.5, 'eps': CONFIG["eps"],
        'beta_init': CONFIG["beta_init"], 'beta_max': CONFIG["beta_max"],
        'anneal_steps': CONFIG["anneal_steps"], 'gamma': CONFIG["gamma_softmin"],
    }
    return SOGN(input_dim, CONFIG["hidden_dims"], CONFIG["K"], hc_ls_kwargs).to(DEVICE)


def train_standard(model, train_loader, X_val, y_val, y_val_bin):
    """Standard SOGN training (original recipe, no split)."""
    # Phase 1: Ordinal pretrain
    pretrain_params = (list(model.encoder.parameters()) +
                       list(model.ord_head.parameters()) + [model.tau_raw])
    opt_pre = torch.optim.Adam(pretrain_params, lr=CONFIG["lr"])
    for epoch in range(CONFIG["pretrain_epochs"]):
        model.train(); model.set_step(epoch)
        for xb, yb, bb in train_loader:
            xb, bb = xb.to(DEVICE), bb.to(DEVICE)
            _, w, p = model(xb)
            loss = F.cross_entropy(p, bb) + CONFIG["lambda_cov"] * F.relu(
                torch.tensor(CONFIG["target_coverage"], device=DEVICE) - w.mean())
            opt_pre.zero_grad(); loss.backward(); opt_pre.step()

    # Phase 2: Joint training
    opt = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    best_state, best_val = None, float('inf')
    for epoch in range(CONFIG["epochs"]):
        model.train(); model.set_step(epoch + CONFIG["pretrain_epochs"])
        for xb, yb, bb in train_loader:
            xb, yb, bb = xb.to(DEVICE), yb.to(DEVICE), bb.to(DEVICE)
            y_hat, w, p = model(xb)
            loss_ord = F.cross_entropy(p, bb)
            loss_reg = (w.detach() * (y_hat - yb).pow(2).squeeze()).mean()
            loss_cov = CONFIG["lambda_cov"] * F.relu(
                torch.tensor(CONFIG["target_coverage"], device=DEVICE) - w.mean())
            opt.zero_grad(); (loss_ord + loss_reg + loss_cov).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            y_hat_v, w_v, p_v = model(X_val.to(DEVICE))
            val_loss = (F.cross_entropy(p_v, y_val_bin.to(DEVICE)) +
                        (w_v.detach() * (y_hat_v - y_val.to(DEVICE)).pow(2).squeeze()).mean()
                        + CONFIG["lambda_cov"] * F.relu(
                            torch.tensor(CONFIG["target_coverage"], device=DEVICE) - w_v.mean())).item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def train_split(model, gate_loader, reg_loader, X_val, y_val, y_val_bin):
    """Split-sample training: gate on gate_half, regression on reg_half."""
    # Phase 1: Ordinal pretrain on gate half only
    pretrain_params = (list(model.encoder.parameters()) +
                       list(model.ord_head.parameters()) + [model.tau_raw])
    opt_pre = torch.optim.Adam(pretrain_params, lr=CONFIG["lr"])
    for epoch in range(CONFIG["pretrain_epochs"]):
        model.train(); model.set_step(epoch)
        for xb, yb, bb in gate_loader:
            xb, bb = xb.to(DEVICE), bb.to(DEVICE)
            _, w, p = model(xb)
            loss = F.cross_entropy(p, bb) + CONFIG["lambda_cov"] * F.relu(
                torch.tensor(CONFIG["target_coverage"], device=DEVICE) - w.mean())
            opt_pre.zero_grad(); loss.backward(); opt_pre.step()

    # Phase 2: Joint with alternating batches
    opt = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    best_state, best_val = None, float('inf')
    for epoch in range(CONFIG["epochs"]):
        model.train(); model.set_step(epoch + CONFIG["pretrain_epochs"])
        for (xb_g, yb_g, bb_g), (xb_r, yb_r, bb_r) in zip(gate_loader, reg_loader):
            xb_g, bb_g = xb_g.to(DEVICE), bb_g.to(DEVICE)
            xb_r, yb_r = xb_r.to(DEVICE), yb_r.to(DEVICE)
            # Gate half -> ord loss
            _, w_g, p_g = model(xb_g)
            loss_ord = F.cross_entropy(p_g, bb_g)
            loss_cov = CONFIG["lambda_cov"] * F.relu(
                torch.tensor(CONFIG["target_coverage"], device=DEVICE) - w_g.mean())
            # Reg half -> reg loss (stop-gradient)
            y_hat_r, _, _ = model(xb_r)
            _, w_r_det, _ = model(xb_r)
            loss_reg = (w_r_det.detach() * (y_hat_r - yb_r).pow(2).squeeze()).mean()
            opt.zero_grad(); (loss_ord + loss_reg + loss_cov).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            y_hat_v, w_v, p_v = model(X_val.to(DEVICE))
            val_loss = (F.cross_entropy(p_v, y_val_bin.to(DEVICE)) +
                        (w_v.detach() * (y_hat_v - y_val.to(DEVICE)).pow(2).squeeze()).mean()
                        + CONFIG["lambda_cov"] * F.relu(
                            torch.tensor(CONFIG["target_coverage"], device=DEVICE) - w_v.mean())).item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def train_one_seed(seed):
    sep = '=' * 50
    print(f'\n{sep}\nSeed {seed}\n{sep}')

    X_train, y_train, X_val, y_val, X_test, y_test, scaler, input_dim = prepare_diabetes_data()

    K = CONFIG["K"]
    y_train_np = y_train.numpy().squeeze()
    quantiles = np.linspace(0, 1, K + 1)[1:-1]
    bin_edges = np.quantile(y_train_np, quantiles)
    bin_edges = np.concatenate([[-np.inf], bin_edges, [np.inf]])
    def to_bins(y_arr):
        return np.digitize(y_arr, bin_edges[1:-1], right=True).astype(np.int64)

    y_train_bin = torch.tensor(to_bins(y_train_np), dtype=torch.long)
    y_val_bin = torch.tensor(to_bins(y_val.numpy().squeeze()), dtype=torch.long)

    # ---- Random 50/50 split (shared across variants) ----
    n_train = len(X_train)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_train)
    half = n_train // 2
    idx_a = indices[:half]
    idx_b = indices[half:]

    # ---- Variant 1: Full Data (all training samples) ----
    print('\n--- Full Data ---')
    full_ds = torch.utils.data.TensorDataset(X_train, y_train, y_train_bin)
    full_loader = torch.utils.data.DataLoader(full_ds, batch_size=CONFIG["batch_size"], shuffle=True)
    model_full = make_model(input_dim)
    model_full = train_standard(model_full, full_loader, X_val, y_val, y_val_bin)
    model_full.eval()
    with torch.no_grad():
        y_hat, _, p = model_full(X_test.to(DEVICE))
        y_pred_full = y_hat.cpu().numpy().squeeze()
        scores_full = compute_confidence_score(p, eps=CONFIG["eps"]).cpu().numpy()

    # ---- Variant 2: Half Data (standard SOGN on 50% data) ----
    print('\n--- Half Data ---')
    X_half_a, y_half_a, b_half_a = X_train[idx_a], y_train[idx_a], y_train_bin[idx_a]
    half_ds = torch.utils.data.TensorDataset(X_half_a, y_half_a, b_half_a)
    half_loader = torch.utils.data.DataLoader(half_ds, batch_size=CONFIG["batch_size"], shuffle=True)
    model_half = make_model(input_dim)
    model_half = train_standard(model_half, half_loader, X_val, y_val, y_val_bin)
    model_half.eval()
    with torch.no_grad():
        y_hat, _, p = model_half(X_test.to(DEVICE))
        y_pred_half = y_hat.cpu().numpy().squeeze()
        scores_half = compute_confidence_score(p, eps=CONFIG["eps"]).cpu().numpy()

    # ---- Variant 3: Split-Sample (gate on idx_a, regression on idx_b) ----
    print('\n--- Split-Sample ---')
    X_gate, y_gate, b_gate = X_train[idx_a], y_train[idx_a], y_train_bin[idx_a]
    X_reg, y_reg, b_reg = X_train[idx_b], y_train[idx_b], y_train_bin[idx_b]
    gate_ds = torch.utils.data.TensorDataset(X_gate, y_gate, b_gate)
    reg_ds = torch.utils.data.TensorDataset(X_reg, y_reg, b_reg)
    gate_loader = torch.utils.data.DataLoader(gate_ds, batch_size=CONFIG["batch_size"], shuffle=True)
    reg_loader = torch.utils.data.DataLoader(reg_ds, batch_size=CONFIG["batch_size"], shuffle=True)
    model_split = make_model(input_dim)
    model_split = train_split(model_split, gate_loader, reg_loader, X_val, y_val, y_val_bin)
    model_split.eval()
    with torch.no_grad():
        y_hat, _, p = model_split(X_test.to(DEVICE))
        y_pred_split = y_hat.cpu().numpy().squeeze()
        scores_split = compute_confidence_score(p, eps=CONFIG["eps"]).cpu().numpy()

    y_true = y_test.numpy().squeeze()
    from sklearn.metrics import r2_score
    print(f'\n  Full R2={r2_score(y_true, y_pred_full):.4f}  Half R2={r2_score(y_true, y_pred_half):.4f}  Split R2={r2_score(y_true, y_pred_split):.4f}')
    return (y_pred_full, scores_full, y_pred_half, scores_half, y_pred_split, scores_split, y_true)


if __name__ == "__main__":
    variants = ["full", "half", "split"]
    for seed in CONFIG["seeds"]:
        results = train_one_seed(seed)
        y_pred_full, scores_full, y_pred_half, scores_half, y_pred_split, scores_split, y_true = results
        for var_name, y_pred, scores in zip(variants,
            [y_pred_full, y_pred_half, y_pred_split],
            [scores_full, scores_half, scores_split]):
            sd = os.path.join(BASE_DIR, "results", f"sogn_{var_name}", f"seed_{seed}")
            os.makedirs(sd, exist_ok=True)
            np.save(os.path.join(sd, "test_predictions.npy"), y_pred)
            np.save(os.path.join(sd, "test_scores.npy"), scores)
            np.save(os.path.join(sd, "test_labels.npy"), y_true)
        print(f'Seed {seed} saved.')

    for var_name in variants:
        config_out = {k: v for k, v in CONFIG.items() if k != "seeds"}
        config_out["variant"] = var_name
        sd = os.path.join(BASE_DIR, "results", f"sogn_{var_name}")
        with open(os.path.join(sd, "config.json"), "w") as f:
            json.dump(config_out, f, indent=2)
    print('\nAll done.')

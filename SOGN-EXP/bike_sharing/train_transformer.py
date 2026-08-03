"""Bike Sharing — Transformer backbone: SOGN / Standard / MC Dropout

用法: python train_transformer.py
"""

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, os, sys, math
from sklearn.metrics import r2_score
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shared.data_utils import load_bike_sharing, quantile_bins, to_bin_labels

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results')
SEEDS = [42, 43, 44]

# ============================================================
# Positional Encoding
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])

# ============================================================
# SOGN 组件
# ============================================================
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
        self.eps = eps
        self.beta_init = beta_init; self.beta_max = beta_max
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

# ============================================================
class SOGN_Transformer(nn.Module):
    def __init__(self, n_features, d_model, nhead, num_layers, K, dim_feedforward=256, dropout=0.1, hc_ls_kwargs=None):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=100, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.reg_net = nn.Sequential(
            nn.Linear(d_model, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 1))
        self.ord_head = nn.Linear(d_model, K)
        self.gate = HCLSGate(**(hc_ls_kwargs or {}))
        self.tau_raw = nn.Parameter(torch.tensor(0.0))
    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        out = self.transformer(x)
        h = out.mean(dim=1)  # mean pooling over time steps
        y_hat = self.reg_net(h)
        p = F.softmax(self.ord_head(h), dim=-1)
        self.gate.hc.tau = torch.sigmoid(self.tau_raw)
        w = self.gate(p)
        return y_hat, w, p
    def set_step(self, step): self.gate.set_step(step)

class StandardTransformer(nn.Module):
    def __init__(self, n_features, d_model, nhead, num_layers, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=100, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)
    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        out = self.transformer(x)
        return self.head(out.mean(dim=1))

class MCDropoutTransformer(nn.Module):
    def __init__(self, n_features, d_model, nhead, num_layers, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=100, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.mc_dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1)
    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        out = self.transformer(x)
        return self.head(self.mc_dropout(out.mean(dim=1)))
    def predict_mc(self, x, n_samples=50):
        self.train()
        preds = []
        for _ in range(n_samples):
            with torch.no_grad(): preds.append(self.forward(x).cpu().numpy())
        preds = np.array(preds).squeeze(-1)
        return preds.mean(0), preds.var(0)

def compute_confidence_score(p, eps=0.1):
    max_prob, _ = p.max(dim=-1)
    K = p.size(-1)
    kernel = torch.ones(1, 1, 3, device=p.device, dtype=p.dtype)
    window_sums = F.conv1d(p.unsqueeze(1), kernel, padding=0).squeeze(1)
    max_window, _ = window_sums.max(dim=-1)
    return max_prob + (max_window > (1.0 - eps)).float()

# ============================================================
# Shared transformer config
# ============================================================
TF_CONFIG = {'d_model': 64, 'nhead': 4, 'num_layers': 2, 'dim_feedforward': 128, 'dropout': 0.1}

def train_sogn(seed):
    print(f'\n  [SOGN-Transformer] Seed {seed}')
    X_train, y_train, X_val, y_val, X_test, y_test, scaler, (seq_len, n_feat) = load_bike_sharing(random_state=seed)
    K = 6
    y_train_np = y_train.numpy().squeeze()
    bin_edges = quantile_bins(y_train_np, K)
    train_bin = torch.tensor(to_bin_labels(y_train_np, bin_edges), dtype=torch.long)
    train_ds = torch.utils.data.TensorDataset(X_train, y_train, train_bin)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)

    hc_ls_kwargs = {'alpha': 5.0, 'tau': 0.5, 'eps': 0.1, 'beta_init': 1.0,
                    'beta_max': 50.0, 'anneal_steps': 100, 'gamma': 10.0}
    model = SOGN_Transformer(n_feat, K=K, hc_ls_kwargs=hc_ls_kwargs, **TF_CONFIG).to(DEVICE)

    # Phase 1
    pretrain_params = (list(model.transformer.parameters()) +
                       list(model.input_proj.parameters()) +
                       list(model.ord_head.parameters()) + [model.tau_raw])
    opt_pre = torch.optim.Adam(pretrain_params, lr=1e-3)
    for epoch in range(30):
        model.train(); model.set_step(epoch)
        for xb, yb, bb in train_loader:
            xb, bb = xb.to(DEVICE), bb.to(DEVICE)
            _, w, p = model(xb)
            loss = F.cross_entropy(p, bb) + 0.2 * F.relu(torch.tensor(0.55, device=DEVICE) - w.mean())
            opt_pre.zero_grad(); loss.backward(); opt_pre.step()

    # Phase 2
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_state, best_val = None, float('inf')
    for epoch in range(100):
        model.train(); model.set_step(epoch + 30)
        for xb, yb, bb in train_loader:
            xb, yb, bb = xb.to(DEVICE), yb.to(DEVICE), bb.to(DEVICE)
            y_hat, w, p = model(xb)
            loss_ord = F.cross_entropy(p, bb)
            loss_reg = (w.detach() * (y_hat - yb).pow(2).squeeze()).mean()
            loss_cov = 0.2 * F.relu(torch.tensor(0.55, device=DEVICE) - w.mean())
            opt.zero_grad(); (loss_ord + loss_reg + loss_cov).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            y_hat_v, _, _ = model(X_val.to(DEVICE))
            vl = F.mse_loss(y_hat_v, y_val.to(DEVICE)).item()
        if vl < best_val: best_val = vl; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Phase 3
    model.load_state_dict(best_state)
    for p in model.ord_head.parameters(): p.requires_grad = False
    for p in model.gate.parameters(): p.requires_grad = False
    model.tau_raw.requires_grad = False
    ft_params = (list(model.transformer.parameters()) +
                 list(model.input_proj.parameters()) +
                 list(model.reg_net.parameters()))
    opt_ft = torch.optim.Adam(ft_params, lr=3e-4)
    best_ft_state, best_ft_val = best_state, best_val
    for epoch in range(60):
        model.train()
        for xb, yb, bb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            y_hat, w, p = model(xb)
            loss = (w.detach() * (y_hat - yb).pow(2).squeeze()).mean()
            opt_ft.zero_grad(); loss.backward(); opt_ft.step()
        model.eval()
        with torch.no_grad():
            y_hat_v, _, _ = model(X_val.to(DEVICE))
            vl = F.mse_loss(y_hat_v, y_val.to(DEVICE)).item()
        if vl < best_ft_val: best_ft_val = vl; best_ft_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_ft_state); model.eval()
    with torch.no_grad():
        y_hat_t, _, p_t = model(X_test.to(DEVICE))
        y_pred = y_hat_t.cpu().numpy().squeeze()
        scores = compute_confidence_score(p_t, eps=0.1).cpu().numpy()
        y_true = y_test.numpy().squeeze()
    return y_pred, scores, y_true

def train_standard(seed):
    print(f'\n  [Standard-Transformer] Seed {seed}')
    X_train, y_train, X_val, y_val, X_test, y_test, scaler, (seq_len, n_feat) = load_bike_sharing(random_state=seed)
    train_ds = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
    model = StandardTransformer(n_feat, **TF_CONFIG).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()
    best_state, best_val = None, float('inf')
    for _ in range(80):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
        model.eval()
        with torch.no_grad(): vl = crit(model(X_val.to(DEVICE)), y_val.to(DEVICE)).item()
        if vl < best_val: best_val = vl; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        y_pred = model(X_test.to(DEVICE)).cpu().numpy().squeeze()
        scores = np.random.permutation(len(y_pred)).astype(np.float32)
    return y_pred, scores, y_test.numpy().squeeze()

def train_mc_dropout(seed):
    print(f'\n  [MC Dropout-Transformer] Seed {seed}')
    X_train, y_train, X_val, y_val, X_test, y_test, scaler, (seq_len, n_feat) = load_bike_sharing(random_state=seed)
    train_ds = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
    model = MCDropoutTransformer(n_feat, **TF_CONFIG).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()
    best_state, best_val = None, float('inf')
    for _ in range(80):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
        model.eval()
        with torch.no_grad(): vl = crit(model(X_val.to(DEVICE)), y_val.to(DEVICE)).item()
        if vl < best_val: best_val = vl; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    y_mean, y_var = model.predict_mc(X_test.to(DEVICE), 50)
    return y_mean, -y_var, y_test.numpy().squeeze()

# ============================================================
if __name__ == '__main__':
    print(f"Device: {DEVICE}")
    for method_name, train_fn in [("sogn_transformer", train_sogn),
                                    ("standard_transformer", train_standard),
                                    ("mc_dropout_transformer", train_mc_dropout)]:
        for seed in SEEDS:
            y_pred, scores, y_true = train_fn(seed)
            sd = os.path.join(RESULT_DIR, method_name, f'seed_{seed}')
            os.makedirs(sd, exist_ok=True)
            np.save(os.path.join(sd, 'test_predictions.npy'), y_pred)
            np.save(os.path.join(sd, 'test_scores.npy'), scores)
            np.save(os.path.join(sd, 'test_labels.npy'), y_true)
            print(f'  {method_name} seed {seed} R2: {r2_score(y_true, y_pred):.4f}')
    print(f'\nDone. Transformer results saved.')

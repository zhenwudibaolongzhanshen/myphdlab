"""Train SOGN on Bike Sharing with 64-dim x 1-layer LSTM backbone (fair comparison)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json, os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shared.data_utils import load_bike_sharing, quantile_bins, to_bin_labels, build_lstm_encoder

CONFIG = {
    'method': 'sogn_64x1',
    'lstm_hidden': 64, 'lstm_layers': 1, 'lstm_dropout': 0.0,
    'K': 6, 'alpha': 5.0, 'eps': 0.1, 'gamma_softmin': 10.0,
    'beta_init': 1.0, 'beta_max': 50.0, 'anneal_steps': 80,
    'lambda_cov': 0.2, 'target_coverage': 0.55,
    'pretrain_epochs': 30, 'epochs': 120, 'finetune_epochs': 80,
    'batch_size': 64, 'lr': 1e-3, 'ft_lr': 3e-4,
    'seeds': [42, 43, 44],
}

RESULT_DIR = os.path.join(os.path.dirname(__file__), 'results', 'sogn_64x1')
os.makedirs(RESULT_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')


# ============================================================
# Gate modules (identical to original SOGN)
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
# SOGN Model (64x1 LSTM backbone)
# ============================================================
class SOGN_LSTM(nn.Module):
    def __init__(self, n_features, lstm_hidden, lstm_layers, lstm_dropout, K, hc_ls_kwargs):
        super().__init__()
        self.lstm, _ = build_lstm_encoder(n_features, lstm_hidden, lstm_layers, lstm_dropout)
        self.hidden_dim = lstm_hidden
        self.reg_net = nn.Sequential(
            nn.Linear(lstm_hidden, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 1))
        self.ord_head = nn.Linear(lstm_hidden, K)
        self.gate = HCLSGate(**hc_ls_kwargs)
        self.tau_raw = nn.Parameter(torch.tensor(0.0))
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        h = lstm_out[:, -1, :]
        y_hat = self.reg_net(h)
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


# ============================================================
# Training (identical 3-phase pipeline, same hyperparams)
# ============================================================
def train_one_seed(seed):
    sep = '=' * 50
    print(f'\n{sep}\nSeed {seed}\n{sep}')
    X_train, y_train, X_val, y_val, X_test, y_test, scaler, (seq_len, n_feat) = \
        load_bike_sharing(random_state=seed)

    K = CONFIG['K']
    y_train_np = y_train.numpy().squeeze()
    bin_edges = quantile_bins(y_train_np, K)
    train_bin = torch.tensor(to_bin_labels(y_train_np, bin_edges), dtype=torch.long)

    train_ds = torch.utils.data.TensorDataset(X_train, y_train, train_bin)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)

    hc_ls_kwargs = {
        'alpha': CONFIG['alpha'], 'tau': 0.5, 'eps': CONFIG['eps'],
        'beta_init': CONFIG['beta_init'], 'beta_max': CONFIG['beta_max'],
        'anneal_steps': CONFIG['anneal_steps'], 'gamma': CONFIG['gamma_softmin'],
    }
    model = SOGN_LSTM(n_feat, CONFIG['lstm_hidden'], CONFIG['lstm_layers'],
                      CONFIG['lstm_dropout'], K, hc_ls_kwargs).to(DEVICE)

    # ---- Phase 1: Pretrain ordinal classifier ----
    pretrain_params = (list(model.lstm.parameters()) +
                       list(model.ord_head.parameters()) + [model.tau_raw])
    opt_pre = torch.optim.Adam(pretrain_params, lr=CONFIG['lr'])
    for epoch in range(CONFIG['pretrain_epochs']):
        model.train(); model.set_step(epoch)
        for xb, yb, bb in train_loader:
            xb, bb = xb.to(DEVICE), bb.to(DEVICE)
            _, w, p = model(xb)
            loss = F.cross_entropy(p, bb) + CONFIG['lambda_cov'] * F.relu(
                torch.tensor(CONFIG['target_coverage'], device=DEVICE) - w.mean())
            opt_pre.zero_grad(); loss.backward(); opt_pre.step()
        if (epoch + 1) % 10 == 0:
            val_bin = torch.tensor(to_bin_labels(y_val.numpy().squeeze(), bin_edges), dtype=torch.long, device=DEVICE)
            with torch.no_grad():
                acc = (model(X_val.to(DEVICE))[2].argmax(dim=1) == val_bin).float().mean()
            print(f'  Pre Epoch {epoch+1:2d} | Ord Acc: {acc.item():.3f}')

    # ---- Phase 2: Joint training ----
    opt = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
    best_state, best_val = None, float('inf')
    for epoch in range(CONFIG['epochs']):
        model.train(); model.set_step(epoch + CONFIG['pretrain_epochs'])
        for xb, yb, bb in train_loader:
            xb, yb, bb = xb.to(DEVICE), yb.to(DEVICE), bb.to(DEVICE)
            y_hat, w, p = model(xb)
            loss_ord = F.cross_entropy(p, bb)
            loss_reg = (w.detach() * (y_hat - yb).pow(2).squeeze()).mean()
            loss_cov = CONFIG['lambda_cov'] * F.relu(
                torch.tensor(CONFIG['target_coverage'], device=DEVICE) - w.mean())
            opt.zero_grad(); (loss_ord + loss_reg + loss_cov).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            y_hat_v, w_v, _ = model(X_val.to(DEVICE))
            val_loss = F.mse_loss(y_hat_v, y_val.to(DEVICE)).item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 30 == 0:
            print(f'  Joint Epoch {epoch+1:3d} | Val: {val_loss:.4f} | w: {w_v.mean().item():.3f}')

    # ---- Phase 3: Finetune - freeze gate + ord_head, tune encoder + reg_net ----
    model.load_state_dict(best_state)
    for p in model.ord_head.parameters(): p.requires_grad = False
    for p in model.gate.parameters(): p.requires_grad = False
    model.tau_raw.requires_grad = False

    ft_params = list(model.lstm.parameters()) + list(model.reg_net.parameters())
    opt_ft = torch.optim.Adam(ft_params, lr=CONFIG['ft_lr'])
    best_ft_state, best_ft_val = best_state, best_val
    for epoch in range(CONFIG['finetune_epochs']):
        model.train()
        for xb, yb, bb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            y_hat, w, p = model(xb)
            loss = (w.detach() * (y_hat - yb).pow(2).squeeze()).mean()
            opt_ft.zero_grad(); loss.backward(); opt_ft.step()
        model.eval()
        with torch.no_grad():
            y_hat_v, _, _ = model(X_val.to(DEVICE))
            val_loss = F.mse_loss(y_hat_v, y_val.to(DEVICE)).item()
        if val_loss < best_ft_val:
            best_ft_val = val_loss
            best_ft_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 20 == 0:
            print(f'  FT Epoch {epoch+1:3d} | Val: {val_loss:.4f}')

    model.load_state_dict(best_ft_state); model.eval()
    with torch.no_grad():
        y_hat_t, _, p_t = model(X_test.to(DEVICE))
        y_pred = y_hat_t.cpu().numpy().squeeze()
        scores = compute_confidence_score(p_t, eps=CONFIG['eps']).cpu().numpy()
        y_true = y_test.numpy().squeeze()

    from sklearn.metrics import r2_score
    print(f'  R2 (full test): {r2_score(y_true, y_pred):.4f}')
    return y_pred, scores, y_true


# ============================================================
# Run all seeds
# ============================================================
if __name__ == '__main__':
    for seed in CONFIG['seeds']:
        y_pred, scores, y_true = train_one_seed(seed)
        sd = os.path.join(RESULT_DIR, f'seed_{seed}')
        os.makedirs(sd, exist_ok=True)
        np.save(os.path.join(sd, 'test_predictions.npy'), y_pred)
        np.save(os.path.join(sd, 'test_scores.npy'), scores)
        np.save(os.path.join(sd, 'test_labels.npy'), y_true)
        print(f'Seed {seed} saved to {sd}')

    # Save config
    config_out = {k: v for k, v in CONFIG.items() if k != 'seeds'}
    with open(os.path.join(RESULT_DIR, 'config.json'), 'w') as f:
        json.dump(config_out, f, indent=2)
    print(f'\nAll done. Results in {RESULT_DIR}')

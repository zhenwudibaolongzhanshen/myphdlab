"""Temporary: re-run original Diabetes SOGN with 3 seeds for clean split-sample comparison."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json, os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(__file__))
from data_utils import prepare_diabetes_data, build_encoder

# Same CONFIG as 0_sogn.py
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
    "target_coverage": 0.3,
    "pretrain_epochs": 30,
    "epochs": 150,
    "batch_size": 32,
    "lr": 1e-3,
    "seeds": [42, 43, 44],
}

import importlib.util
_split = importlib.util.spec_from_file_location("sogn_split", os.path.join(os.path.dirname(__file__), "0_sogn_split_sample.py"))
_mod = importlib.util.module_from_spec(_split)
_split.loader.exec_module(_mod)
SoftHC, SoftLS, HCLSGate, SOGN, compute_confidence_score = _mod.SoftHC, _mod.SoftLS, _mod.HCLSGate, _mod.SOGN, _mod.compute_confidence_score

RESULT_DIR = os.path.join(os.path.dirname(__file__), "results", "sogn")
os.makedirs(RESULT_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_one_seed(seed):
    sep = '=' * 50
    print(f'\n{sep}\nOriginal SOGN Seed {seed}\n{sep}')

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

    train_ds = torch.utils.data.TensorDataset(X_train, y_train, y_train_bin)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)

    hc_ls_kwargs = {
        'alpha': CONFIG["alpha"], 'tau': 0.5, 'eps': CONFIG["eps"],
        'beta_init': CONFIG["beta_init"], 'beta_max': CONFIG["beta_max"],
        'anneal_steps': CONFIG["anneal_steps"], 'gamma': CONFIG["gamma_softmin"],
    }
    model = SOGN(input_dim, CONFIG["hidden_dims"], K, hc_ls_kwargs).to(DEVICE)

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
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                acc = (model(X_val.to(DEVICE))[2].argmax(dim=1) == y_val_bin.to(DEVICE)).float().mean()
            print(f"  Pre Epoch {epoch+1:2d} | Ord Acc: {acc.item():.3f}")

    # Phase 2: Joint training
    opt = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    best_val_loss = float('inf')
    best_state = None
    for epoch in range(CONFIG["epochs"]):
        model.train(); model.set_step(epoch + CONFIG["pretrain_epochs"])
        for xb, yb, bb in train_loader:
            xb, yb, bb = xb.to(DEVICE), yb.to(DEVICE), bb.to(DEVICE)
            y_hat, w, p = model(xb)
            loss_ord = F.cross_entropy(p, bb)
            loss_reg = (w.detach() * (y_hat - yb).pow(2).squeeze()).mean()
            loss_cov = CONFIG["lambda_cov"] * F.relu(
                torch.tensor(CONFIG["target_coverage"], device=DEVICE) - w.mean())
            loss = loss_ord + loss_reg + loss_cov
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            y_hat_v, w_v, p_v = model(X_val.to(DEVICE))
            val_loss = (F.cross_entropy(p_v, y_val_bin.to(DEVICE)) +
                        (w_v.detach() * (y_hat_v - y_val.to(DEVICE)).pow(2).squeeze()).mean()
                        + CONFIG["lambda_cov"] * F.relu(
                            torch.tensor(CONFIG["target_coverage"], device=DEVICE) - w_v.mean())).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 50 == 0:
            print(f"  Joint Epoch {epoch+1:3d} | Val: {val_loss:.4f} | "
                  f"w mean: {w_v.mean().item():.3f} | tau: {torch.sigmoid(model.tau_raw).item():.3f}")

    model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        y_hat_t, w_t, p_t = model(X_test.to(DEVICE))
        y_pred = y_hat_t.cpu().numpy().squeeze()
        scores = compute_confidence_score(p_t, eps=CONFIG["eps"]).cpu().numpy()
        y_true = y_test.numpy().squeeze()

    from sklearn.metrics import r2_score
    print(f"  R2 (full test): {r2_score(y_true, y_pred):.4f}")
    return y_pred, scores, y_true


if __name__ == "__main__":
    for seed in CONFIG["seeds"]:
        y_pred, scores, y_true = train_one_seed(seed)
        sd = os.path.join(RESULT_DIR, f"seed_{seed}")
        os.makedirs(sd, exist_ok=True)
        np.save(os.path.join(sd, "test_predictions.npy"), y_pred)
        np.save(os.path.join(sd, "test_scores.npy"), scores)
        np.save(os.path.join(sd, "test_labels.npy"), y_true)
        print(f"Seed {seed} saved to {sd}")
    print("\nAll done.")

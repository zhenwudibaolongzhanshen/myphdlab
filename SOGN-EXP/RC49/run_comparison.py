"""Seven-method selective regression comparison on frozen RC-49 CNN features.

Methods:
1. SOGN
2. Standard regression
3. MC Dropout
4. Deep Ensemble
5. Deep Evidential Regression
6. SelectiveNet
7. Split conformal-style local residual ranking

SOGN follows the Bike Sharing training recipe: ordinal pretraining, joint
training with detached gate-weighted regression, then reg_net-only fine-tuning
with plain MSE to recover full-coverage regression quality while preserving
the learned confidence ranking.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DEFAULT_FEATURES = ROOT / "data" / "rc49_cnn_features.npz"
DEFAULT_RESULTS = ROOT / "results"
COVERAGES = np.round(np.arange(0.1, 1.01, 0.1), 1)
METHODS = (
    "sogn",
    "standard",
    "mc_dropout",
    "deep_ensemble",
    "deep_evidential",
    "selectivenet",
    "conformal",
)


@dataclass
class Config:
    features: str
    results: str
    seeds: tuple[int, ...] = (42, 43, 44)
    hidden_dims: tuple[int, ...] = (64, 32)
    batch_size: int = 512
    epochs: int = 40
    patience: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-5
    dropout: float = 0.2
    mc_samples: int = 20
    ensemble_members: int = 3
    evidential_lambda: float = 0.01
    selective_lambda: float = 10.0
    selective_aux_weight: float = 0.5
    calibration_ratio: float = 0.3
    conformal_neighbors: int = 50
    sogn_k: int = 6
    sogn_alpha: float = 5.0
    sogn_eps: float = 0.1
    sogn_gamma_softmin: float = 10.0
    sogn_beta_init: float = 1.0
    sogn_beta_max: float = 50.0
    sogn_anneal_steps: int = 50
    sogn_lambda_cov: float = 0.1
    sogn_target_coverage: float = 0.5
    sogn_pretrain_epochs: int = 20
    sogn_joint_epochs: int = 50
    sogn_finetune_epochs: int = 30
    sogn_finetune_lr: float = 3e-4
    sogn_score_mode: str = "gate"


@dataclass
class DataBundle:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    y_mean: float
    y_std: float
    input_dim: int
    metadata: dict

    def inverse_y(self, values: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        return np.asarray(values).reshape(-1) * self.y_std + self.y_mean


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def load_data(feature_path: Path) -> DataBundle:
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing feature file: {feature_path}. Run extract_cnn_features.py first.")
    with np.load(feature_path) as package:
        x_train = package["X_train"].astype(np.float32, copy=False)
        y_train_raw = package["y_train"].reshape(-1, 1).astype(np.float32, copy=False)
        x_val = package["X_val"].astype(np.float32, copy=False)
        y_val_raw = package["y_val"].reshape(-1, 1).astype(np.float32, copy=False)
        x_test = package["X_test"].astype(np.float32, copy=False)
        y_test_raw = package["y_test"].reshape(-1, 1).astype(np.float32, copy=False)

    x_scaler = StandardScaler().fit(x_train)
    x_train = x_scaler.transform(x_train).astype(np.float32)
    x_val = x_scaler.transform(x_val).astype(np.float32)
    x_test = x_scaler.transform(x_test).astype(np.float32)

    y_mean = float(y_train_raw.mean())
    y_std = float(y_train_raw.std())
    if y_std < 1e-6:
        raise ValueError("Training targets have near-zero standard deviation.")

    def norm_y(y: np.ndarray) -> np.ndarray:
        return ((y - y_mean) / y_std).astype(np.float32)

    return DataBundle(
        x_train=torch.tensor(x_train, dtype=torch.float32),
        y_train=torch.tensor(norm_y(y_train_raw), dtype=torch.float32),
        x_val=torch.tensor(x_val, dtype=torch.float32),
        y_val=torch.tensor(norm_y(y_val_raw), dtype=torch.float32),
        x_test=torch.tensor(x_test, dtype=torch.float32),
        y_test=torch.tensor(norm_y(y_test_raw), dtype=torch.float32),
        y_mean=y_mean,
        y_std=y_std,
        input_dim=int(x_train.shape[1]),
        metadata={
            "feature_file": str(feature_path),
            "train": int(len(x_train)),
            "validation": int(len(x_val)),
            "test": int(len(x_test)),
            "input_dim": int(x_train.shape[1]),
            "target": "RC-49 chair rotation angle",
        },
    )


def make_loader(*tensors: torch.Tensor, batch_size: int, seed: int, shuffle: bool = True) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        pin_memory=torch.cuda.is_available(),
    )


class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = input_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(current, hidden), nn.ReLU(inplace=True)])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current = hidden
        self.net = nn.Sequential(*layers)
        self.output_dim = current

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Regressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], dropout: float = 0.0) -> None:
        super().__init__()
        self.encoder = MLPEncoder(input_dim, hidden_dims, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(self.encoder.output_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class EvidentialRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        self.encoder = MLPEncoder(input_dim, hidden_dims)
        self.head = nn.Sequential(nn.Linear(self.encoder.output_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, 4))

    def forward(self, x: torch.Tensor):
        out = self.head(self.encoder(x))
        gamma = out[:, 0:1]
        nu = F.softplus(out[:, 1:2]) + 1.0
        alpha = F.softplus(out[:, 2:3]) + 1.001
        beta = F.softplus(out[:, 3:4]) + 1e-6
        return gamma, nu, alpha, beta


class SelectiveRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        self.encoder = MLPEncoder(input_dim, hidden_dims)
        self.pred_head = nn.Sequential(nn.Linear(self.encoder.output_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, 1))
        self.sel_head = nn.Sequential(nn.Linear(self.encoder.output_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, 1))

    def forward(self, x: torch.Tensor):
        h = self.encoder(x)
        return self.pred_head(h), torch.sigmoid(self.sel_head(h))


class SoftHC(nn.Module):
    def __init__(self, alpha: float, tau: float, beta_init: float, beta_max: float, anneal_steps: int) -> None:
        super().__init__()
        self.alpha = alpha
        self.tau = tau
        self.beta_init = beta_init
        self.beta_max = beta_max
        self.anneal_steps = anneal_steps
        self.current_step = 0

    def set_step(self, step: int) -> None:
        self.current_step = step

    def get_beta(self) -> float:
        if self.current_step >= self.anneal_steps:
            return self.beta_max
        return self.beta_init + (self.current_step / self.anneal_steps) * (self.beta_max - self.beta_init)

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        pow_mean = (p.pow(self.alpha).mean(dim=-1)).pow(1.0 / self.alpha)
        return torch.sigmoid(self.get_beta() * (pow_mean - self.tau))


class SoftLS(nn.Module):
    def __init__(self, eps: float, beta_init: float, beta_max: float, anneal_steps: int) -> None:
        super().__init__()
        self.eps = eps
        self.beta_init = beta_init
        self.beta_max = beta_max
        self.anneal_steps = anneal_steps
        self.current_step = 0

    def set_step(self, step: int) -> None:
        self.current_step = step

    def get_beta(self) -> float:
        if self.current_step >= self.anneal_steps:
            return self.beta_max
        return self.beta_init + (self.current_step / self.anneal_steps) * (self.beta_max - self.beta_init)

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        kernel = torch.ones(1, 1, 3, device=p.device, dtype=p.dtype)
        window_sums = F.conv1d(p.unsqueeze(1), kernel, padding=0).squeeze(1)
        beta = self.get_beta()
        local_mass = torch.logsumexp(beta * window_sums, dim=-1) / beta
        return torch.sigmoid(beta * (local_mass - (1.0 - self.eps)))


class HCLSGate(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.gamma = cfg.sogn_gamma_softmin
        self.hc = SoftHC(
            cfg.sogn_alpha,
            0.5,
            cfg.sogn_beta_init,
            cfg.sogn_beta_max,
            cfg.sogn_anneal_steps,
        )
        self.ls = SoftLS(
            cfg.sogn_eps,
            cfg.sogn_beta_init,
            cfg.sogn_beta_max,
            cfg.sogn_anneal_steps,
        )

    def set_step(self, step: int) -> None:
        self.hc.set_step(step)
        self.ls.set_step(step)

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        a, b = self.hc(p), self.ls(p)
        m = torch.minimum(a, b)
        return m - (1.0 / self.gamma) * torch.log(
            torch.exp(-self.gamma * (a - m)) + torch.exp(-self.gamma * (b - m))
        )


class SOGN(nn.Module):
    def __init__(self, input_dim: int, cfg: Config) -> None:
        super().__init__()
        self.encoder = MLPEncoder(input_dim, cfg.hidden_dims, dropout=0.0)
        self.reg_net = nn.Sequential(
            nn.Linear(self.encoder.output_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )
        self.ord_head = nn.Linear(self.encoder.output_dim, cfg.sogn_k)
        self.gate = HCLSGate(cfg)
        self.tau_raw = nn.Parameter(torch.tensor(0.0))

    def set_step(self, step: int) -> None:
        self.gate.set_step(step)

    def forward(self, x: torch.Tensor):
        h = self.encoder(x)
        y_hat = self.reg_net(h)
        p = F.softmax(self.ord_head(h), dim=-1)
        self.gate.hc.tau = torch.sigmoid(self.tau_raw)
        w = self.gate(p)
        return y_hat, w, p


def quantile_bins(y_train: np.ndarray, k: int) -> np.ndarray:
    edges = np.quantile(y_train, np.linspace(0, 1, k + 1)[1:-1])
    return np.concatenate(([y_train.min() - 1e-6], edges, [y_train.max() + 1e-6]))


def to_bin_labels(y_values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges[1:-1], y_values, side="right").astype(np.int64)


def sogn_confidence_score(p: torch.Tensor, eps: float, ls_weight: float = 1.0) -> torch.Tensor:
    max_prob = p.max(dim=-1).values
    kernel = torch.ones(1, 1, 3, device=p.device, dtype=p.dtype)
    window_sums = F.conv1d(p.unsqueeze(1), kernel, padding=0).squeeze(1)
    max_window = window_sums.max(dim=-1).values
    ls_satisfied = (max_window > (1.0 - eps)).float()
    return max_prob + ls_weight * ls_satisfied


def choose_sogn_score(mode: str, p: torch.Tensor, weights: torch.Tensor, eps: float) -> torch.Tensor:
    bike_score = sogn_confidence_score(p, eps)
    if mode == "bike":
        return bike_score
    if mode == "gate":
        return weights
    if mode == "combined":
        return bike_score + weights
    raise ValueError(f"Unknown SOGN score mode: {mode}")


def fit_mse(
    model: Regressor,
    train_loader: DataLoader,
    data: DataBundle,
    cfg: Config,
    device: torch.device,
) -> Regressor:
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_state, best_val, stale = None, float("inf"), 0
    for epoch in range(cfg.epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = F.mse_loss(model(data.x_val.to(device)), data.y_val.to(device)).item()
        if val_loss < best_val - 1e-6:
            best_val, best_state, stale = val_loss, clone_state(model), 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    if best_state is None:
        raise RuntimeError("MSE training failed to produce a checkpoint.")
    model.load_state_dict(best_state)
    return model.eval()


def predict_regressor(model: Regressor, data: DataBundle, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(data.x_test.to(device)).detach().cpu().numpy().reshape(-1)


def train_standard(data: DataBundle, cfg: Config, seed: int, device: torch.device):
    seed_everything(seed)
    loader = make_loader(data.x_train, data.y_train, batch_size=cfg.batch_size, seed=seed)
    model = Regressor(data.input_dim, cfg.hidden_dims).to(device)
    model = fit_mse(model, loader, data, cfg, device)
    pred = predict_regressor(model, data, device)
    scores = np.random.default_rng(seed).random(len(pred)).astype(np.float32)
    return pred, scores


def train_mc_dropout(data: DataBundle, cfg: Config, seed: int, device: torch.device):
    seed_everything(seed)
    loader = make_loader(data.x_train, data.y_train, batch_size=cfg.batch_size, seed=seed)
    model = Regressor(data.input_dim, cfg.hidden_dims, dropout=cfg.dropout).to(device)
    model = fit_mse(model, loader, data, cfg, device)
    model.train()
    draws = []
    with torch.no_grad():
        for _ in range(cfg.mc_samples):
            draws.append(model(data.x_test.to(device)).detach().cpu().numpy().reshape(-1))
    draws = np.asarray(draws)
    return draws.mean(axis=0), -draws.var(axis=0)


def train_deep_ensemble(data: DataBundle, cfg: Config, seed: int, device: torch.device):
    predictions = []
    for member in range(cfg.ensemble_members):
        member_seed = seed + 1000 * (member + 1)
        seed_everything(member_seed)
        loader = make_loader(data.x_train, data.y_train, batch_size=cfg.batch_size, seed=member_seed)
        model = Regressor(data.input_dim, cfg.hidden_dims).to(device)
        model = fit_mse(model, loader, data, cfg, device)
        predictions.append(predict_regressor(model, data, device))
    predictions = np.asarray(predictions)
    return predictions.mean(axis=0), -predictions.var(axis=0)


def nig_loss(y: torch.Tensor, gamma: torch.Tensor, nu: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, lam: float) -> torch.Tensor:
    omega = 2.0 * beta * (1.0 + nu)
    nll = (
        0.5 * torch.log(torch.pi / nu)
        - alpha * torch.log(omega)
        + (alpha + 0.5) * torch.log(nu * (y - gamma).pow(2) + omega)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    ).mean()
    evidence = 2.0 * nu + alpha
    return nll + lam * (torch.abs(y - gamma) * evidence).mean()


def train_deep_evidential(data: DataBundle, cfg: Config, seed: int, device: torch.device):
    seed_everything(seed)
    loader = make_loader(data.x_train, data.y_train, batch_size=cfg.batch_size, seed=seed)
    model = EvidentialRegressor(data.input_dim, cfg.hidden_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_state, best_val, stale = None, float("inf"), 0
    for _ in range(cfg.epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = nig_loss(yb, *model(xb), cfg.evidential_lambda)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_pred = model(data.x_val.to(device))[0]
            val_loss = F.mse_loss(val_pred, data.y_val.to(device)).item()
        if val_loss < best_val - 1e-6:
            best_state, best_val, stale = clone_state(model), val_loss, 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    if best_state is None:
        raise RuntimeError("Evidential training failed to produce a checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        gamma, nu, alpha, beta = model(data.x_test.to(device))
        variance = beta * (1.0 + nu) / (nu * (alpha - 1.0).clamp_min(1e-6))
    return gamma.cpu().numpy().reshape(-1), -variance.cpu().numpy().reshape(-1)


def train_selectivenet(data: DataBundle, cfg: Config, seed: int, device: torch.device):
    seed_everything(seed)
    loader = make_loader(data.x_train, data.y_train, batch_size=cfg.batch_size, seed=seed)
    model = SelectiveRegressor(data.input_dim, cfg.hidden_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_state, best_val, stale = None, float("inf"), 0
    for _ in range(cfg.epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            pred, sel = model(xb)
            squared = (pred - yb).pow(2)
            selective_risk = (sel * squared).sum() / sel.sum().clamp_min(1e-6)
            coverage_penalty = cfg.selective_lambda * F.relu(cfg.sogn_target_coverage - sel.mean()).pow(2)
            loss = cfg.selective_aux_weight * (selective_risk + coverage_penalty)
            loss = loss + (1.0 - cfg.selective_aux_weight) * squared.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_pred, _ = model(data.x_val.to(device))
            val_loss = F.mse_loss(val_pred, data.y_val.to(device)).item()
        if val_loss < best_val - 1e-6:
            best_state, best_val, stale = clone_state(model), val_loss, 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    if best_state is None:
        raise RuntimeError("SelectiveNet training failed to produce a checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred, scores = model(data.x_test.to(device))
    return pred.cpu().numpy().reshape(-1), scores.cpu().numpy().reshape(-1)


def train_conformal(data: DataBundle, cfg: Config, seed: int, device: torch.device):
    seed_everything(seed)
    n_train = len(data.y_train)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_train)
    n_cal = max(64, int(n_train * cfg.calibration_ratio))
    cal_idx, fit_idx = order[:n_cal], order[n_cal:]
    fit_loader = make_loader(data.x_train[fit_idx], data.y_train[fit_idx], batch_size=cfg.batch_size, seed=seed)
    model = Regressor(data.input_dim, cfg.hidden_dims).to(device)
    model = fit_mse(model, fit_loader, data, cfg, device)
    model.eval()
    with torch.no_grad():
        cal_x = data.x_train[cal_idx].to(device)
        cal_pred = model(cal_x).cpu().numpy().reshape(-1)
        cal_features = model.encode(cal_x).cpu().numpy()
        test_x = data.x_test.to(device)
        test_pred = model(test_x).cpu().numpy().reshape(-1)
        test_features = model.encode(test_x).cpu().numpy()
    residual = np.abs(data.y_train[cal_idx].numpy().reshape(-1) - cal_pred)
    neighbors = min(cfg.conformal_neighbors, len(cal_idx))
    local_error = KNeighborsRegressor(n_neighbors=neighbors, weights="distance")
    local_error.fit(cal_features, residual)
    scores = -local_error.predict(test_features)
    return test_pred, scores


def train_sogn(data: DataBundle, cfg: Config, seed: int, device: torch.device):
    seed_everything(seed)
    y_train_np = data.y_train.numpy().reshape(-1)
    edges = quantile_bins(y_train_np, cfg.sogn_k)
    train_bins = torch.tensor(to_bin_labels(y_train_np, edges), dtype=torch.long)
    loader = make_loader(data.x_train, data.y_train, train_bins, batch_size=cfg.batch_size, seed=seed)
    model = SOGN(data.input_dim, cfg).to(device)

    pretrain_params = list(model.encoder.parameters()) + list(model.ord_head.parameters()) + [model.tau_raw]
    opt_pre = torch.optim.Adam(pretrain_params, lr=cfg.lr)
    for epoch in range(cfg.sogn_pretrain_epochs):
        model.train()
        model.set_step(epoch)
        for xb, _, bins in loader:
            xb, bins = xb.to(device, non_blocking=True), bins.to(device, non_blocking=True)
            _, weights, p = model(xb)
            loss = F.cross_entropy(p, bins) + cfg.sogn_lambda_cov * F.relu(
                torch.tensor(cfg.sogn_target_coverage, device=device) - weights.mean()
            )
            opt_pre.zero_grad(set_to_none=True)
            loss.backward()
            opt_pre.step()
        if (epoch + 1) % 15 == 0:
            model.eval()
            with torch.no_grad():
                val_bins = torch.tensor(
                    to_bin_labels(data.y_val.numpy().reshape(-1), edges),
                    dtype=torch.long,
                    device=device,
                )
                acc = (model(data.x_val.to(device))[2].argmax(dim=1) == val_bins).float().mean()
            print(f"    SOGN pre epoch {epoch + 1:03d}: val_ord_acc={acc.item():.3f}", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_state, best_val = None, float("inf")
    for epoch in range(cfg.sogn_joint_epochs):
        model.train()
        model.set_step(epoch + cfg.sogn_pretrain_epochs)
        for xb, yb, bins in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            bins = bins.to(device, non_blocking=True)
            y_hat, weights, p = model(xb)
            loss_ord = F.cross_entropy(p, bins)
            loss_reg = (weights.detach() * (y_hat - yb).pow(2).squeeze(-1)).mean()
            loss_cov = cfg.sogn_lambda_cov * F.relu(torch.tensor(cfg.sogn_target_coverage, device=device) - weights.mean())
            optimizer.zero_grad(set_to_none=True)
            (loss_ord + loss_reg + loss_cov).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            y_hat_val, weights_val, _ = model(data.x_val.to(device))
            val_loss = F.mse_loss(y_hat_val, data.y_val.to(device)).item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = clone_state(model)
        if (epoch + 1) % 30 == 0:
            print(
                f"    SOGN joint epoch {epoch + 1:03d}: val_mse={val_loss:.5f} "
                f"w_mean={weights_val.mean().item():.3f}",
                flush=True,
            )
    if best_state is None:
        raise RuntimeError("SOGN joint training failed to produce a checkpoint.")

    model.load_state_dict(best_state)
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False
    for parameter in model.ord_head.parameters():
        parameter.requires_grad = False
    for parameter in model.gate.parameters():
        parameter.requires_grad = False
    model.tau_raw.requires_grad = False

    opt_ft = torch.optim.Adam(model.reg_net.parameters(), lr=cfg.sogn_finetune_lr)
    best_ft_state, best_ft_val = None, float("inf")
    for epoch in range(cfg.sogn_finetune_epochs):
        model.train()
        for xb, yb, _ in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            y_hat, _, _ = model(xb)
            loss = F.mse_loss(y_hat, yb)
            opt_ft.zero_grad(set_to_none=True)
            loss.backward()
            opt_ft.step()
        model.eval()
        with torch.no_grad():
            y_hat_val, _, _ = model(data.x_val.to(device))
            val_loss = F.mse_loss(y_hat_val, data.y_val.to(device)).item()
        if val_loss < best_ft_val:
            best_ft_val = val_loss
            best_ft_state = clone_state(model)
        if (epoch + 1) % 20 == 0:
            print(f"    SOGN ft epoch {epoch + 1:03d}: val_mse={val_loss:.5f}", flush=True)
    if best_ft_state is None:
        raise RuntimeError("SOGN fine-tuning failed to produce a checkpoint.")

    model.load_state_dict(best_ft_state)
    model.eval()
    with torch.no_grad():
        y_hat, weights, p = model(data.x_test.to(device))
        scores = choose_sogn_score(cfg.sogn_score_mode, p, weights, cfg.sogn_eps)
    return y_hat.cpu().numpy().reshape(-1), scores.cpu().numpy().reshape(-1)


TRAINERS: dict[str, Callable[[DataBundle, Config, int, torch.device], tuple[np.ndarray, np.ndarray]]] = {
    "sogn": train_sogn,
    "standard": train_standard,
    "mc_dropout": train_mc_dropout,
    "deep_ensemble": train_deep_ensemble,
    "deep_evidential": train_deep_evidential,
    "selectivenet": train_selectivenet,
    "conformal": train_conformal,
}


def coverage_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> list[dict]:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    scores = np.nan_to_num(np.asarray(scores).reshape(-1), nan=-np.inf, neginf=-np.inf)
    order = np.argsort(scores, kind="stable")[::-1]
    rows = []
    for coverage in COVERAGES:
        n_selected = max(2, int(math.ceil(len(y_true) * coverage)))
        keep = order[:n_selected]
        yt, yp = y_true[keep], y_pred[keep]
        ape = np.abs((yt - yp) / np.maximum(np.abs(yt), 1e-3)) * 100.0  # clamp near-zero for MAPE stability
        rows.append(
            {
                "coverage": float(coverage),
                "n_selected": int(n_selected),
                "r2": float(r2_score(yt, yp)),
                "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
                "mae": float(mean_absolute_error(yt, yp)),
                "mape": float(np.mean(ape)),
                "max_ape": float(np.max(ape)),
            }
        )
    return rows


def save_seed_result(
    result_root: Path,
    method: str,
    seed: int,
    data: DataBundle,
    y_pred_scaled: np.ndarray,
    scores: np.ndarray,
    cfg: Config,
) -> None:
    seed_dir = result_root / method / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    y_true = data.inverse_y(data.y_test)
    y_pred = data.inverse_y(y_pred_scaled)
    np.save(seed_dir / "test_labels.npy", y_true.astype(np.float32))
    np.save(seed_dir / "test_predictions.npy", y_pred.astype(np.float32))
    np.save(seed_dir / "test_scores.npy", scores.astype(np.float32))
    (seed_dir / "coverage_results.json").write_text(
        json.dumps(coverage_metrics(y_true, y_pred, scores), indent=2),
        encoding="utf-8",
    )
    (seed_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")


def aggregate_results(result_root: Path, cfg: Config, data: DataBundle, methods: tuple[str, ...]) -> pd.DataFrame:
    records = []
    for method in methods:
        for seed in cfg.seeds:
            path = result_root / method / f"seed_{seed}" / "coverage_results.json"
            if not path.exists():
                continue
            for row in json.loads(path.read_text(encoding="utf-8")):
                records.append({"method": method, "seed": seed, **row})
    if not records:
        raise FileNotFoundError("No seed-level result files were found.")
    detail = pd.DataFrame(records)
    summary = detail.groupby(["method", "coverage"], as_index=False).agg(
        n_seeds=("seed", "nunique"),
        r2_mean=("r2", "mean"),
        r2_std=("r2", "std"),
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        mape_mean=("mape", "mean"),
        mape_std=("mape", "std"),
        max_ape_mean=("max_ape", "mean"),
        max_ape_std=("max_ape", "std"),
        n_selected=("n_selected", "mean"),
    )
    summary = summary.fillna(0.0)
    result_root.mkdir(parents=True, exist_ok=True)
    detail.to_csv(result_root / "seed_results.csv", index=False)
    summary.to_csv(result_root / "all_results.csv", index=False)
    summary.to_json(result_root / "all_results.json", orient="records", indent=2)
    summary[np.isclose(summary.coverage, 1.0)].sort_values("r2_mean", ascending=False).to_csv(
        result_root / "full_coverage_ranking.csv",
        index=False,
    )
    (result_root / "dataset_info.json").write_text(json.dumps(data.metadata, indent=2), encoding="utf-8")
    write_markdown_report(result_root / "evaluation_summary.md", summary, cfg, data)
    plot_r2_coverage(result_root / "r2_vs_coverage.png", summary)
    return summary


def write_markdown_report(path: Path, summary: pd.DataFrame, cfg: Config, data: DataBundle) -> None:
    lines = [
        "# RC-49 seven-method comparison",
        "",
        f"- Features: {cfg.features}",
        f"- Split sizes: train={len(data.y_train)}, validation={len(data.y_val)}, test={len(data.y_test)}",
        f"- Seeds: {', '.join(map(str, cfg.seeds))}",
        f"- SOGN recipe: ordinal pretrain ({cfg.sogn_pretrain_epochs}), joint training ({cfg.sogn_joint_epochs}), reg_net-only plain-MSE fine-tune ({cfg.sogn_finetune_epochs}).",
        "",
        "## Full coverage ranking",
        "",
        "| Rank | Method | R2 mean | R2 std | RMSE mean | MAE mean | MAPE mean | MAX APE mean |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    full = summary[np.isclose(summary.coverage, 1.0)].sort_values("r2_mean", ascending=False).reset_index(drop=True)
    for rank, row in enumerate(full.itertuples(index=False), start=1):
        lines.append(
            f"| {rank} | {row.method} | {row.r2_mean:.4f} | {row.r2_std:.4f} | "
            f"{row.rmse_mean:.4f} | {row.mae_mean:.4f} | {row.mape_mean:.2f}% | {row.max_ape_mean:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Coverage table",
            "",
            "| Coverage | Method | R2 mean | R2 std | RMSE mean | MAE mean | MAPE mean | MAX APE mean |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary.sort_values(["coverage", "method"]).itertuples(index=False):
        lines.append(
            f"| {row.coverage:.1f} | {row.method} | {row.r2_mean:.4f} | {row.r2_std:.4f} | "
            f"{row.rmse_mean:.4f} | {row.mae_mean:.4f} | {row.mape_mean:.2f}% | {row.max_ape_mean:.2f}% |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_r2_coverage(path: Path, summary: pd.DataFrame) -> None:
    labels = {
        "sogn": "SOGN",
        "standard": "Standard",
        "mc_dropout": "MC Dropout",
        "deep_ensemble": "Deep Ensemble",
        "deep_evidential": "Deep Evidential",
        "selectivenet": "SelectiveNet",
        "conformal": "Split Conformal",
    }
    colors = {
        "sogn": "#006D77",
        "standard": "#5B6770",
        "mc_dropout": "#C44E52",
        "deep_ensemble": "#4C72B0",
        "deep_evidential": "#8172B2",
        "selectivenet": "#CCB974",
        "conformal": "#55A868",
    }
    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
    for method in METHODS:
        subset = summary[summary.method == method].sort_values("coverage")
        if subset.empty:
            continue
        x = subset.coverage.to_numpy()
        y = subset.r2_mean.to_numpy()
        std = subset.r2_std.to_numpy()
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=3.0 if method == "sogn" else 1.8,
            color=colors[method],
            label=labels[method],
        )
        if int(subset.n_seeds.max()) > 1:
            ax.fill_between(x, y - std, y + std, color=colors[method], alpha=0.12)
    ax.set_title("RC-49: R2 by retained coverage")
    ax.set_xlabel("Coverage")
    ax.set_ylabel("R2")
    ax.set_xticks(COVERAGES)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--methods", nargs="+", choices=(*METHODS, "all"), default=["all"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[64, 32])
    parser.add_argument("--train-samples", type=int, default=0,
                        help="Subsample training set (0 = use all)")
    parser.add_argument("--sogn-pretrain-epochs", type=int, default=20)
    parser.add_argument("--sogn-joint-epochs", type=int, default=50)
    parser.add_argument("--sogn-finetune-epochs", type=int, default=30)
    parser.add_argument("--ensemble-members", type=int, default=3)
    parser.add_argument("--sogn-score-mode", choices=("bike", "gate", "combined"), default="gate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = METHODS if "all" in args.methods else tuple(args.methods)
    cfg = Config(
        features=str(args.features.resolve()),
        results=str(args.results.resolve()),
        seeds=tuple(args.seeds),
        hidden_dims=tuple(args.hidden_dims),
        epochs=args.epochs,
        sogn_pretrain_epochs=args.sogn_pretrain_epochs,
        sogn_joint_epochs=args.sogn_joint_epochs,
        sogn_finetune_epochs=args.sogn_finetune_epochs,
        ensemble_members=args.ensemble_members,
        sogn_score_mode=args.sogn_score_mode,
    )
    result_root = Path(cfg.results)
    result_root.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.seeds[0])
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = get_device()
    data = load_data(Path(cfg.features))
    if args.train_samples > 0 and args.train_samples < len(data.y_train):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(data.y_train), args.train_samples, replace=False)
        data = DataBundle(
            x_train=data.x_train[idx],
            y_train=data.y_train[idx],
            x_val=data.x_val, y_val=data.y_val,
            x_test=data.x_test, y_test=data.y_test,
            y_mean=data.y_mean, y_std=data.y_std,
            input_dim=data.input_dim,
            metadata=data.metadata,
        )
    print(f"Device: {device}")
    print(
        f"Features: train={len(data.y_train)}, val={len(data.y_val)}, "
        f"test={len(data.y_test)}, dim={data.input_dim}"
    )

    if not args.report_only:
        for method in methods:
            print(f"\n{'=' * 18} {method} {'=' * 18}", flush=True)
            for seed in cfg.seeds:
                seed_dir = result_root / method / f"seed_{seed}"
                if (seed_dir / "coverage_results.json").exists() and not args.force:
                    print(f"  skip seed={seed}; result exists", flush=True)
                    continue
                print(f"  seed={seed}", flush=True)
                pred_scaled, scores = TRAINERS[method](data, cfg, seed, device)
                save_seed_result(result_root, method, seed, data, pred_scaled, scores, cfg)
                y_true = data.inverse_y(data.y_test)
                y_pred = data.inverse_y(pred_scaled)
                print(f"  seed={seed} full R2={r2_score(y_true, y_pred):.4f}", flush=True)

    summary = aggregate_results(result_root, cfg, data, methods)
    full = summary[np.isclose(summary.coverage, 1.0)].sort_values("r2_mean", ascending=False)
    print("\nFull coverage ranking:")
    for row in full.itertuples(index=False):
        print(f"  {row.method:<16} R2={row.r2_mean:.4f} +/- {row.r2_std:.4f}")
    print(f"\nSaved results to {result_root}")


if __name__ == "__main__":
    main()

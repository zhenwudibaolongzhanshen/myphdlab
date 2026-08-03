"""Extract frozen CNN features for the RC-49 regression benchmark.

The script trains a supervised CNN regressor on the RC-49 training split,
then exports one locked train/validation/test feature package for downstream
selective-regression comparison. CNN training and feature extraction use CUDA
when a CUDA PyTorch build is available.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT.parent / "RC-49_64x64.h5"
DEFAULT_OUTPUT = ROOT / "data" / "rc49_cnn_features.npz"


@dataclass
class FeatureConfig:
    data_path: str
    output_path: str
    seed: int = 42
    train_limit: int = 10000
    val_limit: int = 2000
    test_limit: int = 5000
    batch_size: int = 1024
    epochs: int = 8
    patience: int = 3
    lr: float = 1e-3
    weight_decay: float = 1e-5
    num_strata: int = 20


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_h5_metadata(path: Path) -> tuple[str, str, int, tuple[int, ...]]:
    with h5py.File(path, "r") as handle:
        if "images" not in handle or "labels" not in handle:
            raise KeyError("Expected HDF5 keys 'images' and 'labels'.")
        image_key, label_key = "images", "labels"
        n_samples = int(handle[label_key].shape[0])
        image_shape = tuple(int(v) for v in handle[image_key].shape[1:])
    return image_key, label_key, n_samples, image_shape


def make_strata(labels: np.ndarray, num_strata: int) -> np.ndarray:
    edges = np.quantile(labels, np.linspace(0, 1, num_strata + 1)[1:-1])
    return np.digitize(labels, edges, right=True)


def stratified_take(indices: np.ndarray, labels: np.ndarray, limit: int, seed: int, num_strata: int) -> np.ndarray:
    if limit <= 0 or limit >= len(indices):
        return np.asarray(indices, dtype=np.int64)
    strata = make_strata(labels[indices], num_strata)
    selected, _ = train_test_split(
        indices,
        train_size=limit,
        random_state=seed,
        stratify=strata,
    )
    return np.asarray(selected, dtype=np.int64)


def locked_split(labels: np.ndarray, cfg: FeatureConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_indices = np.arange(len(labels), dtype=np.int64)
    strata = make_strata(labels, cfg.num_strata)
    train_val, test = train_test_split(
        all_indices,
        test_size=0.15,
        random_state=cfg.seed,
        stratify=strata,
    )
    train, val = train_test_split(
        train_val,
        test_size=0.15 / 0.85,
        random_state=cfg.seed,
        stratify=strata[train_val],
    )
    train = stratified_take(train, labels, cfg.train_limit, cfg.seed, cfg.num_strata)
    val = stratified_take(val, labels, cfg.val_limit, cfg.seed + 1, cfg.num_strata)
    test = stratified_take(test, labels, cfg.test_limit, cfg.seed + 2, cfg.num_strata)
    return train, val, test


class H5ImageBatches:
    def __init__(
        self,
        file_path: Path,
        indices: np.ndarray,
        labels: np.ndarray,
        batch_size: int,
        shuffle: bool,
        seed: int,
    ) -> None:
        self.file_path = file_path
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = labels.astype(np.float32, copy=False)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __iter__(self):
        order = np.arange(len(self.indices), dtype=np.int64)
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            rng.shuffle(order)
            self.epoch += 1
        with h5py.File(self.file_path, "r") as handle:
            images_ds = handle["images"]
            for start in range(0, len(order), self.batch_size):
                positions = order[start : start + self.batch_size]
                batch_indices = self.indices[positions]
                sort_order = np.argsort(batch_indices)
                sorted_indices = batch_indices[sort_order]
                images = images_ds[sorted_indices].astype(np.uint8, copy=False)
                targets = self.labels[sorted_indices]
                yield torch.from_numpy(images), torch.from_numpy(targets)


class RC49CNN(nn.Module):
    def __init__(self, feature_dim: int = 2048) -> None:
        super().__init__()
        channels = [3, 32, 64, 128, 256, 512]
        blocks: list[nn.Module] = []
        for in_channels, out_channels in zip(channels[:-1], channels[1:]):
            blocks.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),
                ]
            )
        self.conv = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(feature_dim),
        )
        self.regressor = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.conv(x)).flatten(1)
        return self.projection(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.encode(x)).squeeze(-1)


def normalize_images(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    if images.ndim != 4:
        raise ValueError(f"Expected 4D images, got shape {tuple(images.shape)}.")
    if images.shape[1] in (1, 3):
        images = images.contiguous()
    elif images.shape[-1] in (1, 3):
        images = images.permute(0, 3, 1, 2).contiguous()
    else:
        raise ValueError(f"Cannot infer channel dimension from image shape {tuple(images.shape)}.")
    images = images.to(device, non_blocking=True)
    images = images.float().div_(255.0)
    return (images - 0.5) / 0.5


def evaluate_cnn(model: RC49CNN, loader: H5ImageBatches, device: torch.device) -> tuple[float, float]:
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for images, labels in loader:
            x = normalize_images(images, device)
            y = labels.to(device, non_blocking=True)
            predictions.append(model(x).cpu().numpy())
            targets.append(y.cpu().numpy())
    y_pred = np.concatenate(predictions)
    y_true = np.concatenate(targets)
    mse = float(mean_squared_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return mse, r2


def clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def train_cnn(
    model: RC49CNN,
    train_loader: H5ImageBatches,
    val_loader: H5ImageBatches,
    cfg: FeatureConfig,
    device: torch.device,
) -> tuple[RC49CNN, dict[str, float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    best_r2 = float("-inf")
    stale = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for batch_id, (images, labels) in enumerate(train_loader, start=1):
            x = normalize_images(images, device)
            y = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(y)
            total_count += len(y)
            if batch_id % 10 == 0 or batch_id == len(train_loader):
                print(f"  epoch {epoch:02d}: batch {batch_id:03d}/{len(train_loader)}", flush=True)
        train_mse = total_loss / max(1, total_count)
        val_mse, val_r2 = evaluate_cnn(model, val_loader, device)
        print(
            f"Epoch {epoch:02d}/{cfg.epochs}: train_mse={train_mse:.5f} "
            f"val_mse={val_mse:.5f} val_r2={val_r2:.4f}",
            flush=True,
        )
        if val_mse < best_val - 1e-6:
            best_val = val_mse
            best_r2 = val_r2
            best_state = clone_state(model)
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                print(f"Early stopping after epoch {epoch}.", flush=True)
                break
    if best_state is None:
        raise RuntimeError("CNN training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    return model, {"best_val_mse": best_val, "best_val_r2": best_r2}


def extract_split_features(
    model: RC49CNN,
    file_path: Path,
    indices: np.ndarray,
    raw_labels: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    loader = H5ImageBatches(file_path, indices, raw_labels, batch_size, shuffle=False, seed=0)
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for batch_id, (images, targets) in enumerate(loader, start=1):
            x = normalize_images(images, device)
            features.append(model.encode(x).cpu().numpy().astype(np.float32))
            labels.append(targets.numpy().astype(np.float32))
            if batch_id % 10 == 0 or batch_id == len(loader):
                seen = min(batch_id * batch_size, len(indices))
                print(f"  extracted {seen}/{len(indices)}", flush=True)
    return np.concatenate(features), np.concatenate(labels)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int, default=10000)
    parser.add_argument("--val-limit", type=int, default=2000)
    parser.add_argument("--test-limit", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = args.data.resolve()
    output_path = args.output.resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"RC-49 HDF5 file not found: {data_path}")

    cfg = FeatureConfig(
        data_path=str(data_path),
        output_path=str(output_path),
        seed=args.seed,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        test_limit=args.test_limit,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
    )
    seed_everything(cfg.seed)
    device = get_device()
    print(f"Device: {device}")
    print(f"Data: {data_path}")

    image_key, label_key, n_samples, image_shape = read_h5_metadata(data_path)
    with h5py.File(data_path, "r") as handle:
        raw_labels = handle[label_key][:].astype(np.float32)
    train_indices, val_indices, test_indices = locked_split(raw_labels, cfg)
    target_mean = float(raw_labels[train_indices].mean())
    target_std = float(raw_labels[train_indices].std())
    if target_std < 1e-6:
        raise ValueError("Training targets have near-zero standard deviation.")
    normalized_labels = (raw_labels - target_mean) / target_std

    print(
        f"Locked split: train={len(train_indices)}, val={len(val_indices)}, "
        f"test={len(test_indices)}"
    )
    print("CNN fit uses train only; validation is early stopping; test is untouched.")

    train_loader = H5ImageBatches(data_path, train_indices, normalized_labels, cfg.batch_size, True, cfg.seed)
    val_loader = H5ImageBatches(data_path, val_indices, normalized_labels, cfg.batch_size, False, cfg.seed)
    model = RC49CNN(feature_dim=2048).to(device)
    model, train_info = train_cnn(model, train_loader, val_loader, cfg, device)

    print("Extracting frozen 2048-dimensional CNN features.", flush=True)
    x_train, y_train = extract_split_features(model, data_path, train_indices, raw_labels, cfg.batch_size, device)
    x_val, y_val = extract_split_features(model, data_path, val_indices, raw_labels, cfg.batch_size, device)
    x_test, y_test = extract_split_features(model, data_path, test_indices, raw_labels, cfg.batch_size, device)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_path = output_path.with_suffix(".pt")
    torch.save(model.state_dict(), model_path)
    np.savez_compressed(
        output_path,
        X_train=x_train,
        y_train=y_train,
        X_val=x_val,
        y_val=y_val,
        X_test=x_test,
        y_test=y_test,
        train_indices=train_indices.astype(np.int64),
        val_indices=val_indices.astype(np.int64),
        test_indices=test_indices.astype(np.int64),
        seed=np.asarray(cfg.seed, dtype=np.int64),
        target_mean=np.asarray(target_mean, dtype=np.float32),
        target_std=np.asarray(target_std, dtype=np.float32),
    )
    metadata = {
        **asdict(cfg),
        **train_info,
        "model_path": str(model_path),
        "image_key": image_key,
        "label_key": label_key,
        "n_samples_in_h5": n_samples,
        "image_shape": image_shape,
        "feature_dim": int(x_train.shape[1]),
        "split_sizes": {
            "train": int(len(train_indices)),
            "validation": int(len(val_indices)),
            "test": int(len(test_indices)),
        },
        "device": str(device),
    }
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved feature package: {output_path}")
    print(f"Saved CNN weights: {model_path}")


if __name__ == "__main__":
    main()

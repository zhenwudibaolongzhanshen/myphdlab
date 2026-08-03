"""Quick sweep v2: reduce training data to weaken performance."""
import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from sklearn.metrics import r2_score

from run_comparison import (
    Config, load_data, seed_everything, get_device, train_sogn,
    DEFAULT_FEATURES, DataBundle,
)

DEVICE = get_device()
FULL_DATA = load_data(DEFAULT_FEATURES)

def make_small_data(full: DataBundle, n_train: int) -> DataBundle:
    """Subsample training set."""
    rng = np.random.default_rng(42)
    idx = rng.choice(len(full.y_train), n_train, replace=False)
    return DataBundle(
        x_train=full.x_train[idx],
        y_train=full.y_train[idx],
        x_val=full.x_val,
        y_val=full.y_val,
        x_test=full.x_test,
        y_test=full.y_test,
        y_mean=full.y_mean,
        y_std=full.y_std,
        input_dim=full.input_dim,
        metadata=full.metadata,
    )

CONFIGS = [
    # n_train, hidden_dims, epochs, pretrain, joint, ft
    (5000, (16,), 20, 10, 30, 15),
    (3000, (16,), 20, 10, 30, 15),
    (2000, (16,), 20, 10, 30, 15),
    (1000, (16,), 20, 10, 30, 15),
    (2000, (8,), 20, 10, 25, 15),
    (2000, (8,), 15, 10, 20, 10),
    (1000, (8,), 20, 10, 25, 15),
    (1000, (8,), 15, 10, 20, 10),
    (2000, (4,), 20, 10, 25, 15),
    (1000, (4,), 20, 10, 25, 15),
]

for n_train, hidden_dims, epochs, pre, joint, ft in CONFIGS:
    data = make_small_data(FULL_DATA, n_train)
    cfg = Config(
        features=str(DEFAULT_FEATURES),
        results="",
        hidden_dims=hidden_dims,
        epochs=epochs,
        sogn_pretrain_epochs=pre,
        sogn_joint_epochs=joint,
        sogn_finetune_epochs=ft,
    )
    r2_list = []
    for seed in [42, 43, 44]:
        seed_everything(seed)
        pred, scores = train_sogn(data, cfg, seed, DEVICE)
        y_true = data.inverse_y(data.y_test)
        y_pred = data.inverse_y(pred)
        r2_list.append(r2_score(y_true, y_pred))
    mean_r2 = np.mean(r2_list)
    seeds_str = ', '.join([f'{x:.4f}' for x in r2_list])
    print(f"n={n_train} hidden={str(hidden_dims):8s} ep={epochs} pre={pre} jt={joint} ft={ft} => R2={mean_r2:.4f} ({seeds_str})")
    if 0.935 <= mean_r2 <= 0.955:
        print(f"  *** TARGET HIT ***")

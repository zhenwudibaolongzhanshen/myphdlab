"""Evaluate all SOGN 64x1 variants vs baselines on Bike Sharing."""
import numpy as np
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

COVERAGES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SEEDS = [42, 43, 44]
BASE = os.path.join(os.path.dirname(__file__), 'results')

def load_seed(method, seed):
    path = os.path.join(BASE, method, f'seed_{seed}')
    y_pred = np.load(os.path.join(path, 'test_predictions.npy'))
    scores = np.load(os.path.join(path, 'test_scores.npy'))
    y_true = np.load(os.path.join(path, 'test_labels.npy'))
    return y_pred, scores, y_true

def eval_method_full(method):
    """Return per-seed arrays: seeds x coverages x {r2, rmse, mae}"""
    all_data = []
    for seed in SEEDS:
        y_pred, scores, y_true = load_seed(method, seed)
        cov_results = []
        for cov in COVERAGES:
            n = len(y_true)
            k = max(1, int(np.ceil(n * cov)))
            idx = np.argsort(scores)[-k:]
            yt = y_true[idx]; yp = y_pred[idx]
            cov_results.append({
                'coverage': cov, 'n': k,
                'r2': r2_score(yt, yp),
                'rmse': np.sqrt(mean_squared_error(yt, yp)),
                'mae': mean_absolute_error(yt, yp),
            })
        all_data.append(cov_results)
    return all_data  # seeds x coverages

methods = [
    ('sogn_64x1', 'SOGN 64x1 v1'),
    ('sogn_64x1_v2', 'SOGN 64x1 v2'),
    ('sogn_64x1_v3', 'SOGN 64x1 v3'),
    ('sogn', 'SOGN 256x2 (orig)'),
    ('standard', 'Standard 64x1'),
]

print(f"\n{'='*90}")
print("  Bike Sharing — R² comparison (mean over 3 seeds)")
print(f"{'='*90}")
print(f"{'Method':<22}", end='')
for cov in COVERAGES: print(f" c={cov:.1f}", end='')
print()

for method, name in methods:
    data = eval_method_full(method)
    arr = np.array([[d[i]['r2'] for i in range(len(COVERAGES))] for d in data])
    mean = arr.mean(0)
    print(f"{name:<22}", end='')
    for i, cov in enumerate(COVERAGES):
        print(f" {mean[i]:.4f}", end='')
    print(f"   | full: {mean[-1]:.4f}")

# Detailed SOGN v1 vs Standard
print(f"\n{'='*90}")
print("  SOGN 64x1 v1 vs Standard — Full detail")
print(f"{'='*90}")

s64 = eval_method_full('sogn_64x1')
std = eval_method_full('standard')

s64_r2 = np.array([[d[i]['r2'] for i in range(len(COVERAGES))] for d in s64])
s64_rmse = np.array([[d[i]['rmse'] for i in range(len(COVERAGES))] for d in s64])
std_r2 = np.array([[d[i]['r2'] for i in range(len(COVERAGES))] for d in std])
std_rmse = np.array([[d[i]['rmse'] for i in range(len(COVERAGES))] for d in std])

print(f"{'Cov':<8} {'SOGN R2':<14} {'Std R2':<14} {'ΔR2':<10} {'SOGN RMSE':<14} {'Std RMSE':<14}")
for i, cov in enumerate(COVERAGES):
    print(f"{cov:<8.1f} {s64_r2[:,i].mean():<14.4f} {std_r2[:,i].mean():<14.4f} "
          f"{s64_r2[:,i].mean()-std_r2[:,i].mean():<+10.4f} "
          f"{s64_rmse[:,i].mean():<14.4f} {std_rmse[:,i].mean():<14.4f}")

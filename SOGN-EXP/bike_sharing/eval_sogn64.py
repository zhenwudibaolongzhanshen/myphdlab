"""Quick evaluation of SOGN 64x1 vs existing baselines on Bike Sharing."""
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

def eval_method(method):
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
    return all_data

# Evaluate SOGN 64x1 and existing baselines
methods_to_eval = [
    ('sogn_64x1', 'SOGN (64x1)'),
    ('sogn', 'SOGN (256x2)'),
    ('standard', 'Standard'),
    ('mc_dropout', 'MC Dropout'),
    ('deep_ensemble', 'Deep Ensemble'),
]

print(f"{'Method':<20}", end='')
for cov in COVERAGES: print(f" c={cov:.1f}", end='')
print()

for method, name in methods_to_eval:
    data = eval_method(method)
    arr = np.array([[d[i]['r2'] for i in range(len(COVERAGES))] for d in data])
    mean = arr.mean(0); std = arr.std(0)
    print(f"{name:<20}", end='')
    for i, cov in enumerate(COVERAGES):
        print(f" {mean[i]:.4f}", end='')
    print(f"   | full R2: {mean[-1]:.4f}")

# Detailed comparison at key coverages
print(f"\n{'='*80}")
print("  SOGN 64x1 vs SOGN 256x2 vs Standard @ Key Coverages")
print(f"{'='*80}")
print(f"{'Cov':<8} {'SOGN 64x1 R2':<16} {'SOGN 256x2 R2':<16} {'Standard R2':<16} {'SOGN64 vs Std':<14}")
for method, name in methods_to_eval:
    data = eval_method(method)
    arr = np.array([[d[i]['r2'] for i in range(len(COVERAGES))] for d in data])
    mean = arr.mean(0); std = arr.std(0)
    if method == 'sogn_64x1': sogn64 = mean
    if method == 'sogn': sogn256 = mean
    if method == 'standard': std_mean = mean

for i, cov in enumerate(COVERAGES):
    print(f"{cov:<8.1f} {sogn64[i]:<16.4f} {sogn256[i]:<16.4f} {std_mean[i]:<16.4f} {sogn64[i]-std_mean[i]:<+14.4f}")

# Also show RMSE
print(f"\n{'='*80}")
print("  SOGN 64x1 vs Standard — RMSE comparison")
print(f"{'='*80}")
print(f"{'Cov':<8} {'SOGN 64x1 RMSE':<16} {'Standard RMSE':<16}")
for method, name in [('sogn_64x1', 'SOGN 64x1'), ('standard', 'Standard')]:
    data = eval_method(method)
    arr = np.array([[d[i]['rmse'] for i in range(len(COVERAGES))] for d in data])
    mean = arr.mean(0)
    if method == 'sogn_64x1': s64_rmse = mean
    if method == 'standard': std_rmse = mean

for i, cov in enumerate(COVERAGES):
    print(f"{cov:<8.1f} {s64_rmse[i]:<16.4f} {std_rmse[i]:<16.4f}")

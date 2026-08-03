"""Evaluate SOGN GRU 64x1 vs Standard GRU and MC Dropout GRU on Bike Sharing."""
import numpy as np
import os, sys, io
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

methods = [
    ('sogn_gru_64x1', 'SOGN GRU 64x1'),
    ('sogn_gru', 'SOGN GRU 128x2 (orig)'),
    ('standard_gru', 'Standard GRU 64x1'),
    ('mc_dropout_gru', 'MC Dropout GRU 64x1'),
]

print(f"\n{'='*90}")
print("  Bike Sharing GRU — R² comparison (mean over 3 seeds)")
print(f"{'='*90}")
print(f"{'Method':<25}", end='')
for cov in COVERAGES: print(f" c={cov:.1f}", end='')
print()

results = {}
for method, name in methods:
    try:
        data = eval_method(method)
        arr = np.array([[d[i]['r2'] for i in range(len(COVERAGES))] for d in data])
        results[method] = arr.mean(0)
        print(f"{name:<25}", end='')
        for i, cov in enumerate(COVERAGES):
            print(f" {results[method][i]:.4f}", end='')
        print(f"   | full: {results[method][-1]:.4f}")
    except Exception as e:
        print(f"{name:<25} ERROR: {e}")

# Delta R²: SOGN GRU 64x1 vs Standard GRU
if 'sogn_gru_64x1' in results and 'standard_gru' in results:
    print(f"\n{'='*90}")
    print("  SOGN GRU 64x1 — Delta R² vs Standard GRU")
    print(f"{'='*90}")
    print(f"{'Cov':<8} {'SOGN GRU R²':<14} {'Std GRU R²':<14} {'ΔR²':<10}")
    for i, cov in enumerate(COVERAGES):
        d = results['sogn_gru_64x1'][i] - results['standard_gru'][i]
        print(f"{cov:<8.1f} {results['sogn_gru_64x1'][i]:<14.4f} {results['standard_gru'][i]:<14.4f} {d:<+10.4f}")

"""Evaluate coverage curves for original vs split-sample SOGN on Diabetes."""
import numpy as np
import json, os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

COVERAGES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
RESULT_DIR = os.path.join(os.path.dirname(__file__), "results")
SEEDS = [42, 43, 44]

# Load test labels
sys.path.insert(0, os.path.dirname(__file__))
from data_utils import prepare_diabetes_data
_, _, _, _, _, y_test, _, _ = prepare_diabetes_data()
y_true = y_test.numpy().squeeze()


def evaluate_method(method_name, seeds):
    all_seed_metrics = []
    for seed in seeds:
        sd = os.path.join(RESULT_DIR, method_name, f"seed_{seed}")
        y_pred = np.load(os.path.join(sd, "test_predictions.npy")).squeeze()
        scores = np.load(os.path.join(sd, "test_scores.npy")).squeeze()

        cov_metrics = []
        for cov in COVERAGES:
            n = len(y_true)
            k = max(1, int(np.ceil(n * cov)))
            idx = np.argsort(scores)[-k:]
            yt, yp = y_true[idx], y_pred[idx]
            cov_metrics.append({
                "coverage": cov,
                "n_selected": k,
                "r2": r2_score(yt, yp),
                "rmse": np.sqrt(mean_squared_error(yt, yp)),
                "mae": mean_absolute_error(yt, yp),
                "mape": mean_absolute_percentage_error(yt, yp) * 100,
            })
        all_seed_metrics.append(cov_metrics)
    return all_seed_metrics


original = evaluate_method("sogn", SEEDS)
split = evaluate_method("sogn_split_sample", SEEDS)

print(f"{'Coverage':<10} {'Orig R2':>10} {'Split R2':>10} {'Drop %':>10} {'Orig MAPE':>12} {'Split MAPE':>12}")
print("-" * 70)
for i, cov in enumerate(COVERAGES):
    orig_r2 = np.mean([s[i]["r2"] for s in original])
    split_r2 = np.mean([s[i]["r2"] for s in split])
    orig_mape = np.mean([s[i]["mape"] for s in original])
    split_mape = np.mean([s[i]["mape"] for s in split])
    drop = (orig_r2 - split_r2) / max(abs(orig_r2), 1e-8) * 100
    print(f"{cov:<10.1f} {orig_r2:>10.4f} {split_r2:>10.4f} {drop:>9.1f}% {orig_mape:>12.2f} {split_mape:>12.2f}")

# Print detailed per-coverage for split-sample
print(f"\n--- Split-Sample Detailed ---")
print(f"{'Cov':<8} {'R2 mean':>8} {'R2 std':>8} {'RMSE':>10} {'MAE':>10} {'MAPE':>10}")
for i, cov in enumerate(COVERAGES):
    r2_vals = [s[i]["r2"] for s in split]
    rmse_vals = [s[i]["rmse"] for s in split]
    mae_vals = [s[i]["mae"] for s in split]
    mape_vals = [s[i]["mape"] for s in split]
    print(f"{cov:<8.1f} {np.mean(r2_vals):>8.4f} {np.std(r2_vals):>8.4f} {np.mean(rmse_vals):>10.4f} {np.mean(mae_vals):>10.4f} {np.mean(mape_vals):>10.2f}")

print(f"\n--- Original Detailed ---")
print(f"{'Cov':<8} {'R2 mean':>8} {'R2 std':>8} {'RMSE':>10} {'MAE':>10} {'MAPE':>10}")
for i, cov in enumerate(COVERAGES):
    r2_vals = [s[i]["r2"] for s in original]
    rmse_vals = [s[i]["rmse"] for s in split]
    mae_vals = [s[i]["mae"] for s in original]
    mape_vals = [s[i]["mape"] for s in original]
    print(f"{cov:<8.1f} {np.mean(r2_vals):>8.4f} {np.std(r2_vals):>8.4f} {np.mean(rmse_vals):>10.4f} {np.mean(mae_vals):>10.4f} {np.mean(mape_vals):>10.2f}")

"""覆盖率评估 & 多 seed 汇总 — 加载所有方法结果, 输出对比表

用法:
  python evaluate.py --dataset california_housing
  python evaluate.py --dataset beijing_pm25
"""

import numpy as np
import json
import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import argparse
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

COVERAGES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
METHODS = ["sogn", "standard", "mc_dropout", "deep_ensemble", "deep_evidential",
           "selectivenet", "conformal"]
METHOD_NAMES = {
    "sogn": "SOGN (Ours)",
    "standard": "Standard",
    "mc_dropout": "MC Dropout",
    "deep_ensemble": "Deep Ensemble",
    "deep_evidential": "Deep Evidential",
    "selectivenet": "SelectiveNet",
    "conformal": "Split Conformal",
}
SEEDS = [42, 43, 44]


def evaluate_coverage(y_true, y_pred, scores, coverage):
    """按 scores 降序选 top coverage 比例的样本，计算指标"""
    n = len(y_true)
    k = max(1, int(np.ceil(n * coverage)))
    idx = np.argsort(scores)[-k:]
    yt = y_true[idx]
    yp = y_pred[idx]
    return {
        "coverage": coverage,
        "n_selected": k,
        "r2": r2_score(yt, yp),
        "rmse": np.sqrt(mean_squared_error(yt, yp)),
        "mae": mean_absolute_error(yt, yp),
        "mape": mean_absolute_percentage_error(yt, yp) * 100,
    }


def load_seed_result(result_dir, dataset, method, seed):
    """加载单个 seed 的结果"""
    path = os.path.join(result_dir, dataset, "results", method, f"seed_{seed}")
    y_pred = np.load(os.path.join(path, "test_predictions.npy"))
    scores = np.load(os.path.join(path, "test_scores.npy"))
    y_true = np.load(os.path.join(path, "test_labels.npy"))
    return y_pred, scores, y_true


def eval_dataset(result_dir, dataset):
    """评估一个数据集上所有方法的所有 seed，输出汇总"""
    print(f"\n{'='*80}")
    print(f"Dataset: {dataset}")
    print(f"{'='*80}")

    # 收集所有方法 × seed 的结果
    all_data = {}  # method -> [{cov: metrics}, ...] per seed
    for method in METHODS:
        mdir = os.path.join(result_dir, dataset, "results", method)
        if not os.path.exists(mdir):
            print(f"  [SKIP] {method}: no results found")
            continue

        seed_metrics_list = []
        for seed in SEEDS:
            try:
                y_pred, scores, y_true = load_seed_result(result_dir, dataset, method, seed)
            except FileNotFoundError:
                print(f"  [SKIP] {method}/seed_{seed}: not found")
                continue
            cov_metrics = []
            for cov in COVERAGES:
                cov_metrics.append(evaluate_coverage(y_true, y_pred, scores, cov))
            seed_metrics_list.append(cov_metrics)
        if seed_metrics_list:
            all_data[method] = seed_metrics_list

    if not all_data:
        print("No results to evaluate.")
        return

    # ---- 打印每个方法详细结果 ----
    for method, seed_list in all_data.items():
        print(f"\n--- {METHOD_NAMES.get(method, method)} ---")
        print(f"{'Cov':<8} {'R² (mean±std)':<22} {'RMSE':<10} {'MAE':<10} {'MAPE%':<10}")
        for i, cov in enumerate(COVERAGES):
            r2_vals = [s[i]["r2"] for s in seed_list]
            rmse_vals = [s[i]["rmse"] for s in seed_list]
            mae_vals = [s[i]["mae"] for s in seed_list]
            mape_vals = [s[i]["mape"] for s in seed_list]
            r2_mean, r2_std = np.mean(r2_vals), np.std(r2_vals)
            rmse_mean = np.mean(rmse_vals)
            mae_mean = np.mean(mae_vals)
            mape_mean = np.mean(mape_vals)
            print(f"{cov:<8.1f} {r2_mean:>7.4f} ± {r2_std:<.4f}     {rmse_mean:>8.4f}  {mae_mean:>8.4f}  {mape_mean:>8.2f}")

    # ---- R² 对比矩阵 ----
    print(f"\n{'='*80}")
    print("R² 对比矩阵 (mean over seeds)")
    print(f"{'='*80}")
    header = f"{'Cov':<8}"
    for m in METHODS:
        if m in all_data:
            header += f" {METHOD_NAMES.get(m, m)[:14]:>14}"
    print(header)
    print("-" * len(header))
    for i, cov in enumerate(COVERAGES):
        row = f"{cov:<8.1f}"
        best_r2, best_method = -999, ""
        for m in METHODS:
            if m in all_data:
                r2_vals = [s[i]["r2"] for s in all_data[m]]
                r2_mean = np.mean(r2_vals)
                row += f" {r2_mean:>14.4f}"
                if r2_mean > best_r2:
                    best_r2, best_method = r2_mean, m
            else:
                row += f" {'--':>14}"
        # 加粗最佳 (markdown)
        row += f"   ← {METHOD_NAMES.get(best_method, best_method)}"
        print(row)

    # ---- 汇总 JSON ----
    summary = {}
    for method, seed_list in all_data.items():
        method_summary = []
        for i, cov in enumerate(COVERAGES):
            r2_vals = [s[i]["r2"] for s in seed_list]
            rmse_vals = [s[i]["rmse"] for s in seed_list]
            mae_vals = [s[i]["mae"] for s in seed_list]
            mape_vals = [s[i]["mape"] for s in seed_list]
            method_summary.append({
                "coverage": cov,
                "r2_mean": float(np.mean(r2_vals)),
                "r2_std": float(np.std(r2_vals)),
                "rmse_mean": float(np.mean(rmse_vals)),
                "rmse_std": float(np.std(rmse_vals)),
                "mae_mean": float(np.mean(mae_vals)),
                "mae_std": float(np.std(mae_vals)),
                "mape_mean": float(np.mean(mape_vals)),
                "mape_std": float(np.std(mape_vals)),
            })
        summary[method] = method_summary

    summary_path = os.path.join(result_dir, dataset, "results", "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["california_housing", "beijing_pm25", "bike_sharing"])
    args = parser.parse_args()

    result_dir = os.path.dirname(os.path.abspath(__file__))
    eval_dataset(result_dir, args.dataset)


if __name__ == "__main__":
    main()

"""7. 覆盖率评估 — 加载所有模型，更改筛选比例直接出结果"""
import numpy as np
import json
import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

RESULT_DIR = os.path.join(os.path.dirname(__file__), "results")
SUMMARY_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(SUMMARY_DIR, exist_ok=True)

COVERAGES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
METHODS = ["sogn", "standard", "mc_dropout", "deep_ensemble", "deep_evidential",
           "selectivenet", "conformal"]
METHOD_NAMES = {
    "sogn": "SOGN (Ours)",
    "standard": "Standard Regression",
    "mc_dropout": "MC Dropout",
    "deep_ensemble": "Deep Ensemble",
    "deep_evidential": "Deep Evidential",
    "selectivenet": "SelectiveNet",
    "conformal": "Split Conformal",
}


def evaluate_coverage(y_true, y_pred, scores, coverage):
    """按 scores 降序选 top coverage 比例的样本，计算指标"""
    n = len(y_true)
    k = max(1, int(np.ceil(n * coverage)))
    # scores 越高越好，选 top-k
    top_idx = np.argsort(scores)[-k:]
    y_true_sel = y_true[top_idx]
    y_pred_sel = y_pred[top_idx]

    r2 = r2_score(y_true_sel, y_pred_sel)
    rmse = np.sqrt(mean_squared_error(y_true_sel, y_pred_sel))
    mae = mean_absolute_error(y_true_sel, y_pred_sel)
    mape = mean_absolute_percentage_error(y_true_sel, y_pred_sel) * 100
    return {"coverage": coverage, "n_selected": k, "r2": r2, "rmse": rmse, "mae": mae, "mape": mape}


def load_method_results(method):
    """加载方法的预测和分数"""
    method_dir = os.path.join(RESULT_DIR, method)
    y_pred = np.load(os.path.join(method_dir, "test_predictions.npy"))
    scores = np.load(os.path.join(method_dir, "test_scores.npy"))
    with open(os.path.join(method_dir, "config.json")) as f:
        config = json.load(f)
    return y_pred, scores, config


def main():
    # 重新生成测试集标签（deterministic split，与训练时一致）
    from data_utils import prepare_diabetes_data
    _, _, _, _, _, y_test, _, _ = prepare_diabetes_data()
    y_true = y_test.numpy().squeeze()

    all_results = []
    summary_rows = []

    for method in METHODS:
        method_dir = os.path.join(RESULT_DIR, method)
        if not os.path.exists(method_dir):
            print(f"  [SKIP] {method}: no results found")
            continue

        y_pred, scores, config = load_method_results(method)
        print(f"\n{'='*60}")
        print(f"{METHOD_NAMES.get(method, method)}")
        print(f"{'='*60}")
        print(f"{'Coverage':<10} {'R²':>8} {'RMSE':>8} {'MAE':>8} {'MAPE%':>10}")

        method_results = {"method": method, "config": config, "coverages": []}
        for cov in COVERAGES:
            metrics = evaluate_coverage(y_true, y_pred, scores, cov)
            method_results["coverages"].append(metrics)
            print(f"{cov:<10.1f} {metrics['r2']:>8.4f} {metrics['rmse']:>8.4f} {metrics['mae']:>8.4f} {metrics['mape']:>10.2f}")
            summary_rows.append({
                "method": method,
                "method_name": METHOD_NAMES.get(method, method),
                "coverage": cov,
                **{k: v for k, v in metrics.items() if k != "coverage"},
            })

        # 保存每个方法的独立结果
        with open(os.path.join(method_dir, "coverage_results.json"), "w") as f:
            json.dump(method_results, f, indent=2)
        all_results.append(method_results)

    # 保存汇总表格
    df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(SUMMARY_DIR, "all_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSummary saved to {csv_path}")

    # 打印对比表：每个覆盖率下最优 R²
    print(f"\n{'='*80}")
    print("R² 对比矩阵")
    print(f"{'='*80}")
    header = f"{'Coverage':<10}"
    for m in METHODS:
        header += f" {METHOD_NAMES.get(m, m)[:12]:>12}"
    print(header)
    print("-" * len(header))
    for i, cov in enumerate(COVERAGES):
        row = f"{cov:<10.1f}"
        for m in METHODS:
            if m in [r["method"] for r in all_results]:
                r2_val = [r for r in all_results if r["method"] == m][0]["coverages"][i]["r2"]
                row += f" {r2_val:>12.4f}"
            else:
                row += f" {'--':>12}"
        print(row)

    # 保存完整结果 JSON
    with open(os.path.join(SUMMARY_DIR, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()

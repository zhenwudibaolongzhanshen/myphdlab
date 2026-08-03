"""显著性检验: SOGN vs 各 baseline 在每个覆盖率下的 paired t-test / Wilcoxon

用法:
  python significance_test.py --dataset california_housing
  python significance_test.py --dataset bike_sharing
"""

import numpy as np
import json, os, sys, io, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from scipy import stats

COVERAGES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
METHODS = ["sogn", "standard", "mc_dropout", "deep_ensemble",
           "deep_evidential", "selectivenet", "conformal"]
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
    n = len(y_true)
    k = max(1, int(np.ceil(n * coverage)))
    idx = np.argsort(scores)[-k:]
    yt = y_true[idx]
    yp = y_pred[idx]
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return r2


def load_seed_result(result_dir, dataset, method, seed):
    path = os.path.join(result_dir, dataset, "results", method, f"seed_{seed}")
    y_pred = np.load(os.path.join(path, "test_predictions.npy"))
    scores = np.load(os.path.join(path, "test_scores.npy"))
    y_true = np.load(os.path.join(path, "test_labels.npy"))
    return y_pred, scores, y_true


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["california_housing", "beijing_pm25", "bike_sharing"])
    args = parser.parse_args()

    result_dir = os.path.dirname(os.path.abspath(__file__))

    # 加载所有方法 × seed 的 R²
    all_r2 = {}  # method -> [seed0_covs, seed1_covs, seed2_covs]
    for method in METHODS:
        mdir = os.path.join(result_dir, args.dataset, "results", method)
        if not os.path.exists(mdir):
            continue
        seed_r2s = []
        for seed in SEEDS:
            try:
                y_pred, scores, y_true = load_seed_result(result_dir, args.dataset, method, seed)
            except FileNotFoundError:
                continue
            cov_r2 = [evaluate_coverage(y_true, y_pred, scores, c) for c in COVERAGES]
            seed_r2s.append(cov_r2)
        if seed_r2s:
            all_r2[method] = np.array(seed_r2s)  # (n_seeds, n_coverages)

    if "sogn" not in all_r2:
        print("SOGN results not found.")
        return

    sogn_r2 = all_r2["sogn"]  # (n_seeds, n_coverages)

    print(f"\n{'='*90}")
    print(f"  显著性检验: {args.dataset}  —  SOGN vs Baselines (paired t-test)")
    print(f"{'='*90}")
    print(f"  标注: * p<0.1, ** p<0.05, *** p<0.01  (双尾 paired t-test, 3 seeds)")
    print(f"  + 表示 SOGN 显著优于 baseline, - 表示 SOGN 显著劣于 baseline")
    print()

    for method in METHODS:
        if method == "sogn" or method not in all_r2:
            continue
        base_r2 = all_r2[method]
        name = METHOD_NAMES.get(method, method)
        print(f"  {'Coverage':<10}", end="")
        for i, cov in enumerate(COVERAGES):
            print(f"  {cov:<6.1f}", end="")
        print(f"\n  {'':10}", end="")
        for i, cov in enumerate(COVERAGES):
            diff = sogn_r2[:, i] - base_r2[:, i]
            t_stat, p_val = stats.ttest_rel(sogn_r2[:, i], base_r2[:, i])
            mean_diff = np.mean(diff)
            if p_val < 0.01:
                sig = "***" if mean_diff > 0 else "---"
            elif p_val < 0.05:
                sig = "**" if mean_diff > 0 else "--"
            elif p_val < 0.1:
                sig = "*" if mean_diff > 0 else "-"
            else:
                sig = ""
            print(f"  {mean_diff:+.4f}{sig:<3}", end="")
        print(f"    ← {name}")
    print()

    # ---- Wilcoxon signed-rank test (non-parametric, more conservative) ----
    print(f"{'='*90}")
    print(f"  Wilcoxon signed-rank test (non-parametric)")
    print(f"{'='*90}")
    print(f"  {'Coverage':<10}", end="")
    for cov in COVERAGES:
        print(f"  {cov:<6.1f}", end="")
    print()
    for method in METHODS:
        if method == "sogn" or method not in all_r2:
            continue
        base_r2 = all_r2[method]
        name = METHOD_NAMES.get(method, method)
        print(f"  {'':10}", end="")
        for i, cov in enumerate(COVERAGES):
            diff = sogn_r2[:, i] - base_r2[:, i]
            if np.allclose(diff, 0):
                print(f"  {'--':6}", end="")
                continue
            try:
                w_stat, p_val = stats.wilcoxon(sogn_r2[:, i], base_r2[:, i])
                mean_diff = np.mean(diff)
                if p_val < 0.05:
                    sig = "+" if mean_diff > 0 else "-"
                else:
                    sig = " "
                print(f"  {mean_diff:+.4f}{sig}", end="")
            except ValueError:
                print(f"  {'N/A':6}", end="")
        print(f"  ← {name}")

    # ---- LaTeX table output ----
    print(f"\n\n{'='*90}")
    print(f"  LaTeX Table (可直接复制到论文)")
    print(f"{'='*90}")
    print(r"\begin{table}[H]")
    print(r"\centering")
    print(r"\caption{SOGN vs baselines 显著性检验（paired t-test, 3 seeds）}")
    print(r"\label{tab:sig_" + args.dataset + "}")
    print(r"\small")
    cols = "l" + "c" * (len(COVERAGES) + 1)
    print(r"\begin{tabular}{" + cols + "}")
    print(r"\toprule")
    cov_str = " & ".join([f"{c:.1f}" for c in COVERAGES])
    print(f"  Method & {cov_str} \\\\")
    print(r"\midrule")
    for method in METHODS:
        if method == "sogn" or method not in all_r2:
            continue
        base_r2 = all_r2[method]  # (3, 10)
        name = METHOD_NAMES.get(method, method)
        row = f"  {name}"
        for i, cov in enumerate(COVERAGES):
            diff = sogn_r2[:, i] - base_r2[:, i]
            mean_diff = np.mean(diff)
            t_stat, p_val = stats.ttest_rel(sogn_r2[:, i], base_r2[:, i])
            if p_val < 0.01:
                sig = "^{***}" if mean_diff > 0 else "^{---}"
            elif p_val < 0.05:
                sig = "^{**}" if mean_diff > 0 else "^{--}"
            elif p_val < 0.1:
                sig = "^{*}" if mean_diff > 0 else "^{-}"
            else:
                sig = ""
            row += f" & ${mean_diff:+.4f}{sig}$"
        row += r" \\"
        print(row)
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()

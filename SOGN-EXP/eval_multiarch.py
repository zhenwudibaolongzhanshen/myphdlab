"""多架构评估: Bike Sharing 上 LSTM / GRU / Transformer × SOGN / Standard / MC Dropout

用法: python eval_multiarch.py
"""

import numpy as np
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ARCHITECTURES = ["lstm", "gru", "transformer"]
METHODS = ["sogn", "standard", "mc_dropout"]
METHOD_NAMES = {"sogn": "SOGN", "standard": "Standard", "mc_dropout": "MC Dropout"}
ARCH_NAMES = {"lstm": "LSTM", "gru": "GRU", "transformer": "Transformer"}
SEEDS = [42, 43, 44]
COVERAGES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def evaluate_coverage(y_true, y_pred, scores, coverage):
    n = len(y_true)
    k = max(1, int(np.ceil(n * coverage)))
    idx = np.argsort(scores)[-k:]
    yt = y_true[idx]
    yp = y_pred[idx]
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


def load_seed_result(result_dir, method, seed):
    path = os.path.join(result_dir, method, f"seed_{seed}")
    if not os.path.exists(path):
        return None, None, None
    y_pred = np.load(os.path.join(path, "test_predictions.npy"))
    scores = np.load(os.path.join(path, "test_scores.npy"))
    y_true = np.load(os.path.join(path, "test_labels.npy"))
    return y_pred, scores, y_true


def main():
    result_dir = os.path.join(os.path.dirname(__file__), "bike_sharing", "results")

    # 收集所有 arch × method 的结果
    # arch -> method -> {mean_r2: [10], std_r2: [10]}
    all_data = {}

    for arch in ARCHITECTURES:
        all_data[arch] = {}
        for method in METHODS:
            # LSTM results are in "sogn", "standard", "mc_dropout"
            # GRU results are in "sogn_gru", etc.
            # Transformer results are in "sogn_transformer", etc.
            if arch == "lstm":
                method_dir = method
            else:
                method_dir = f"{method}_{arch}"

            seed_r2s = []
            for seed in SEEDS:
                y_pred, scores, y_true = load_seed_result(result_dir, method_dir, seed)
                if y_pred is None:
                    continue
                cov_r2 = [evaluate_coverage(y_true, y_pred, scores, c) for c in COVERAGES]
                seed_r2s.append(cov_r2)
            if seed_r2s:
                arr = np.array(seed_r2s)
                all_data[arch][method] = {"mean": arr.mean(0), "std": arr.std(0)}

    # ========================================
    # 详细表：每个架构的方法对比
    # ========================================
    for arch in ARCHITECTURES:
        print(f"\n{'='*80}")
        print(f"  {ARCH_NAMES[arch]} Backbone — Coverage-based R2")
        print(f"{'='*80}")
        header = f"  {'Cov':<8}"
        for m in METHODS:
            if m in all_data[arch]:
                header += f" {METHOD_NAMES[m]:>16}"
        print(header)
        print("  " + "-" * (8 + 16 * len(METHODS)))
        for i, cov in enumerate(COVERAGES):
            row = f"  {cov:<8.1f}"
            for m in METHODS:
                if m in all_data[arch]:
                    row += f" {all_data[arch][m]['mean'][i]:>16.4f}"
            print(row)

    # ========================================
    # 核心对比：SOGN vs Standard 的 Delta R2
    # ========================================
    print(f"\n{'='*80}")
    print(f"  SOGN - Standard  Delta R2 (正=SOGN优于Standard)")
    print(f"{'='*80}")
    header = f"  {'Cov':<8}"
    for arch in ARCHITECTURES:
        header += f" {ARCH_NAMES[arch]:>12}"
    print(header)
    print("  " + "-" * (8 + 12 * len(ARCHITECTURES)))
    for i, cov in enumerate(COVERAGES):
        row = f"  {cov:<8.1f}"
        for arch in ARCHITECTURES:
            if "sogn" in all_data[arch] and "standard" in all_data[arch]:
                delta = all_data[arch]["sogn"]["mean"][i] - all_data[arch]["standard"]["mean"][i]
                row += f" {delta:>+12.4f}"
            else:
                row += f" {'--':>12}"
        print(row)

    # ========================================
    # LaTeX 表：多架构选择性对比
    # ========================================
    print(f"\n\n{'='*80}")
    print(f"  LaTeX Table")
    print(f"{'='*80}")
    print(r"\begin{table}[H]")
    print(r"\centering")
    print(r"\caption{SOGN vs. Standard across backbone architectures (Bike Sharing)}")
    print(r"\label{tab:multiarch}")
    print(r"\small")
    print(r"\begin{tabular}{lcccccc}")
    print(r"\toprule")
    print(r"& \multicolumn{3}{c}{SOGN} & \multicolumn{3}{c}{Standard} \\")
    print(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}")
    print(r"Coverage & LSTM & GRU & Transformer & LSTM & GRU & Transformer \\")
    print(r"\midrule")
    for i, cov in enumerate(COVERAGES):
        row = f"  {cov:.1f}"
        for arch in ARCHITECTURES:
            if "sogn" in all_data[arch]:
                row += f" & {all_data[arch]['sogn']['mean'][i]:.4f}"
            else:
                row += f" & --"
        for arch in ARCHITECTURES:
            if "standard" in all_data[arch]:
                row += f" & {all_data[arch]['standard']['mean'][i]:.4f}"
            else:
                row += f" & --"
        row += r" \\"
        print(row)
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    # ========================================
    # Delta 表
    # ========================================
    print()
    print(r"\begin{table}[H]")
    print(r"\centering")
    print(r"\caption{SOGN selectivity gain ($\Delta R^2 = R^2_{\text{SOGN}} - R^2_{\text{Standard}}$) across backbones}")
    print(r"\label{tab:multiarch_delta}")
    print(r"\small")
    print(r"\begin{tabular}{lccc}")
    print(r"\toprule")
    print(r"Coverage & LSTM & GRU & Transformer \\")
    print(r"\midrule")
    for i, cov in enumerate(COVERAGES):
        row = f"  {cov:.1f}"
        for arch in ARCHITECTURES:
            if "sogn" in all_data[arch] and "standard" in all_data[arch]:
                delta = all_data[arch]["sogn"]["mean"][i] - all_data[arch]["standard"]["mean"][i]
                row += f" & ${delta:+.4f}$"
            else:
                row += f" & --"
        row += r" \\"
        print(row)
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()

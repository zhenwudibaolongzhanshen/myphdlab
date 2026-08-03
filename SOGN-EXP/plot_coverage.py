"""R² vs Coverage 折线图 — 三数据集 × 7 方法

用法:
  python plot_coverage.py --dataset california_housing
  python plot_coverage.py --dataset bike_sharing
  python plot_coverage.py --dataset all          # 三数据集并排
"""

import numpy as np
import json, os, sys, io, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

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
# 颜色和线条风格
METHOD_STYLES = {
    "sogn":           {"color": "#D62728", "marker": "o", "lw": 2.5, "ms": 8, "ls": "-",  "zorder": 10},
    "standard":       {"color": "#7F7F7F", "marker": "s", "lw": 1.2, "ms": 6, "ls": "--", "zorder": 3},
    "mc_dropout":     {"color": "#1F77B4", "marker": "^", "lw": 1.2, "ms": 6, "ls": "--", "zorder": 3},
    "deep_ensemble":  {"color": "#2CA02C", "marker": "D", "lw": 1.2, "ms": 6, "ls": "-.", "zorder": 4},
    "deep_evidential":{"color": "#FF7F0E", "marker": "v", "lw": 1.2, "ms": 6, "ls": ":",  "zorder": 3},
    "selectivenet":   {"color": "#9467BD", "marker": "p", "lw": 1.2, "ms": 6, "ls": "--", "zorder": 3},
    "conformal":      {"color": "#8C564B", "marker": "*", "lw": 1.2, "ms": 7, "ls": ":",  "zorder": 3},
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
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


def load_seed_result(result_dir, dataset, method, seed):
    path = os.path.join(result_dir, dataset, "results", method, f"seed_{seed}")
    y_pred = np.load(os.path.join(path, "test_predictions.npy"))
    scores = np.load(os.path.join(path, "test_scores.npy"))
    y_true = np.load(os.path.join(path, "test_labels.npy"))
    return y_pred, scores, y_true


def load_dataset_results(result_dir, dataset):
    all_data = {}
    for method in METHODS:
        mdir = os.path.join(result_dir, dataset, "results", method)
        if not os.path.exists(mdir):
            continue
        seed_r2s = []
        for seed in SEEDS:
            try:
                y_pred, scores, y_true = load_seed_result(result_dir, dataset, method, seed)
            except FileNotFoundError:
                continue
            cov_r2 = [evaluate_coverage(y_true, y_pred, scores, c) for c in COVERAGES]
            seed_r2s.append(cov_r2)
        if seed_r2s:
            arr = np.array(seed_r2s)
            all_data[method] = {"mean": arr.mean(0), "std": arr.std(0)}
    return all_data


def plot_one_dataset(ax, all_data, title, show_legend=True):
    for method in METHODS:
        if method not in all_data:
            continue
        d = all_data[method]
        sty = METHOD_STYLES[method]
        name = METHOD_NAMES.get(method, method)
        ax.errorbar(COVERAGES, d["mean"], yerr=d["std"],
                    label=name, **sty, capsize=2, capthick=0.5, alpha=0.9)
    ax.set_xlabel("Coverage", fontsize=11)
    ax.set_ylabel("$R^2$", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlim(0.05, 1.05)
    ax.set_xticks(COVERAGES)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    if show_legend:
        ax.legend(fontsize=7, ncol=2, loc='lower right',
                  framealpha=0.9, edgecolor='gray', fancybox=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="california_housing",
                        choices=["california_housing", "beijing_pm25", "bike_sharing", "all"])
    args = parser.parse_args()

    result_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(result_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)

    # 中文字体设置
    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    plt.rcParams['font.size'] = 10

    if args.dataset == "all":
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        datasets = ["california_housing", "bike_sharing"]
        titles = ["California Housing (MLP)", "Bike Sharing (LSTM)"]
        for ax, ds, title in zip(axes, datasets, titles):
            data = load_dataset_results(result_dir, ds)
            plot_one_dataset(ax, data, title, show_legend=(ds == datasets[-1]))
        plt.tight_layout()
        out_path = os.path.join(output_dir, "r2_coverage_all.pdf")
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        fig.savefig(out_path.replace('.pdf', '.png'), dpi=200, bbox_inches='tight')
        print(f"Saved to {out_path}")
    else:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        data = load_dataset_results(result_dir, args.dataset)
        ds_names = {"california_housing": "California Housing (MLP)",
                    "bike_sharing": "Bike Sharing (LSTM)",
                    "beijing_pm25": "Beijing PM2.5 (LSTM)"}
        plot_one_dataset(ax, data, ds_names.get(args.dataset, args.dataset))
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"r2_coverage_{args.dataset}.pdf")
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        fig.savefig(out_path.replace('.pdf', '.png'), dpi=200, bbox_inches='tight')
        print(f"Saved to {out_path}")

    plt.close()
    print("Done.")


if __name__ == "__main__":
    main()

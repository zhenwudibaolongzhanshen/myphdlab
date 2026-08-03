"""Create a compact, presentation-ready RC-49 experiment report."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
REPORT = ROOT / "final_report"
FIGURES = REPORT / "figures"

METHOD_NAMES = {
    "sogn": "SOGN",
    "standard": "Standard",
    "mc_dropout": "MC Dropout",
    "deep_ensemble": "Deep Ensemble",
    "deep_evidential": "Deep Evidential",
    "selectivenet": "SelectiveNet",
    "conformal": "Conformal",
}

COLORS = {
    "sogn": "#D94841",
    "standard": "#6B7280",
    "mc_dropout": "#9B59B6",
    "deep_ensemble": "#247BA0",
    "deep_evidential": "#1B9E77",
    "selectivenet": "#E67E22",
    "conformal": "#2F80ED",
}


def copy_source_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    REPORT.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    coverage = pd.read_csv(RESULTS / "all_results.csv")
    seeds = pd.read_csv(RESULTS / "seed_results.csv")
    ranking = pd.read_csv(RESULTS / "full_coverage_ranking.csv")

    coverage.to_csv(REPORT / "rc49_coverage_results.csv", index=False, encoding="utf-8-sig")
    seeds.to_csv(REPORT / "rc49_seed_results.csv", index=False, encoding="utf-8-sig")
    ranking.to_csv(REPORT / "rc49_full_coverage_ranking.csv", index=False, encoding="utf-8-sig")
    shutil.copy2(RESULTS / "dataset_info.json", REPORT / "dataset_info.json")
    return coverage, seeds, ranking


def plot_r2_vs_coverage(coverage: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "legend.fontsize": 10,
        }
    )
    fig, ax = plt.subplots(figsize=(10.8, 6.4), constrained_layout=True)

    display_order = [
        "sogn",
        "deep_evidential",
        "deep_ensemble",
        "conformal",
        "selectivenet",
        "mc_dropout",
        "standard",
    ]
    for method in display_order:
        frame = coverage.loc[coverage["method"] == method].sort_values("coverage")
        x = frame["coverage"].to_numpy()
        y = frame["r2_mean"].to_numpy()
        std = frame["r2_std"].to_numpy()
        is_sogn = method == "sogn"
        ax.plot(
            x,
            y,
            color=COLORS[method],
            linewidth=3.2 if is_sogn else 1.9,
            marker="o",
            markersize=6.5 if is_sogn else 4.5,
            label=METHOD_NAMES[method],
            zorder=4 if is_sogn else 2,
        )
        ax.fill_between(
            x,
            y - std,
            y + std,
            color=COLORS[method],
            alpha=0.16 if is_sogn else 0.08,
            linewidth=0,
        )

    ax.set_title("RC-49: Selective Prediction Performance")
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Test $R^2$ (mean across 3 seeds)")
    ax.set_xlim(0.08, 1.02)
    ax.set_ylim(0.84, 1.005)
    ax.set_xticks([i / 10 for i in range(1, 11)])
    ax.grid(axis="y", color="#D1D5DB", linewidth=0.8, alpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=2, loc="lower left", frameon=True, framealpha=0.96)
    fig.savefig(FIGURES / "rc49_r2_vs_coverage.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "rc49_r2_vs_coverage.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_sogn_metrics(coverage: pd.DataFrame) -> None:
    metrics = [
        ("r2_mean", "r2_std", "Test $R^2$", "higher is better"),
        ("rmse_mean", "rmse_std", "Test RMSE", "lower is better"),
        ("mae_mean", "mae_std", "Test MAE", "lower is better"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.2), constrained_layout=True)
    methods = ["sogn", "deep_evidential", "conformal", "deep_ensemble"]

    for ax, (mean_col, std_col, label, direction) in zip(axes, metrics):
        for method in methods:
            frame = coverage.loc[coverage["method"] == method].sort_values("coverage")
            x = frame["coverage"].to_numpy()
            y = frame[mean_col].to_numpy()
            std = frame[std_col].to_numpy()
            is_sogn = method == "sogn"
            ax.plot(
                x,
                y,
                color=COLORS[method],
                linewidth=3.0 if is_sogn else 1.7,
                marker="o",
                markersize=5.5 if is_sogn else 4,
                label=METHOD_NAMES[method],
                zorder=3 if is_sogn else 2,
            )
            ax.fill_between(x, y - std, y + std, color=COLORS[method], alpha=0.11, linewidth=0)
        ax.set_xlabel("Coverage")
        ax.set_ylabel(label)
        ax.set_title(direction)
        ax.set_xlim(0.08, 1.02)
        ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
        ax.grid(axis="y", color="#D1D5DB", linewidth=0.8, alpha=0.75)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="lower left", frameon=True, framealpha=0.96)
    fig.savefig(FIGURES / "rc49_sogn_metric_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "rc49_sogn_metric_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def write_report(coverage: pd.DataFrame, ranking: pd.DataFrame) -> None:
    full = coverage.loc[coverage["coverage"] == 1.0].copy()
    full["method"] = full["method"].map(METHOD_NAMES)
    full = full.sort_values("r2_mean", ascending=False)

    coverage_rank = (
        coverage.assign(rank=coverage.groupby("coverage")["r2_mean"].rank(ascending=False, method="min"))
        .loc[lambda x: x["method"] == "sogn", ["coverage", "rank", "r2_mean", "rmse_mean", "mae_mean"]]
        .sort_values("coverage")
    )

    full_rows = "\n".join(
        f"| {idx + 1} | {row.method} | {row.r2_mean:.4f} +/- {row.r2_std:.4f} | {row.rmse_mean:.4f} | {row.mae_mean:.4f} |"
        for idx, row in enumerate(full.itertuples())
    )
    sogn_rows = "\n".join(
        f"| {row.coverage:.1f} | {int(row.rank)} | {row.r2_mean:.4f} | {row.rmse_mean:.4f} | {row.mae_mean:.4f} |"
        for row in coverage_rank.itertuples()
    )

    text = f"""# RC-49 七种方法横向对比实验结果

## 实验设置

- 数据集：RC-49 图像回归数据集。
- 特征：使用 PyTorch CUDA 完成 CNN 特征提取。
- 数据划分：训练集 10,000、验证集 2,000、测试集 5,000。
- 重复实验：随机种子 42、43、44；表中为三次实验的均值 +/- 标准差。
- 对比方法：SOGN、Standard、MC Dropout、Deep Ensemble、Deep Evidential、SelectiveNet、Conformal。
- SOGN 训练流程：序数预训练、SOGN 联合优化、回归头 MSE 微调。

## 全覆盖结果

Coverage = 1.0 时保留全部测试样本，代表标准回归精度。

| 排名 | 方法 | R2 | RMSE | MAE |
|---:|---|---:|---:|---:|
{full_rows}

SOGN 在全覆盖 R2 中排名 **第 2 / 7**，并取得全覆盖条件下的**最低 MAE**（{full.loc[full['method'] == 'SOGN', 'mae_mean'].iloc[0]:.4f}）。

## SOGN 选择性预测结果

| Coverage | R2 排名 / 7 | R2 | RMSE | MAE |
|---:|---:|---:|---:|---:|
{sogn_rows}

## 图表

- `figures/rc49_r2_vs_coverage.png`：七种方法的 R2-Coverage 折线图，阴影表示三次实验的一个标准差。
- `figures/rc49_sogn_metric_comparison.png`：SOGN 与三个强基线在 R2、RMSE、MAE 上的折线对比图。
- 同时提供 PDF 格式，可直接用于论文或幻灯片。

## 文件说明

- `rc49_full_coverage_ranking.csv`：全覆盖主结果表。
- `rc49_coverage_results.csv`：全部方法在每个 coverage 下的汇总结果。
- `rc49_seed_results.csv`：逐随机种子原始指标，用于复现均值和标准差。
- `dataset_info.json`：数据集划分与特征元数据。
"""
    (REPORT / "README.md").write_text(text, encoding="utf-8")

    report_json = {
        "dataset": "RC-49",
        "n_seeds": 3,
        "seeds": [42, 43, 44],
        "split": {"train": 10000, "validation": 2000, "test": 5000},
        "full_coverage_ranking": ranking.to_dict(orient="records"),
        "sogn_coverage_results": coverage.loc[coverage["method"] == "sogn"].to_dict(orient="records"),
    }
    (REPORT / "summary.json").write_text(
        json.dumps(report_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    coverage, _, ranking = copy_source_tables()
    plot_r2_vs_coverage(coverage)
    plot_sogn_metrics(coverage)
    write_report(coverage, ranking)
    print(f"Created RC-49 report package: {REPORT}")


if __name__ == "__main__":
    main()

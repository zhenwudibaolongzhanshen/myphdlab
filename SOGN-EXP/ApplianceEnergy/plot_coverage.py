"""Plot R-squared against coverage from evaluate_results.py output."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
SUMMARY = ROOT / "results" / "summary.json"
OUTPUT = ROOT / "figures" / "r2_coverage_appliance_energy.png"
LABELS = {
    "sogn": "SOGN (Ours)",
    "standard": "Standard",
    "mc_dropout": "MC Dropout",
    "deep_ensemble": "Deep Ensemble",
    "deep_evidential": "Deep Evidential",
    "selectivenet": "SelectiveNet",
    "conformal": "Split Conformal",
}
STYLES = {
    "sogn": {"color": "#b42318", "marker": "o", "linewidth": 2.4},
    "standard": {"color": "#555555", "marker": "s", "linewidth": 1.3},
    "mc_dropout": {"color": "#1d70b8", "marker": "^", "linewidth": 1.3},
    "deep_ensemble": {"color": "#198754", "marker": "D", "linewidth": 1.3},
    "deep_evidential": {"color": "#d97706", "marker": "v", "linewidth": 1.3},
    "selectivenet": {"color": "#7c3aed", "marker": "P", "linewidth": 1.3},
    "conformal": {"color": "#8b5e3c", "marker": "*", "linewidth": 1.3},
}


def main():
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    OUTPUT.parent.mkdir(exist_ok=True)
    fig, axis = plt.subplots(figsize=(8.4, 5.6))
    for method, rows in summary.items():
        coverage = [row["coverage"] for row in rows]
        r2_mean = [row["r2_mean"] for row in rows]
        r2_std = [row["r2_std"] for row in rows]
        axis.errorbar(
            coverage,
            r2_mean,
            yerr=r2_std,
            label=LABELS.get(method, method),
            capsize=2,
            **STYLES.get(method, {}),
        )
    axis.set(xlabel="Coverage", ylabel="R-squared", xlim=(0.05, 1.05), title="Appliance Energy")
    axis.grid(alpha=0.3, linestyle="--")
    axis.legend(fontsize=8, ncol=2, frameon=True)
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=200)
    fig.savefig(OUTPUT.with_suffix(".pdf"), dpi=200)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()

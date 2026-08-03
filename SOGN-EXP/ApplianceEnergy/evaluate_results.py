"""Evaluate all seven Appliance Energy methods at fixed coverage levels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
COVERAGES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
METHODS = (
    "sogn",
    "standard",
    "mc_dropout",
    "deep_ensemble",
    "deep_evidential",
    "selectivenet",
    "conformal",
)


def metrics_at_coverage(y_true, y_pred, scores, coverage):
    count = max(1, int(np.ceil(len(y_true) * coverage)))
    indices = np.argsort(scores.reshape(-1))[-count:]
    y_true, y_pred = y_true.reshape(-1)[indices], y_pred.reshape(-1)[indices]
    return {
        "coverage": coverage,
        "n_selected": count,
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def main():
    summary = {}
    for method in METHODS:
        seed_results = []
        for seed_dir in sorted((RESULTS / method).glob("seed_*")):
            try:
                y_true = np.load(seed_dir / "test_labels.npy")
                y_pred = np.load(seed_dir / "test_predictions.npy")
                scores = np.load(seed_dir / "test_scores.npy")
            except FileNotFoundError:
                continue
            seed_results.append([metrics_at_coverage(y_true, y_pred, scores, c) for c in COVERAGES])
        if not seed_results:
            continue

        summary[method] = []
        for index, coverage in enumerate(COVERAGES):
            rows = [seed[index] for seed in seed_results]
            summary[method].append(
                {
                    "coverage": coverage,
                    "seeds": len(rows),
                    "r2_mean": float(np.mean([row["r2"] for row in rows])),
                    "r2_std": float(np.std([row["r2"] for row in rows])),
                    "rmse_mean": float(np.mean([row["rmse"] for row in rows])),
                    "mae_mean": float(np.mean([row["mae"] for row in rows])),
                }
            )

    output = RESULTS / "summary.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Method                 R2 at full coverage (mean +/- std)")
    for method in METHODS:
        if method in summary:
            full = summary[method][-1]
            print(f"{method:<22} {full['r2_mean']:.4f} +/- {full['r2_std']:.4f}")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()

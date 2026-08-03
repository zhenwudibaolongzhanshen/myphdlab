"""Create Appliance Energy history windows for this self-contained benchmark."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "energydata_complete.csv"
OUTPUT = ROOT / "data" / "appliance_energy.npz"
WINDOW = 24


def main():
    frame = pd.read_csv(SOURCE, parse_dates=["date"])
    target = frame["Appliances"].to_numpy(dtype=np.float32)
    timestamps = frame.pop("date")
    features = frame.drop(columns=["Appliances", "rv1", "rv2"]).copy()
    features.insert(0, "Appliances_lag1", pd.Series(target).shift(1).bfill())
    minute = timestamps.dt.hour * 60 + timestamps.dt.minute
    features["hour_sin"] = np.sin(2 * np.pi * minute / 1440)
    features["hour_cos"] = np.cos(2 * np.pi * minute / 1440)
    features["weekday_sin"] = np.sin(2 * np.pi * timestamps.dt.dayofweek / 7)
    features["weekday_cos"] = np.cos(2 * np.pi * timestamps.dt.dayofweek / 7)

    values = features.to_numpy(dtype=np.float32)
    windows = np.stack([values[end - WINDOW : end] for end in range(WINDOW, len(values))])
    labels = target[WINDOW:].reshape(-1, 1)
    _, _, n_features = windows.shape
    x_scaler = StandardScaler().fit(windows.reshape(-1, n_features))
    y_scaler = StandardScaler().fit(labels)
    x_scaled = x_scaler.transform(windows.reshape(-1, n_features)).reshape(windows.shape).astype(np.float32)
    y_scaled = y_scaler.transform(labels).astype(np.float32)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUTPUT,
        X=x_scaled,
        y=y_scaled,
        x_mean=x_scaler.mean_,
        x_scale=x_scaler.scale_,
        y_mean=y_scaler.mean_,
        y_scale=y_scaler.scale_,
        window_size=WINDOW,
        n_features=n_features,
    )
    print(f"Saved {len(labels)} windows: X={x_scaled.shape}, y={y_scaled.shape}")


if __name__ == "__main__":
    main()

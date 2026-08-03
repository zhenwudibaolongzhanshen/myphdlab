"""Appliance Energy data interface used by the seven benchmark notebooks."""

from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_appliance_energy_bike_style(random_state=42, test_size=0.2, val_size=0.1):
    """Return the same random 70/10/20 split interface used by Bike Sharing."""
    data_file = Path(__file__).resolve().parent / "data" / "appliance_energy.npz"
    if not data_file.exists():
        raise FileNotFoundError(f"{data_file} does not exist; run prepare_data.py first")

    data = np.load(data_file)
    X, y = data["X"], data["y"]
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_size / (1 - test_size),
        random_state=random_state,
    )

    n_train, window, features = X_train.shape
    scaler = StandardScaler().fit(X_train.reshape(-1, features))
    X_train = scaler.transform(X_train.reshape(-1, features)).reshape(n_train, window, features)
    X_val = scaler.transform(X_val.reshape(-1, features)).reshape(-1, window, features)
    X_test = scaler.transform(X_test.reshape(-1, features)).reshape(-1, window, features)

    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).view(-1, 1),
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32).view(-1, 1),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32).view(-1, 1),
        scaler,
        (window, features),
    )

"""Hyperparameter tuning utilities using Optuna.

This module provides an Optuna-based routine to search over
Liquid Neural Network hyperparameters and persist the best
values to a ``.env`` file. The tuning relies on the existing
model construction and training helpers defined in
``time_series_deep_learner``.
"""

from __future__ import annotations

import os
import types
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from dotenv import set_key
from loguru import logger

try:  # pragma: no cover - handle optional dependency at import time
    import optuna  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - replaced in tests
    # monkeypatch in tests

    def _missing_optuna(*_args, **_kwargs):
        msg = "optuna is required for tuning; install it with `pip install optuna`."  # noqa: E501
        raise ModuleNotFoundError(msg)

    optuna = types.SimpleNamespace(create_study=_missing_optuna)

if TYPE_CHECKING:  # pragma: no cover - used only for type checkers
    import optuna as optuna  # noqa: F401

from .time_series_deep_learner import (
    build_liquid_model,
    create_rolling_window_sequences,
    train_model_with_rolling_window,
)


def _load_series(data_path: str) -> pd.DataFrame:
    """Load price data expected to contain a ``Date`` and ``Close`` column."""
    df = pd.read_csv(data_path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
    return df


def tune(data_path: str | None = None, n_trials: int = 40) -> "optuna.Study":
    """Run Optuna hyperparameter search.

    Parameters
    ----------
    data_path:
        Path to the CSV containing price data. If ``None`` the ``DATA_PATH``
        environment variable is used.
    n_trials:
        Number of Optuna trials to execute.
    """

    data_path = data_path or os.getenv("DATA_PATH", "./data/your_data.csv")
    df = _load_series(data_path)
    train_size = int(len(df) * float(os.getenv("TRAIN_SIZE_RATIO", 0.8)))
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]

    def objective(trial: "optuna.Trial") -> float:
        window = trial.suggest_int("WINDOW_SIZE", 5, 60, step=5)
        units = trial.suggest_categorical("LNN_UNITS", [32, 64, 96, 128, 192])
        drp = trial.suggest_float("DROPOUT_RATE", 0.0, 0.5)
        lr = trial.suggest_float("LEARNING_RATE", 1e-5, 5e-3, log=True)
        batch = trial.suggest_categorical("BATCH_SIZE", [16, 32, 64, 128])
        epochs = trial.suggest_int("EPOCHS", 20, 120)

        X_train, y_train = create_rolling_window_sequences(
            train_df["Close"].values, window
        )
        X_val, y_val = create_rolling_window_sequences(
            val_df["Close"].values,
            window,
        )
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

        model = build_liquid_model(
            input_shape=(window, 1),
            lnn_units=units,
            dropout_rate=drp,
            learning_rate=lr,
        )
        train_model_with_rolling_window(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            window_size=window,
            batch_size=batch,
            epochs=epochs,
        )
        preds = model.predict(X_val).flatten()
        val_loss = np.mean((preds - y_val) ** 2)
        return val_loss

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    env_path = Path(".") / ".env"
    best = study.best_params
    for key, value in best.items():
        set_key(str(env_path), key, str(value))
        logger.info("Set %s=%s", key, value)

    logger.info("Best validation loss %.5f", study.best_value)
    return study


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    tune()

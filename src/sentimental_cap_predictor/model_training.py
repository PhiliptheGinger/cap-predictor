"""Model construction, training and prediction helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from .modeling.time_series_deep_learner import (
    build_liquid_model,
    train_model_with_rolling_window,
)
from .modeling.bias_predictions import bias_predictions_with_sentiment


def train_model(
    train_data: pd.DataFrame,
    timesteps: int = 1,
    random_state: int | None = None,
    validation_split: float = 0.2,
):
    """Train the Liquid Neural Network model on ``train_data``.

    Parameters
    ----------
    train_data:
        DataFrame containing the ``close`` series used for training.
    timesteps:
        Number of timesteps for the model input shape.
    random_state:
        Optional seed to make training deterministic.
    validation_split:
        Fraction of data used for validation during training.
    """
    logger.info("Applying Liquid Neural Network for non-linear feature extraction.")

    if random_state is not None:
        np.random.seed(random_state)
        try:  # TensorFlow may not be installed in all environments
            import tensorflow as tf

            tf.random.set_seed(random_state)
        except ModuleNotFoundError:  # pragma: no cover - fallback for missing TF
            pass

    X_train = np.reshape(
        train_data["close"].values, (train_data.shape[0], timesteps, 1)
    )
    y_train = train_data["close"].values

    model = build_liquid_model(input_shape=(timesteps, 1))

    val_size = int(len(X_train) * validation_split)
    if val_size > 0:
        X_val, y_val = X_train[-val_size:], y_train[-val_size:]
        train_model_with_rolling_window(
            model,
            X_train[:-val_size],
            y_train[:-val_size],
            X_val,
            y_val,
            window_size=100,
        )
    else:
        train_model_with_rolling_window(model, X_train, y_train)

    return model


def predict_on_test_data(
    price_df: pd.DataFrame,
    model,
    test_data: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    timesteps: int = 1,
) -> pd.DataFrame:
    """Generate predictions for ``test_data`` and append to ``price_df``."""
    X_test = np.reshape(test_data["close"].values, (test_data.shape[0], timesteps, 1))
    predictions = model.predict(X_test).flatten()
    test_data = test_data.copy()
    test_data["predicted"] = predictions
    test_data = bias_predictions_with_sentiment(test_data, sentiment_df)
    price_df.loc[test_data.index, "predicted"] = predictions
    price_df.update(test_data)
    return price_df


def predict_on_future_data(
    price_df: pd.DataFrame,
    model,
    prediction_days: int,
    sentiment_df: pd.DataFrame,
    timesteps: int = 1,
) -> pd.DataFrame:
    """Forecast future days and append predictions to ``price_df``."""
    last_data = np.reshape(
        price_df["close"].values[-timesteps:], (timesteps, 1)
    )
    predictions: list[float] = []
    for _ in range(prediction_days):
        next_pred = model.predict(last_data.reshape(1, timesteps, 1)).flatten()
        predictions.append(next_pred[0])
        last_data = np.append(last_data[1:], next_pred).reshape(timesteps, 1)

    future_dates = pd.date_range(
        start=price_df.index[-1] + pd.Timedelta(days=1),
        periods=prediction_days,
        freq="D",
    )
    future_df = pd.DataFrame(index=future_dates, data=predictions, columns=["predicted"])
    future_df = bias_predictions_with_sentiment(future_df, sentiment_df)
    price_df = pd.concat([price_df, future_df])
    price_df.update(future_df)
    return price_df

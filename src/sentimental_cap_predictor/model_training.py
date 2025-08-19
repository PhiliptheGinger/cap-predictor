"""Model construction and training helpers.

This module isolates the logic for building the Liquid Neural
Network model and generating predictions so that command line
interfaces can remain focused on argument parsing and I/O.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from .modeling.time_series_deep_learner import build_liquid_model, train_model_with_rolling_window
from .modeling.bias_predictions import bias_predictions_with_sentiment


def train_and_predict(
    price_df: pd.DataFrame,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    mode: str,
    prediction_days: int,
    sentiment_df: pd.DataFrame,
    timesteps: int = 1,
) -> pd.DataFrame:
    """Train the LNN model and append predictions to ``price_df``.

    Parameters
    ----------
    price_df:
        Complete dataframe containing the ``close`` series used for
        training. It will be updated with prediction columns.
    train_data, test_data:
        Split subsets of ``price_df`` used for training and testing.
    mode:
        Either ``'train_test'`` or ``'production'``.
    prediction_days:
        Number of future days to predict in production mode.
    sentiment_df:
        Dataframe containing sentiment information used to bias
        predictions.
    timesteps:
        Number of timesteps for the model input shape.
    """
    logger.info("Applying Liquid Neural Network for non-linear feature extraction.")

    # Prepare training arrays
    X_train = np.reshape(train_data["close"].values, (train_data.shape[0], timesteps, 1))
    y_train = train_data["close"].values

    model = build_liquid_model(input_shape=(timesteps, 1))

    if mode == "train_test":
        X_test = np.reshape(test_data["close"].values, (test_data.shape[0], timesteps, 1))
        y_test = test_data["close"].values  # noqa: F841 - y_test kept for potential future use

        val_size = int(len(X_train) * 0.2)
        X_val, y_val = X_train[-val_size:], y_train[-val_size:]
        train_model_with_rolling_window(
            model,
            X_train[:-val_size],
            y_train[:-val_size],
            X_val,
            y_val,
            window_size=100,
        )

        predictions = model.predict(X_test).flatten()
        test_data = test_data.copy()
        test_data["predicted"] = predictions
        test_data = bias_predictions_with_sentiment(test_data, sentiment_df)
        price_df.loc[test_data.index, "predicted"] = predictions
        price_df.update(test_data)
    else:  # production mode
        train_model_with_rolling_window(model, X_train, y_train)
        last_data = X_train[-timesteps:]
        predictions = []
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

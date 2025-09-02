# flake8: noqa
# ruff: noqa

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from colorama import Fore, init
from dotenv import load_dotenv
from tensorflow.keras import activations, initializers
from tensorflow.keras.layers import RNN, Dense, Dropout, Layer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tqdm import tqdm


# Default hyperparameters; overriden by :func:`setup` reading from the
# environment.  This avoids performing any side effects when the module is
# imported purely for type checking or documentation.
LEARNING_RATE = 0.001
LNN_UNITS = 64
DROPOUT_RATE = 0.2
WINDOW_SIZE = 10
BATCH_SIZE = 32
EPOCHS = 50
PREDICTION_DAYS = 14
TRAIN_SIZE_RATIO = 0.8
DATA_PATH = "./data/your_data.csv"


def setup() -> None:
    """Load environment variables and configure colour handling."""
    load_dotenv()
    init(autoreset=True)

    global LEARNING_RATE, LNN_UNITS, DROPOUT_RATE, WINDOW_SIZE
    global BATCH_SIZE, EPOCHS, PREDICTION_DAYS, TRAIN_SIZE_RATIO, DATA_PATH

    LEARNING_RATE = float(os.getenv("LEARNING_RATE", LEARNING_RATE))
    LNN_UNITS = int(os.getenv("LNN_UNITS", LNN_UNITS))
    DROPOUT_RATE = float(os.getenv("DROPOUT_RATE", DROPOUT_RATE))
    WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", WINDOW_SIZE))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", BATCH_SIZE))
    EPOCHS = int(os.getenv("EPOCHS", EPOCHS))
    PREDICTION_DAYS = int(os.getenv("PREDICTION_DAYS", PREDICTION_DAYS))
    TRAIN_SIZE_RATIO = float(os.getenv("TRAIN_SIZE_RATIO", TRAIN_SIZE_RATIO))
    DATA_PATH = os.getenv("DATA_PATH", DATA_PATH)


# Custom Liquid Time-Constant (LTC) Layer
class LiquidLayer(Layer):
    def __init__(self, units, **kwargs):
        super(LiquidLayer, self).__init__(**kwargs)
        self.units = units
        self.state_size = units

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=initializers.RandomNormal(),
            trainable=True,
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer=initializers.RandomNormal(),
            trainable=True,
        )
        self.bias = self.add_weight(
            shape=(self.units,), initializer=initializers.Zeros(), trainable=True
        )
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        h = (
            tf.matmul(inputs, self.kernel)
            + tf.matmul(prev_output, self.recurrent_kernel)
            + self.bias
        )
        output = activations.tanh(h)
        return output, [output]

    def get_config(self):
        config = super(LiquidLayer, self).get_config()
        config.update({"units": self.units})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def create_rolling_window_sequences(data, window_size):
    """Generate rolling window sequences from time series data."""
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)


def build_liquid_model(
    input_shape,
    lnn_units: int = LNN_UNITS,
    dropout_rate: float = DROPOUT_RATE,
    learning_rate: float = LEARNING_RATE,
):
    """Construct the Liquid Neural Network model.

    Parameters
    ----------
    input_shape:
        Shape of the input data for the model.
    lnn_units:
        Number of units in each Liquid layer.
    dropout_rate:
        Dropout rate applied after each recurrent layer.
    learning_rate:
        Learning rate for the Adam optimizer.
    """

    model = Sequential()
    model.add(
        RNN(LiquidLayer(lnn_units), return_sequences=True, input_shape=input_shape)
    )
    model.add(Dropout(dropout_rate))
    model.add(RNN(LiquidLayer(lnn_units)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(16, activation="relu", kernel_regularizer=l2(0.01)))
    model.add(Dense(1))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse")

    return model


def train_model_with_rolling_window(
    model,
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    window_size=WINDOW_SIZE,
    step_size=1,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
):
    total_windows = (len(X_train) - window_size) // step_size
    progress_bar = tqdm(
        total=total_windows, desc=Fore.GREEN + "Training Progress", ncols=100
    )

    for window_num, start_idx in enumerate(
        range(0, len(X_train) - window_size, step_size), start=1
    ):
        end_idx = start_idx + window_size

        X_window_train = X_train[start_idx:end_idx]
        y_window_train = y_train[start_idx:end_idx]

        # Optionally pass validation data
        validation_data = (
            (X_val, y_val) if X_val is not None and y_val is not None else None
        )

        progress_bar.set_description(
            Fore.GREEN + f"Training Window {window_num}/{total_windows}"
        )

        model.fit(
            X_window_train,
            y_window_train,
            validation_data=validation_data,
            batch_size=batch_size,
            epochs=epochs,
            verbose=0,
        )

        progress_bar.update(1)

    progress_bar.close()
    return model


def manual_clone_model(model):
    model_copy = Sequential()
    for layer in model.layers:
        if isinstance(layer, RNN):
            rnn_layer = RNN(
                LiquidLayer(layer.cell.units),
                return_sequences=layer.return_sequences,
                input_shape=layer.input_shape[1:],
            )
            model_copy.add(rnn_layer)
        else:
            model_copy.add(layer.__class__.from_config(layer.get_config()))
    model_copy.set_weights(model.get_weights())
    return model_copy


def calculate_learning_curve(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    train_sizes=np.linspace(0.1, 1.0, 10),
):
    learning_curve_data = []
    for train_size in train_sizes:
        subset_size = int(train_size * X_train.shape[0])
        X_train_subset = X_train[:subset_size]
        y_train_subset = y_train[:subset_size]
        model_clone = manual_clone_model(model)
        model_clone.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss="mse")
        history = model_clone.fit(
            X_train_subset,
            y_train_subset,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            verbose=0,
        )
        train_loss = history.history["loss"][-1]
        val_loss = history.history["val_loss"][-1]
        learning_curve_data.append(
            {
                "Train Size": subset_size,
                "Train Loss": train_loss,
                "Validation Loss": val_loss,
            }
        )
        print(
            f"Train size: {subset_size}, Train loss: {train_loss}, Val loss: {val_loss}"
        )

    df_learning_curve = pd.DataFrame(learning_curve_data)
    df_learning_curve.to_csv("learning_curve.csv", index=False)
    print("Learning curve data saved to learning_curve.csv")

    return df_learning_curve


def generate_predictions(df, window_size, mode="train_test"):
    # Split data into train and validation sets before creating rolling windows
    if mode == "train_test":
        train_size = int(len(df) * TRAIN_SIZE_RATIO)
    elif mode == "production":
        train_size = int(len(df) * 0.9)

    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]

    # Create rolling windows after splitting the data
    X_train, y_train = create_rolling_window_sequences(
        train_df["Close"].values, window_size
    )
    X_val, y_val = create_rolling_window_sequences(val_df["Close"].values, window_size)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_liquid_model(input_shape=input_shape)
    model = train_model_with_rolling_window(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        window_size=window_size,
        step_size=1,  # Ensures no steps are skipped during window generation
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
    )

    # Prediction logic
    if mode == "production":
        # In production mode, predict future `PREDICTION_DAYS` based on last data
        last_data = X_train[
            -window_size:
        ]  # Use the last sequence to predict the future
        predictions = []

        for _ in range(PREDICTION_DAYS):
            next_pred = model.predict(last_data.reshape(1, window_size, 1)).flatten()
            predictions.append(next_pred[0])
            last_data = np.append(last_data[1:], next_pred).reshape(window_size, 1)

        # Create a DataFrame for future predictions
        last_date = df.index[-1]
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1), periods=PREDICTION_DAYS
        )
        future_df = pd.DataFrame({"predicted": predictions}, index=future_dates)
        df = pd.concat([df, future_df], axis=0)

    else:
        y_pred = model.predict(X_val).flatten()
        pred_dates = val_df.index[
            window_size:
        ]  # Match prediction dates to the validation data
        df_predictions = pd.DataFrame(data={"Date": pred_dates, "predicted": y_pred})
        df_predictions.set_index("Date", inplace=True)
        df = df.join(df_predictions, how="left")

    # Learning curve
    train_sizes = np.linspace(0.1, 1.0, 10)
    df_learning_curve = calculate_learning_curve(
        model, X_train, y_train, X_val, y_val, train_sizes=train_sizes
    )

    return df, df_learning_curve


if __name__ == "__main__":
    setup()
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    window_size = WINDOW_SIZE
    mode = "production"  # Switch to production for 2-week prediction
    df_with_predictions, df_learning_curve = generate_predictions(df, window_size, mode)
    print(df_with_predictions.head())
    print(df_learning_curve)

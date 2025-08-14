import os  # Import os to use environment variables
import sys
import time
import numpy as np
import pandas as pd
from loguru import logger
from sentimental_cap_predictor.sentimental_cap_predictor.sentimental_cap_predictor.config import INTERIM_DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELING_DIR
from pathlib import Path
import warnings
import traceback
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load logging level from .env (defaults to 'INFO')
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logger.remove()
logger.add(sys.stdout, level=LOG_LEVEL)

# Add MODELING_DIR to the Python path
sys.path.append(str(MODELING_DIR))

# Import necessary modules
try:
    from preprocessing import clean_data, handle_missing_values, feature_engineering
    from time_series_deep_learner import build_liquid_model, train_model_with_rolling_window, calculate_learning_curve
    from sentiment_analysis import perform_sentiment_analysis
    from bias_predictions import bias_predictions_with_sentiment
except ImportError as e:
    logger.error(f"ImportError: {e}")
    sys.exit(1)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Model hyperparameters from .env
TRAIN_RATIO = float(os.getenv("TRAIN_RATIO", 0.8))
DEFAULT_PREDICTION_DAYS = int(os.getenv("PREDICTION_DAYS", 14))  # Default to 14 days if not set

# Function to print NaN information at different steps
def print_nan_info(df, step_name):
    nan_info = df.isna().sum().sum()
    logger.info(f"After {step_name}: {nan_info} NaN values.")
    logger.info(df.head())
    logger.info(f"Datetime index range after {step_name}: {df.index.min()} to {df.index.max()}")

# Core function for generating predictions
def generate_predictions(price_df, news_df, ticker, mode='train_test', prediction_days=None, interim_dir=Path(INTERIM_DATA_DIR), processed_dir=Path(PROCESSED_DATA_DIR)):
    logger.info(f"Starting prediction process for ticker: {ticker} in {mode} mode")

    # If prediction_days isn't provided, fallback to DEFAULT_PREDICTION_DAYS
    prediction_days = prediction_days or DEFAULT_PREDICTION_DAYS

    if price_df.empty:
        logger.error("Price DataFrame is empty. Exiting.")
        return None

    logger.info("Price DataFrame before preprocessing:")
    print_nan_info(price_df, "loading data")

    # Initialize MinMaxScaler and scale the 'close' prices
    scaler = MinMaxScaler()
    price_df['close'] = scaler.fit_transform(price_df[['close']])
    logger.info("MinMax scaling applied to 'close' prices.")
    print_nan_info(price_df, "scaling_close_prices")

    # Clean data
    price_df = clean_data(price_df)
    print_nan_info(price_df, "clean_data")

    # Handle missing values
    price_df = handle_missing_values(price_df)
    print_nan_info(price_df, "handle_missing_values")

    # Perform feature engineering
    logger.info("Performing feature engineering.")
    price_df = feature_engineering(price_df)
    print_nan_info(price_df, "feature_engineering")

    # Convert date column to datetime and set as index
    if 'date' in price_df.columns:
        logger.debug("Converting 'date' column to datetime format.")
        price_df['date'] = pd.to_datetime(price_df['date'], errors='coerce')
        price_df.dropna(subset=['date'], inplace=True)
        price_df.set_index('date', inplace=True)
        print_nan_info(price_df, "datetime_conversion")
    else:
        logger.error("No 'date' column found in the price DataFrame. Exiting.")
        return None

    # Set frequency and interpolate missing values
    if price_df.index.freq is None:
        logger.debug("Setting frequency to 'D' for datetime index.")
        price_df = price_df.asfreq('D')
        logger.info(f"Frequency set to 'D'.")
        print_nan_info(price_df, "frequency_set")

        price_df = price_df.interpolate(method='time')
        price_df.fillna(method='bfill', inplace=True)
        price_df.fillna(method='ffill', inplace=True)
        print_nan_info(price_df, "interpolate_after_frequency_set")

    # Perform sentiment analysis on the news DataFrame
    logger.info("Performing sentiment analysis on the news articles.")
    sentiment_df = perform_sentiment_analysis(news_df)

    # Handle train_test and production mode separately
    if mode == 'train_test':
        # Split data into train and test sets
        logger.info("Splitting data into train and test sets.")
        train_size = int(len(price_df) * TRAIN_RATIO)
        train_data = price_df[:train_size]
        test_data = price_df[train_size:]
        logger.debug(f"Training data size: {len(train_data)}, Testing data size: {len(test_data)}")
        print_nan_info(train_data, "train_data_split")
        print_nan_info(test_data, "test_data_split")
    elif mode == 'production':
        train_data = price_df
        test_data = pd.DataFrame(index=pd.date_range(start=price_df.index[-1] + pd.Timedelta(days=1), periods=prediction_days, freq='D'))
        logger.info(f"Production mode: preparing to forecast the next {prediction_days} days.")

    if len(train_data) == 0:
        logger.error("Training data is empty. Exiting.")
        return None

    last_true_value = price_df['close'].iloc[-1]
    logger.debug(f"Last true value in the dataset: {last_true_value}")

    # LNN Modeling (replaces LSTM)
    logger.info("Applying Liquid Neural Network for non-linear feature extraction.")
    try:
        timesteps = 1  # Ensure this matches your LNN input shape requirements

        # Reshape the data for LNN
        X_train = np.reshape(train_data['close'].values, (train_data.shape[0], timesteps, 1))
        y_train = train_data['close'].values

        if mode == 'train_test':
            X_test = np.reshape(test_data['close'].values, (test_data.shape[0], timesteps, 1))
            y_test = test_data['close'].values
        else:
            X_test = None
            y_test = None

        # Define the model
        model = build_liquid_model(input_shape=(timesteps, 1))

        # Train the model with validation data (for train_test mode)
        if mode == 'train_test':
            val_size = int(len(X_train) * 0.2)
            X_val, y_val = X_train[-val_size:], y_train[-val_size:]

            history = train_model_with_rolling_window(model, X_train[:-val_size], y_train[:-val_size], X_val, y_val, window_size=100)

            logger.info("Calculating learning curve.")
            learning_curve_data = calculate_learning_curve(model, X_train[:-val_size], y_train[:-val_size], X_val, y_val)

            plt.figure(figsize=(10, 6))
            plt.plot(learning_curve_data['Train Size'], learning_curve_data['Train Loss'], label='Training Loss')
            plt.plot(learning_curve_data['Train Size'], learning_curve_data['Validation Loss'], label='Validation Loss')
            plt.title('LNN Learning Curve')
            plt.xlabel('Train Size')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)

            learning_curve_plot_path = processed_dir / f"{ticker}_learning_curve.png"
            plt.savefig(learning_curve_plot_path)
            logger.info(f"Learning curve plot saved to {learning_curve_plot_path}")

            # Predict on the test data for train_test mode
            predictions = model.predict(X_test).flatten()
            test_data['LNN_Predictions'] = predictions
            price_df.loc[test_data.index, 'LNN_Predictions'] = predictions

            # Apply bias from sentiment analysis on the test data
            logger.info("Biasing predictions with sentiment data.")
            test_data = bias_predictions_with_sentiment(test_data, sentiment_df)
            price_df.update(test_data)

        if mode == 'production':
            # In production mode, we don't have validation data, so we just train the model
            train_model_with_rolling_window(model, X_train, y_train)

            # Use the last `timesteps` data points from the training data to predict the future
            last_data = X_train[-timesteps:]  # Get the last few days (matching timesteps)
            predictions = []

            # Loop to predict the next `prediction_days` days
            for _ in range(prediction_days):
                # Predict the next day
                next_pred = model.predict(last_data.reshape(1, timesteps, 1)).flatten()
                predictions.append(next_pred[0])
                
                # Shift the window: remove the first and add the predicted value at the end
                last_data = np.append(last_data[1:], next_pred).reshape(timesteps, 1)

            # Generate future dates starting from the last date in price_df
            last_date = price_df.index[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=prediction_days, freq='D')

            # Create a new DataFrame to hold future dates and predictions
            future_df = pd.DataFrame(index=future_dates, data=predictions, columns=['LNN_Predictions'])

            # Append the predictions to the original price_df
            price_df = pd.concat([price_df, future_df])

            # Apply bias from sentiment analysis on the future data
            logger.info("Biasing future predictions with sentiment data.")
            future_df = bias_predictions_with_sentiment(future_df, sentiment_df)
            price_df.update(future_df)

        print_nan_info(price_df, "deep_learning_predictions")

    except Exception as e:
        logger.error(f"Error during LNN model prediction: {e}")
        logger.error(traceback.format_exc())
        return None  # Exit early if there's an error in LNN predictions

    # Invert scaling on all columns
    logger.info("Inverting MinMax scaling on all columns.")
    df_final = pd.DataFrame({
        'Date': price_df.index,  # Ensure the date is included
        'TrueValues': scaler.inverse_transform(price_df[['close']]).flatten(),
        'LNN_Predictions': scaler.inverse_transform(price_df[['LNN_Predictions']]).flatten(),
    }, index=price_df.index)

    logger.info(f"Final DataFrame prepared with shape: {df_final.shape}")

    # Plotting predictions
    plt.figure(figsize=(14, 7))
    plt.plot(df_final['Date'], df_final['TrueValues'], label='True Values', color='blue')
    plt.plot(df_final['Date'], df_final['LNN_Predictions'], label='LNN Predictions', color='green')
    plt.legend()
    plt.title(f'{ticker} - Predictions')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)
    plot_path = processed_dir / f"{ticker}_predictions_plot.png"
    try:
        plt.savefig(plot_path)
        logger.info(f"Plot saved to {plot_path}")
    except Exception as e:
        logger.error(f"Error saving plot to file: {e}")
        logger.error(traceback.format_exc())
        return None

    # Saving the final DataFrame to CSV
    csv_output_path = processed_dir / f"{ticker}_final_predictions.csv"
    try:
        df_final.to_csv(csv_output_path, index=False)  # Save the DataFrame with the 'Date' column
        logger.info(f"Final predictions saved to {csv_output_path}")
    except Exception as e:
        logger.error(f"Error saving predictions to CSV file: {e}")
        logger.error(traceback.format_exc())
        return None

    return df_final

if __name__ == "__main__":
    start_time = time.time()  # Record start time

    if len(sys.argv) < 2:
        logger.error("No ticker provided. Usage: python script.py <TICKER> <MODE> [PREDICTION_DAYS]")
        sys.exit(1)

    ticker = sys.argv[1].upper()
    mode = sys.argv[2].lower() if len(sys.argv) > 2 else 'train_test'
    prediction_days = int(sys.argv[3]) if len(sys.argv) > 3 and mode == 'production' else DEFAULT_PREDICTION_DAYS

    try:
        # Load price data
        df_path = Path(RAW_DATA_DIR) / f"{ticker}.feather"
        price_df = pd.read_feather(df_path)
        logger.info(f"Loaded price data from {df_path}, shape: {price_df.shape}")
    except Exception as e:
        logger.error(f"Error loading Feather file for {ticker}: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

    try:
        # Load news data
        news_path = Path(RAW_DATA_DIR) / f"{ticker}_news.feather"
        news_df = pd.read_feather(news_path)
        logger.info(f"Loaded news data from {news_path}, shape: {news_df.shape}")
    except Exception as e:
        logger.error(f"Error loading news Feather file for {ticker}: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

    try:
        # Generate predictions using the main function
        result = generate_predictions(price_df, news_df, ticker, mode, prediction_days)
        if result is not None:
            logger.info("Result DataFrame:")
            print(result.head())
        else:
            logger.error("No result was generated.")
    except Exception as e:
        logger.error(f"An error occurred in the main function: {e}")
        logger.error(traceback.format_exc())

    # Calculate total time taken
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Total runtime: {total_time:.2f} seconds")

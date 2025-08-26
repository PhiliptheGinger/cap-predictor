import os
import sys
import time
import pandas as pd
import numpy as np
from loguru import logger
from .config import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELING_DIR, ENABLE_TICKER_LOGS
from pathlib import Path
import warnings
import traceback

from .preprocessing import preprocess_price_data, merge_data
from .model_training import train_and_predict
from .data_bundle import DataBundle
from .dataset import load_data_bundle

# Load logging level from .env (defaults to 'INFO')
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logger.remove()
logger.add(sys.stdout, level=LOG_LEVEL)

# Add MODELING_DIR to the Python path
sys.path.append(str(MODELING_DIR))

try:
    from modeling.sentiment_analysis import perform_sentiment_analysis
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
    if not ENABLE_TICKER_LOGS:
        return
    nan_info = df.isna().sum().sum()
    logger.info(f"After {step_name}: {nan_info} NaN values.")
    logger.info(df.head())
    logger.info(f"Datetime index range after {step_name}: {df.index.min()} to {df.index.max()}")

# Core function for generating predictions
def generate_predictions(bundle: DataBundle, ticker, mode='train_test', prediction_days=None, processed_dir=Path(PROCESSED_DATA_DIR)):
    if ENABLE_TICKER_LOGS:
        logger.info(f"Starting prediction process for ticker: {ticker} in {mode} mode")

    price_df = bundle.prices.copy()
    news_df = bundle.sentiment.copy() if bundle.sentiment is not None else pd.DataFrame(index=price_df.index)

    # If prediction_days isn't provided, fallback to DEFAULT_PREDICTION_DAYS
    prediction_days = prediction_days or DEFAULT_PREDICTION_DAYS

    if price_df.empty:
        logger.error("Price DataFrame is empty. Exiting.")
        return None

    if ENABLE_TICKER_LOGS:
        logger.info("Price DataFrame before preprocessing:")
    print_nan_info(price_df, "loading data")

    price_df, scaler = preprocess_price_data(price_df)
    print_nan_info(price_df, "preprocess_price_data")

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
        if ENABLE_TICKER_LOGS:
            logger.info(f"Frequency set to 'D'.")
        print_nan_info(price_df, "frequency_set")

        price_df = price_df.interpolate(method='time')
        price_df.fillna(method='bfill', inplace=True)
        price_df.fillna(method='ffill', inplace=True)
        print_nan_info(price_df, "interpolate_after_frequency_set")

    # Perform sentiment analysis on the news DataFrame
    if ENABLE_TICKER_LOGS:
        logger.info("Performing sentiment analysis on the news articles.")
    sentiment_df = perform_sentiment_analysis(news_df)

    # Handle train_test and production mode separately
    if mode == 'train_test':
        # Split data into train and test sets
        if ENABLE_TICKER_LOGS:
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
        if ENABLE_TICKER_LOGS:
            logger.info(f"Production mode: preparing to forecast the next {prediction_days} days.")

    if len(train_data) == 0:
        logger.error("Training data is empty. Exiting.")
        return None

    try:
        price_df = train_and_predict(price_df, train_data, test_data, mode, prediction_days, sentiment_df)
        print_nan_info(price_df, "deep_learning_predictions")
    except Exception as e:
        logger.error(f"Error during LNN model prediction: {e}")
        logger.error(traceback.format_exc())
        return None  # Exit early if there's an error in LNN predictions

    # Load existing processed data if exists
    csv_output_path = processed_dir / f"{ticker}_{mode}_predictions.csv"
    if csv_output_path.exists():
        if ENABLE_TICKER_LOGS:
            logger.info(f"Loading existing processed data from {csv_output_path}.")
        existing_df = pd.read_csv(csv_output_path)
    else:
        existing_df = pd.DataFrame()

    # Invert scaling on all columns and create a new DataFrame
    df_final = pd.DataFrame({
        'Date': price_df.index,  # Ensure the date is included
        'actual': scaler.inverse_transform(price_df[['close']]).flatten(),
        'predicted': scaler.inverse_transform(price_df[['predicted']]).flatten(),
    }, index=price_df.index)

    # Merge new predictions with the existing data
    df_final = merge_data(existing_df, df_final, merge_on='Date')

    if ENABLE_TICKER_LOGS:
        logger.info(f"Final DataFrame prepared with shape: {df_final.shape}")

    # Compute evaluation metrics
    try:
        valid_df = df_final.dropna(subset=['actual', 'predicted'])
        rmse = np.sqrt(((valid_df['actual'] - valid_df['predicted']) ** 2).mean())
        mape = (np.abs((valid_df['actual'] - valid_df['predicted']) / valid_df['actual']).replace([np.inf, -np.inf], np.nan).dropna()).mean() * 100
        if ENABLE_TICKER_LOGS:
            logger.info(f"RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
    except Exception as e:
        logger.error(f"Error computing metrics: {e}")
        rmse, mape = None, None

    # Saving the final DataFrame to CSV
    try:
        df_final.to_csv(csv_output_path, index=False)  # Save the DataFrame with the 'Date' column
        if ENABLE_TICKER_LOGS:
            logger.info(f"Final predictions saved to {csv_output_path}")
    except Exception as e:
        logger.error(f"Error saving predictions to CSV file: {e}")
        logger.error(traceback.format_exc())
        return None

    # Save metrics for future analysis
    if rmse is not None and mape is not None:
        metrics_path = processed_dir / f"{ticker}_{mode}_metrics.csv"
        metrics_df = pd.DataFrame([
            {
                'Timestamp': pd.Timestamp.now(),
                'RMSE': rmse,
                'MAPE': mape,
            }
        ])
        try:
            metrics_df.to_csv(metrics_path, mode='a', header=not metrics_path.exists(), index=False)
            if ENABLE_TICKER_LOGS:
                logger.info(f"Metrics saved to {metrics_path}")
        except Exception as e:
            logger.error(f"Error saving metrics to file: {e}")

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
        bundle = load_data_bundle(ticker)
    except Exception as e:
        logger.error(f"Error loading data bundle for {ticker}: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

    try:
        # Generate predictions using the main function
        result = generate_predictions(bundle, ticker, mode, prediction_days)
        if result is not None:
            if ENABLE_TICKER_LOGS:
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
    if ENABLE_TICKER_LOGS:
        logger.info(f"Total runtime: {total_time:.2f} seconds")

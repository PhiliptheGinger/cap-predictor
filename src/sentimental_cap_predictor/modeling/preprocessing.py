from datetime import datetime as dt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
from loguru import logger

# Configure logger to write to a file if needed, for example:
logger.add("errors.log", format="{time} {level} {message}", level="ERROR")

def clean_data(data):
    try:
        """
        Cleans the data by removing any duplicates and unnecessary columns (if applicable).
        Converts numpy.ndarray columns to lists for compatibility with Pandas operations.
        """
        # Convert numpy.ndarray columns to lists
        for col in data.columns:
            if data[col].apply(lambda x: isinstance(x, np.ndarray)).any():
                data[col] = data[col].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

        # Remove duplicates
        data = data.drop_duplicates()

        return data
    except Exception as e:
        logger.exception("Error in clean_data: ")
        raise e

def handle_missing_values(data):
    try:
        """
        Handles missing values in the dataset by either filling them or dropping them based on the strategy.
        """
        # Example strategy: Fill missing values using forward fill, then backward fill
        data = data.fillna(method='ffill').fillna(method='bfill')

        return data
    except Exception as e:
        logger.exception("Error in handle_missing_values: ")
        raise e

def check_stationarity(data, column, significance=0.05):
    """
    Performs the Augmented Dickey-Fuller test to check for stationarity.
    
    Returns True if the series is stationary, otherwise False.
    """
    try:
        result = adfuller(data[column.lower()].dropna())  # Ensure column name is lowercase
        p_value = result[1]
        return p_value < significance  # Returns True if the series is stationary
    except KeyError:
        logger.exception(f"Error checking stationarity for column '{column}': Column not found.")
        raise

def difference_data(data, column):
    try:
        """
        Applies differencing to make the time series stationary.
        """
        data[f'{column}_diff'] = data[column].diff().dropna()
        return data.dropna(subset=[f'{column}_diff'])
    except Exception as e:
        logger.exception(f"Error in difference_data for column '{column}': ")
        raise e

def detrend_data(data, column):
    try:
        """
        Detrends the data by subtracting the rolling mean.
        """
        rolling_mean = data[column].rolling(window=12, center=False).mean()
        data[f'{column}_detrended'] = data[column] - rolling_mean
        return data.dropna()
    except Exception as e:
        logger.exception(f"Error in detrend_data for column '{column}': ")
        raise e

def log_transform(data, column):
    try:
        """
        Applies log transformation to stabilize the variance.
        """
        data[f'{column}_log'] = np.log(data[column])
        return data
    except Exception as e:
        logger.exception(f"Error in log_transform for column '{column}': ")
        raise e

def feature_engineering(data):
    try:
        """
        Adds new features to the dataset that may be useful for the model.
        """
        # Standardize column names to lowercase
        data.columns = data.columns.str.lower()

        # Example: Create moving averages using standardized column names
        data['ma_10'] = data['close'].rolling(window=10).mean()
        data['ma_30'] = data['close'].rolling(window=30).mean()

        # Example: Create difference features (lagged differences)
        data['diff_close'] = data['close'].diff()

        # Example: Create additional time-based features if the 'date' column exists
        if 'date' in data.columns:
            data['day_of_week'] = data['date'].dt.dayofweek
            data['month'] = data['date'].dt.month

        # Drop rows with NaN values introduced by feature engineering (e.g., from rolling windows)
        data = data.dropna()

        return data
    except Exception as e:
        logger.exception("Error in feature_engineering: ")
        raise e

def min_max_scale(data, feature_range=(0, 1)):
    try:
        """
        Scales the data using Min-Max scaling, excluding non-numerical columns.
        """
        # Select only numerical columns
        data_to_scale = data.select_dtypes(include=[np.number])

        scaler = MinMaxScaler(feature_range=feature_range)
        scaled_data = scaler.fit_transform(data_to_scale)

        # Create DataFrame with scaled data
        scaled_df = pd.DataFrame(scaled_data, index=data.index, columns=data_to_scale.columns)

        # Add back any non-scaled columns (e.g., Date)
        non_scaled_cols = data.columns.difference(scaled_df.columns)
        scaled_df[non_scaled_cols] = data[non_scaled_cols]

        return scaled_df, scaler
    except Exception as e:
        logger.exception("Error in min_max_scale: ")
        raise e

def preprocess_data(data, column='close', apply_log_transform=False, detrend=False):
    try:
        """
        Complete preprocessing pipeline that cleans, handles missing values, performs feature engineering,
        and scales the data. It also includes options for log transformation and detrending.
        """
        data = clean_data(data)
        data = handle_missing_values(data)

        # Check for stationarity
        is_stationary = check_stationarity(data, column)

        if not is_stationary:
            # Apply log transformation if specified
            if apply_log_transform:
                data = log_transform(data, column)
                column = f'{column}_log'

            # Apply detrending if specified
            if detrend:
                data = detrend_data(data, column)
                column = f'{column}_detrended'

            # Apply differencing if the series is still non-stationary
            if not check_stationarity(data, column):
                data = difference_data(data, column)
                column = f'{column}_diff'

        data = feature_engineering(data)
        scaled_data, scaler = min_max_scale(data)

        return scaled_data, scaler
    except Exception as e:
        logger.exception("Error in preprocess_data: ")
        raise e

# preprocessing.py

from datetime import datetime as dt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def generate_brownian_motion(length, start_price=100, drift=0.0001, volatility=0.01):
    """Generates a Brownian motion time series."""
    np.random.seed(0)
    returns = np.random.normal(loc=drift, scale=volatility, size=length)
    price = start_price * np.exp(np.cumsum(returns))
    return price

def min_max_scale(data, feature_range=(0, 1)):
    """Scales the data using Min-Max scaling, excluding non-numerical columns."""
    # Exclude non-numerical columns (e.g., Date) by selecting only numerical data
    data_to_scale = data.select_dtypes(include=[float, int])
    
    # Initialize the scaler
    scaler = MinMaxScaler(feature_range=feature_range)
    
    # Perform the scaling on the numerical data
    scaled_data = scaler.fit_transform(data_to_scale)
    
    # Create a DataFrame with scaled data, retaining the original index and column names
    scaled_df = pd.DataFrame(scaled_data, index=data.index, columns=data_to_scale.columns)
    
    # Add back any non-scaled columns (e.g., Date)
    for col in data.columns:
        if col not in scaled_df.columns:
            scaled_df[col] = data[col]
    
    return scaled_df, scaler


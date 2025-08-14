import sys
import logging
import pytest
import pandas as pd
import  warnings
from warnings import catch_warnings, simplefilter
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
from statsmodels.tools.sm_exceptions import ValueWarning
from sklearn.metrics import mean_squared_error
from sentimental_cap_predictor.sentimental_cap_predictor.sentimental_cap_predictor.config import MODELING_DIR
from .sarima_calculator import (
    fit_sarima,
    predict_seasons_with_intervals,
    evaluate_sarima,
    perform_hyperparameter_optimization,
)

# Add MODELING_DIR to the Python path
sys.path.append(str(MODELING_DIR))

# Setup logging for tests
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@pytest.fixture
def tsla_data():
    """Load the TSLA data from the feather file and ensure proper indexing."""
    file_path = r"D:\Programming Projects\CAP\sentimental_cap_predictor\sentimental_cap_predictor\data\raw\TSLA.feather"
    df = pd.read_feather(file_path)
    
    # Print column names for debugging
    print("Columns in the DataFrame:", df.columns)

    # Check if 'Date' column exists and set it as the index
    if 'Date' in df.columns:
        df.set_index('Date', inplace=True)
    else:
        raise KeyError("The 'Date' column is not present in the DataFrame. Please check the column names.")
    
    # Handle NaN values by forward filling them
    df['Close'] = df['Close'].fillna(method='ffill')
    
    # Return only the 'Close' column
    return df['Close']


def test_fit_sarima(tsla_data):
    """Test fitting a SARIMA model."""
    order = (3, 0, 2)
    seasonal_order = (0, 1, 1, 12)
    logging.info(f"Testing fit_sarima with order={order} and seasonal_order={seasonal_order}")
    
    with catch_warnings():
        simplefilter("error")
        model = fit_sarima(tsla_data, order, seasonal_order, trend="c", enforce_stationarity=False, enforce_invertibility=True)
    
    assert isinstance(model, SARIMAXResultsWrapper), "Model fitting failed, not an instance of SARIMAXResultsWrapper"

def test_predict_seasons_with_intervals(tsla_data):
    """Test prediction with confidence intervals using SARIMA."""
    order = (3, 0, 2)
    seasonal_order = (0, 1, 1, 12)
    logging.info(f"Testing predict_seasons_with_intervals with order={order} and seasonal_order={seasonal_order}")
    
    with catch_warnings():
        simplefilter("error")
        model = fit_sarima(tsla_data, order, seasonal_order, trend="c", enforce_stationarity=False, enforce_invertibility=True)
    
    forecast, conf_int = predict_seasons_with_intervals(model, steps=10)
    
    logging.info(f"Forecast: {forecast}")
    logging.info(f"Confidence intervals: {conf_int}")
    
    assert len(forecast) == 10, "Forecast length is incorrect"
    assert conf_int.shape == (10, 2), "Confidence interval shape is incorrect"
    
    lower_bounds = conf_int.iloc[:, 0]
    upper_bounds = conf_int.iloc[:, 1]
    assert all(lower_bounds <= forecast) and all(forecast <= upper_bounds), "Forecast is not within confidence intervals"

def test_evaluate_sarima(tsla_data):
    """Test SARIMA model evaluation."""
    train_size = int(len(tsla_data) * 0.8)
    train_data = tsla_data[:train_size]
    test_data = tsla_data[train_size:]
    
    params = {
        "p": 3, "d": 0, "q": 2,
        "P": 0, "D": 1, "Q": 1, "S": 12,
        "trend": "c",
        "enforce_stationarity": False,
        "enforce_invertibility": True
    }
    
    logging.info(f"Testing evaluate_sarima with params={params}")
    mse = evaluate_sarima(params, train_data, test_data)
    
    logging.info(f"MSE: {mse}")
    
    assert mse >= 0, "Mean Squared Error should be non-negative"
    assert not pd.isnull(mse), "MSE is NaN"
    assert mse < 1e6, f"Unusually high Mean Squared Error: {mse}"

@pytest.mark.parametrize("sarima_params", [
    {
        "p": [2, 3, 4],
        "d": [0],
        "q": [1, 2, 3],
        "P": [0],
        "D": [1],
        "Q": [0, 1],
        "S": [12],
        "trend": ["c"],
        "enforce_stationarity": [False],
        "enforce_invertibility": [True]
    },
    # Add more parameter configurations to test edge cases
])
def test_perform_hyperparameter_optimization(tsla_data, tmp_path, sarima_params):
    """Test hyperparameter optimization process."""
    train_size = int(len(tsla_data) * 0.8)
    train_data = tsla_data[:train_size]
    test_data = tsla_data[train_size:]
    
    logging.info(f"Testing perform_hyperparameter_optimization with sarima_params={sarima_params}")
    best_params = perform_hyperparameter_optimization(train_data, test_data, sarima_params, timeout_minutes=1, cache_dir=tmp_path)
    
    assert isinstance(best_params, dict), "Best parameters should be returned as a dictionary"
    assert all(key in best_params for key in ["p", "d", "q", "P", "D", "Q", "S", "trend", "enforce_stationarity", "enforce_invertibility"]), \
        "Some SARIMA parameters are missing in the best_params"

def test_invalid_configuration_handling(tsla_data):
    """Test handling of invalid SARIMA configurations."""
    train_size = int(len(tsla_data) * 0.8)
    train_data = tsla_data[:train_size]
    test_data = tsla_data[train_size:]

    invalid_params = {
        "p": 0, "d": 0, "q": 3,
        "P": 0, "D": 1, "Q": 3, "S": 3,
        "trend": "c",
        "enforce_stationarity": False,
        "enforce_invertibility": True
    }
    
    logging.info(f"Testing invalid configuration handling with params={invalid_params}")
    with pytest.raises(ValueError, match=".*Invalid model.*|.*error.*"):
        evaluate_sarima(invalid_params, train_data, test_data)

def test_empty_data_handling():
    """Test handling of empty datasets."""
    empty_data = pd.Series(dtype=float)
    
    logging.info("Testing with empty dataset")
    with pytest.raises(ValueError, match=".*zero-size array.*"):
        fit_sarima(empty_data, (1, 1, 1), (1, 1, 1, 12), trend="c", enforce_stationarity=False, enforce_invertibility=True)

def test_small_data_handling():
    """Test handling of very small datasets."""
    small_data = pd.Series([1.0, 1.5, 2.0], index=pd.date_range(start="2020-01-01", periods=3, freq="D"))
    
    logging.info("Testing with small dataset")
    with catch_warnings(record=True) as w:
        simplefilter("always")
        # Capture both UserWarning and ValueWarning to catch warnings related to small datasets
        warnings.simplefilter("always", UserWarning)
        warnings.simplefilter("always", ValueWarning)
        
        model = fit_sarima(small_data, (1, 1, 1), (1, 1, 1, 12), trend="c", enforce_stationarity=False, enforce_invertibility=True)
        
        # Log any warnings
        for warning in w:
            logging.warning(f"Warning raised: {warning.message}")
        
        # Check if any warnings were raised, especially related to small datasets
        assert len(w) > 0, "Expected warning for small dataset"
        assert isinstance(model, SARIMAXResultsWrapper), "Model fitting failed for small dataset"


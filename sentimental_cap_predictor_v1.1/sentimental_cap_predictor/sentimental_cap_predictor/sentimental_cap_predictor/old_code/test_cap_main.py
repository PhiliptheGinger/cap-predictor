import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import importlib.util
import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the path to the module
module_path = 'D:/Programming Projects/CAP/CAP_PREDICTOR/stable/cap_main_vs2.4.py'

# Load the module dynamically
spec = importlib.util.spec_from_file_location("cap_main_vs2_4", module_path)
cap_main = importlib.util.module_from_spec(spec)
sys.modules["cap_main_vs2_4"] = cap_main
spec.loader.exec_module(cap_main)

@pytest.fixture
def example_data():
    """Fixture to provide example data for testing."""
    train_data = pd.Series(np.random.randn(100), index=pd.date_range('2021-01-01', periods=100))
    test_data = pd.Series(np.random.randn(30), index=pd.date_range('2021-04-11', periods=30))
    sarima_params = {
        'p': [0, 1, 2],
        'd': [0, 1],
        'q': [0, 1, 2],
        'P': [0, 1],
        'D': [0, 1],
        'Q': [0, 1],
        'S': [12]
    }
    return train_data, test_data, sarima_params

def test_load_data_from_yfinance():
    data = cap_main.load_data_from_yfinance('AAPL', '2021-01-01', '2021-12-31')
    assert data is not None
    assert isinstance(data, pd.DataFrame)

def test_adjust_predictions_based_on_sentiment():
    predictions = np.array([100, 200, 300])
    sentiment_score = 0.05  # Assume an arbitrary sentiment score
    adjusted_predictions = cap_main.adjust_predictions_based_on_sentiment(predictions, sentiment_score)
    assert len(adjusted_predictions) == len(predictions)
    assert np.all(adjusted_predictions > predictions)  # Since sentiment_score is positive

def test_calculate_metrics():
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    num_predictors = 3
    metrics = cap_main.calculate_metrics(y_true, y_pred, num_predictors)
    assert 'MAE' in metrics
    assert 'MSE' in metrics
    assert 'RMSE' in metrics
    assert 'R2' in metrics
    assert 'Adjusted R2' in metrics
    assert 'Pearson' in metrics
    assert 'MAPE' in metrics

def test_plot_results(example_data):
    train_data, test_data, _ = example_data
    sarima_predictions = pd.Series(np.random.randn(len(test_data)), index=test_data.index)
    lnn_predictions = pd.Series(np.random.randn(len(test_data)), index=test_data.index)
    transformed_brownian_motion = pd.Series(np.random.randn(len(test_data)), index=test_data.index)
    transformed_lnn_predictions = pd.Series(np.random.randn(len(test_data)), index=test_data.index)
    adjusted_combined_model = pd.Series(np.random.randn(len(test_data)), index=test_data.index)
    adjusted_lnn_pred = pd.Series(np.random.randn(len(test_data)), index=test_data.index)
    output_path = 'test_output.png'
    cap_main.plot_results(train_data, test_data, sarima_predictions, lnn_predictions, transformed_brownian_motion, transformed_lnn_predictions, adjusted_combined_model, adjusted_lnn_pred, output_path)
    assert os.path.exists(output_path)
    os.remove(output_path)

def test_plot_learning_curves():
    train_errors = np.random.randn(100)
    val_errors = np.random.randn(100)
    output_path = 'test_learning_curves.png'
    cap_main.plot_learning_curves(train_errors, val_errors, output_path)
    assert os.path.exists(output_path)
    os.remove(output_path)

@pytest.mark.skip(reason="Requires integration setup")
def test_main():
    stock = 'AAPL'
    start_date = '2010-01-01'
    end_date = '2020-12-31'
    output_graph_path = 'output_graph.png'
    train_ratio = 0.8
    timeout_minutes = 60
    sarima_params = {
        'p': [0, 1, 2],
        'd': [0, 1],
        'q': [0, 1, 2],
        'P': [0, 1],
        'D': [0, 1],
        'Q': [0, 1],
        'S': [12]
    }
    cap_main.analyze_stock(stock, None, None, None, start_date, end_date, output_graph_path, train_ratio, timeout_minutes, sarima_params)
    assert os.path.exists(output_graph_path)

if __name__ == "__main__":
    pytest.main()

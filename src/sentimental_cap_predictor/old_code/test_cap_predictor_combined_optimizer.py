import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import importlib.util
import sys
import logging

# Configure logging
LOG_FILE_PATH = 'D:\\Programming Projects\\CAP\\CAP_PREDICTOR\\stable\\cap_predictor_combined_optimizer_vs2.3.py'
LOG_OUTPUT_PATH = 'D:\\Programming Projects\\CAP\\CAP_PREDICTOR\\stable\\cap_predictor_log.log'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler(LOG_OUTPUT_PATH),
    logging.StreamHandler(sys.stdout)  # Ensure logging to console during pytest run
])
logger = logging.getLogger(__name__)

# Define the path to the module
module_path = 'D:/Programming Projects/CAP/CAP_PREDICTOR/stable/cap_predictor_combined_optimizer_vs2.3.py'

# Load the module dynamically
spec = importlib.util.spec_from_file_location("cap_predictor_combined_optimizer_vs2_3", module_path)
cap_predictor = importlib.util.module_from_spec(spec)
sys.modules["cap_predictor_combined_optimizer_vs2_3"] = cap_predictor
spec.loader.exec_module(cap_predictor)

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
    data = cap_predictor.load_data_from_yfinance('AAPL', '2021-01-01', '2021-12-31')
    assert data is not None
    assert isinstance(data, pd.DataFrame)

def test_generate_brownian_motion():
    length = 100
    bm = cap_predictor.generate_brownian_motion(length)
    assert len(bm) == length

def test_min_max_scale():
    data = pd.DataFrame(np.random.randn(100, 1), columns=['value'], index=pd.date_range('2021-01-01', periods=100))
    scaled_data, scaler = cap_predictor.min_max_scale(data)
    assert isinstance(scaled_data, pd.DataFrame)
    assert isinstance(scaler, MinMaxScaler)

def test_fit_sarima(example_data):
    train_data, _, _ = example_data
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 12)
    model = cap_predictor.fit_sarima(train_data, order, seasonal_order)
    assert model is not None

def test_predict_seasons_with_intervals(example_data):
    train_data, test_data, _ = example_data
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 12)
    model = cap_predictor.fit_sarima(train_data, order, seasonal_order)
    predictions, conf_int = cap_predictor.predict_seasons_with_intervals(model, len(test_data))
    assert len(predictions) == len(test_data)
    assert len(conf_int) == len(test_data)

def test_track_state_transitions_and_averages(example_data):
    train_data, _, _ = example_data
    state_patterns, averages = cap_predictor.track_state_transitions_and_averages(train_data)
    assert isinstance(state_patterns, dict)
    assert isinstance(averages, dict)

def test_calculate_transition_matrix(example_data):
    train_data, _, _ = example_data
    state_patterns, _ = cap_predictor.track_state_transitions_and_averages(train_data)
    transition_matrix, patterns = cap_predictor.calculate_transition_matrix(state_patterns)
    assert transition_matrix.shape[0] == transition_matrix.shape[1]
    assert len(patterns) == transition_matrix.shape[0]

def test_simulate_markov_chain(example_data):
    train_data, _, _ = example_data
    state_patterns, _ = cap_predictor.track_state_transitions_and_averages(train_data)
    transition_matrix, patterns = cap_predictor.calculate_transition_matrix(state_patterns)
    initial_probabilities = np.ones(len(patterns)) / len(patterns)
    states = cap_predictor.simulate_markov_chain(transition_matrix, initial_probabilities, steps=100)
    assert len(states) == 100

def test_perform_hyperparameter_optimization(example_data):
    train_data, test_data, sarima_params = example_data
    best_params = cap_predictor.perform_hyperparameter_optimization(train_data, test_data, sarima_params, timeout_minutes=1)
    assert isinstance(best_params, dict)

def test_fit_predict_lnn(example_data):
    train_data, test_data, _ = example_data
    predictions = cap_predictor.fit_predict_lnn(train_data, test_data, epochs=1)
    assert len(predictions) == len(test_data)

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
    cap_predictor.main(stock, start_date, end_date, output_graph_path, train_ratio, timeout_minutes, sarima_params)
    assert os.path.exists(output_graph_path)

if __name__ == "__main__":
    pytest.main()

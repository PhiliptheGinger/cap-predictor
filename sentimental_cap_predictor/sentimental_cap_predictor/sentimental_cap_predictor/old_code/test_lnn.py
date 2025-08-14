import pytest
import torch
import pandas as pd
from pathlib import Path
from .lnn import LiquidNeuralNetwork, train_liquid_nn, k_fold_cross_validation

@pytest.fixture
def load_data():
    processed_dir = Path("D:/Programming Projects/CAP/sentimental_cap_predictor/sentimental_cap_predictor/data/processed")
    raw_dir = Path("D:/Programming Projects/CAP/sentimental_cap_predictor/sentimental_cap_predictor/data/raw")
    
    # Load the AAPL final predictions CSV
    try:
        aapl_final_predictions = pd.read_csv(processed_dir / "AAPL_final_predictions.csv")
    except FileNotFoundError:
        pytest.fail("AAPL_final_predictions.csv not found in the processed data directory.")
    
    # Load the AAPL Feather file
    try:
        aapl_raw_data = pd.read_feather(raw_dir / "AAPL.feather")
    except FileNotFoundError:
        pytest.fail("AAPL.feather not found in the raw data directory.")
    
    return aapl_final_predictions, aapl_raw_data

@pytest.fixture
def mock_data():
    # Create a small mock dataset for testing
    data = {
        'RandomWalk_L2': [1.2, 2.3, float('nan'), 4.5, 5.6],
        'CIR': [2.1, float('nan'), 3.4, 4.8, 5.1],
        'GBM': [1.8, 2.9, 3.1, 4.2, 5.3],
        'JumpDiffusion': [float('nan'), 2.7, 3.5, 4.6, 5.7],
        'TrueValues': [10, 15, 20, 25, 30]
    }
    return pd.DataFrame(data)

def test_model_initialization():
    model = LiquidNeuralNetwork(input_dim=5, hidden_dim=32, output_dim=1, num_layers=1, dropout_rate=0.0, bidirectional=False)
    assert model.hidden_dim == 32
    assert model.num_layers == 1

def test_forward_pass(mock_data):
    model = LiquidNeuralNetwork(input_dim=5, hidden_dim=32, output_dim=1)
    X = torch.tensor(mock_data.drop('TrueValues', axis=1).fillna(0).values).float().unsqueeze(1)
    output = model(X)
    assert output.shape == (mock_data.shape[0], 1)

def test_nan_handling(mock_data):
    model = LiquidNeuralNetwork(input_dim=5, hidden_dim=32, output_dim=1)
    X = torch.tensor(mock_data.drop('TrueValues', axis=1).values).float().unsqueeze(1)
    
    # Check for NaN handling in forward pass
    output = model(X)
    assert not torch.isnan(output).any()

def test_training_loop(mock_data):
    model = LiquidNeuralNetwork(input_dim=5, hidden_dim=32, output_dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    X = torch.tensor(mock_data.drop('TrueValues', axis=1).fillna(0).values).float().unsqueeze(1)
    y = torch.tensor(mock_data['TrueValues'].values).float().unsqueeze(1)
    
    train_liquid_nn(model, optimizer, criterion, X, y, epochs=10)

def test_k_fold_cross_validation(load_data):
    aapl_final_predictions, aapl_raw_data = load_data
    selected_models = ['RandomWalk_L2', 'CIR', 'GBM', 'JumpDiffusion']

    # Perform k-fold cross-validation
    loss = k_fold_cross_validation(LiquidNeuralNetwork, aapl_final_predictions, selected_models, learning_rate=0.001, epochs=10)
    
    assert loss is not None
    assert isinstance(loss, float)

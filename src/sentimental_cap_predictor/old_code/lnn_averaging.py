import sys
import traceback
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm
from loguru import logger
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
import typer
from pathlib import Path
from sklearn.impute import KNNImputer

# Configure loguru to replace standard logging
logger.remove()
logger.add(sys.stderr, level="INFO")

app = typer.Typer()

# Hyperparameter space defined as a dictionary
hyperparam_space = {
    'learning_rate': Real(1e-5, 1e-2),
    'epochs': Integer(50, 200),
    'hidden_dim': Integer(32, 64),
    'weight_decay': Real(1e-6, 1e-3),
    'num_layers': Integer(1, 20),
    'dropout_rate': Real(0.0, 0.3),
    'bidirectional': Categorical([False, True])
}

# Function to optimize hyperparameters
def hyperparameter_optimization(df, sarima_predictions, k_folds=5, n_calls=10):
    available_models = ['RandomWalk_L2', 'CIR', 'GBM', 'JumpDiffusion']
    selected_models = available_models

    features = selected_models + ['day_of_week', 'month', 'day_of_month', 'quarter', 'year'] + \
               [f'fourier_sin_{i}' for i in range(1, 3)] + \
               [f'fourier_cos_{i}' for i in range(1, 3)]

    if 'SARIMA' not in df.columns:
        df['SARIMA'] = sarima_predictions

    selected_models_with_sarima = selected_models + ['SARIMA']

    def objective(params):
        # Unpack parameters from the list into a dictionary
        params_dict = {
            'learning_rate': params[0],
            'epochs': params[1],
            'hidden_dim': params[2],
            'weight_decay': params[3],
            'num_layers': params[4],
            'dropout_rate': params[5],
            'bidirectional': params[6]
        }

        learning_rate = float(params_dict['learning_rate'])
        epochs = int(params_dict['epochs'])
        hidden_dim = int(params_dict['hidden_dim'])
        weight_decay = float(params_dict['weight_decay'])
        num_layers = int(params_dict['num_layers'])
        dropout_rate = float(params_dict['dropout_rate'])
        bidirectional = bool(params_dict['bidirectional'])

        # Create a model instance
        model = LiquidNeuralNetwork(
            input_dim=len(features) + 1,  # +1 to include the SARIMA predictions
            hidden_dim=hidden_dim,
            output_dim=1,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            bidirectional=bidirectional
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        df_with_sarima = df.copy()
        df_with_sarima['SARIMA'] = sarima_predictions

        check_required_features(df_with_sarima, features + ['SARIMA'])

        val_loss = k_fold_cross_validation(
            model_class=LiquidNeuralNetwork,
            df=df_with_sarima,
            selected_models=selected_models_with_sarima,
            learning_rate=learning_rate,
            epochs=epochs,
            k=k_folds,
            weight_decay=weight_decay,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            bidirectional=bidirectional
        )

        logger.info(f"Validation loss: {val_loss}")
        return val_loss

    best_result = gp_minimize(
        func=objective,
        dimensions=list(hyperparam_space.values()),  # Convert dict values to list
        n_calls=n_calls,
        n_random_starts=5,
        random_state=42
    )

    # Convert best_result.x to a dictionary with named hyperparameters
    best_params = {name: value for name, value in zip(hyperparam_space.keys(), best_result.x)}

    logger.info(f"Best hyperparameters found: {best_params}")
    return best_params

def check_datetime_index(df, df_name="DataFrame"):
    if isinstance(df.index, pd.DatetimeIndex):
        logger.info(f"{df_name} has a DateTime index.")
    else:
        logger.warning(f"{df_name} does NOT have a DateTime index.")
    
    logger.info(f"First few rows of {df_name}:\n{df.head()}")
    logger.info(f"Columns in {df_name}: {list(df.columns)}")

def check_required_features(df, required_features):
    logger.info(f"Checking required features in DataFrame with columns: {list(df.columns)}")
    
    missing_features = [feature for feature in required_features if feature not in df.columns]
    if missing_features:
        logger.error(f"Missing features in the DataFrame: {missing_features}")
        raise ValueError(f"Missing features in the DataFrame: {missing_features}")
    else:
        logger.info("All required features are present in the DataFrame.")

class LiquidNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout_rate=0.0, bidirectional=False):
        super(LiquidNeuralNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        if num_layers == 1:
            dropout_rate = 0.0

        self.recurrent_layer = nn.LSTM(input_dim, int(hidden_dim), num_layers=num_layers, 
                                       dropout=dropout_rate, batch_first=True, bidirectional=bidirectional)
        direction_multiplier = 2 if bidirectional else 1
        self.fc_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim * direction_multiplier, output_dim)
        )
        logger.debug(f"LiquidNeuralNetwork initialized with input_dim: {input_dim}, hidden_dim: {hidden_dim}, output_dim: {output_dim}, num_layers: {num_layers}, dropout_rate: {dropout_rate}, bidirectional: {bidirectional}")

    def forward(self, x):
        logger.debug(f"Input tensor shape: {x.shape}")

        if torch.isnan(x).any():
            logger.warning("NaN values found in input tensor. Forward-filling NaNs.")
            x = self.forward_fill_nan(x)

        x = torch.nan_to_num(x)
        
        batch_size = x.size(0)
        num_directions = 2 if self.bidirectional else 1
        h0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)
        c0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)
        
        logger.debug(f"h0 shape: {h0.shape}, c0 shape: {c0.shape}")
        
        x, _ = self.recurrent_layer(x, (h0, c0))
        logger.debug(f"Output tensor shape after LSTM: {x.shape}")
        
        x = self.fc_layer(x)
        logger.debug(f"Output tensor shape after fully connected layer: {x.shape}")
        
        return x

    def forward_fill_nan(self, x):
        for i in range(1, x.size(1)):
            nan_mask = torch.isnan(x[:, i, :])
            x[:, i, :][nan_mask] = x[:, i - 1, :][nan_mask]
        return x

def add_fourier_terms(df, period, n_terms=2):
    try:
        logger.debug(f"Adding Fourier terms with period: {period}, n_terms: {n_terms}")
        
        if isinstance(df, pd.Series):
            df = df.to_frame()

        if df.empty:
            logger.error("Input DataFrame is empty. Cannot add Fourier terms.")
            raise ValueError("Input DataFrame is empty.")
        
        original_index = df.index
        time = np.arange(len(df))
        
        for i in range(1, n_terms + 1):
            sin_term = f'fourier_sin_{i}'
            cos_term = f'fourier_cos_{i}'
            if sin_term not in df.columns and cos_term not in df.columns:
                df[sin_term] = np.sin(2 * np.pi * i * time / period)
                df[cos_term] = np.cos(2 * np.pi * i * time / period)
                logger.debug(f"Added Fourier terms '{sin_term}' and '{cos_term}'")
        
        df.index = original_index
        logger.info(f"Fourier terms added. Columns now include: {list(df.columns)}")
        
        return df
    
    except Exception as e:
        logger.error(f"An error occurred while adding Fourier terms: {e}")
        raise e

def add_time_based_features(df):
    try:
        if isinstance(df, pd.Series):
            df = df.to_frame()

        if 'Date' not in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("The DataFrame must have a DateTime index or a 'Date' column to convert.")

        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df.set_index('Date', inplace=True)
            logger.info("Converted 'Date' column to DateTime index.")
        
        time_based_features = ['day_of_week', 'day_of_month', 'month', 'quarter', 'year']
        if not all(feature in df.columns for feature in time_based_features):
            df['day_of_week'] = df.index.dayofweek
            df['day_of_month'] = df.index.day
            df['month'] = df.index.month
            df['quarter'] = df.index.quarter
            df['year'] = df.index.year
            logger.info(f"Time-based features added. Columns now include: {df.columns}")
        else:
            logger.info("Time-based features already present in DataFrame. Skipping addition.")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise e

    return df

def train_liquid_nn(model, optimizer, criterion, X_train, y_train, epochs=1000, patience=10):
    logger.debug(f"Training Liquid Neural Network for up to {epochs} epochs with early stopping.")
    model.train()
    
    best_loss = float('inf')
    patience_counter = 0

    for epoch in tqdm(range(int(epochs)), desc="Training Progress", leave=True, dynamic_ncols=True):
        optimizer.zero_grad()

        predictions = model(X_train).squeeze(-1)
        
        if y_train.shape != predictions.shape:
            logger.warning(f"Reshaping y_train from {y_train.shape} to {predictions.shape}")
            y_train = y_train.view_as(predictions)

        loss = criterion(predictions, y_train)

        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs. Best loss: {best_loss}")
            break
        
        if epoch % 10 == 0:
            tqdm.write(f'Epoch {epoch+1}/{epochs}: Loss = {loss.item()}')

def k_fold_cross_validation(model_class, df, selected_models, learning_rate, epochs, k=5, weight_decay=1e-4, num_layers=1, 
                            dropout_rate=0.0, bidirectional=False):
    logger.debug("Starting k-fold cross-validation")
    
    features = selected_models + ['day_of_week', 'month', 'day_of_month', 'quarter', 'year'] + \
               [f'fourier_sin_{i}' for i in range(1, 3)] + \
               [f'fourier_cos_{i}' for i in range(1, 3)]

    check_required_features(df, features)

    X = df[features].values
    y = df['TrueValues'].values.reshape(-1, 1)

    input_dim = len(features)
    X = pd.DataFrame(X).ffill().bfill().fillna(0).values
    y = pd.DataFrame(y).ffill().bfill().fillna(0).values

    total_loss = 0
    kf = KFold(n_splits=int(k), shuffle=True, random_state=42)
    
    # Initialize learnable weights as torch tensors
    weights = {
        'SARIMA': torch.tensor(1.0, requires_grad=True),
        'RandomWalk_L2': torch.tensor(1.0, requires_grad=True),
        'CIR': torch.tensor(1.0, requires_grad=True),
        'GBM': torch.tensor(1.0, requires_grad=True),
        'JumpDiffusion': torch.tensor(1.0, requires_grad=True)
    }

    weight_optimizer = torch.optim.Adam([weights[key] for key in weights], lr=0.01)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        logger.debug(f"Fold {fold + 1} of k-fold cross-validation")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        X_train = torch.tensor(X_train.reshape(X_train.shape[0], 1, input_dim), dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).squeeze(-1)

        model = model_class(input_dim, hidden_dim=32, output_dim=1, num_layers=num_layers, 
                            dropout_rate=dropout_rate, bidirectional=bidirectional)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        model.train()
        for epoch in range(int(epochs)):
            optimizer.zero_grad()
            weight_optimizer.zero_grad()

            predictions = model(X_train).squeeze(-1)

            if y_train.shape != predictions.shape:
                logger.warning(f"Reshaping y_train from {y_train.shape} to {predictions.shape}")
                y_train = y_train.view_as(predictions)

            # Calculate the weighted prediction
            weighted_prediction = (
                weights['SARIMA'] * predictions[:, 0] +
                weights['RandomWalk_L2'] * predictions[:, 1] +
                weights['CIR'] * predictions[:, 2] +
                weights['GBM'] * predictions[:, 3] +
                weights['JumpDiffusion'] * predictions[:, 4]
            )

            # Normalize the weights so they sum to 1
            weight_sum = sum(weights.values())
            weighted_prediction = weighted_prediction / weight_sum

            loss = criterion(weighted_prediction, y_train)
            loss.backward()

            optimizer.step()
            weight_optimizer.step()

        model.eval()
        with torch.no_grad():
            X_val = torch.tensor(X_val.reshape(X_val.shape[0], 1, input_dim), dtype=torch.float32)
            val_predictions = model(X_val).squeeze(-1)
            val_predictions = (
                weights['SARIMA'] * val_predictions[:, 0] +
                weights['RandomWalk_L2'] * val_predictions[:, 1] +
                weights['CIR'] * val_predictions[:, 2] +
                weights['GBM'] * val_predictions[:, 3] +
                weights['JumpDiffusion'] * val_predictions[:, 4]
            )
            val_predictions /= weight_sum

            y_val = torch.tensor(y_val, dtype=torch.float32).squeeze(-1)
            val_predictions = val_predictions.view(-1)
            y_val = y_val.view(-1)

            loss = criterion(val_predictions, y_val)
            total_loss += loss.item()

    return total_loss / k

@app.command()
def main(
    input_csv: Path = typer.Option(..., help="Path to the CSV file containing the data."),
    sarima_predictions_csv: Path = typer.Option(..., help="Path to the CSV file containing SARIMA predictions."),
    output_results: Path = typer.Option(..., help="Path to save the optimization results."),
    k_folds: int = typer.Option(5, help="Number of k-folds for cross-validation."),
    processed_dir: Path = typer.Option(..., help="Directory to save processed files."),
    n_calls: int = typer.Option(10, help="Number of optimization calls for skopt.")
):
    try:
        df = pd.read_csv(input_csv, parse_dates=True, index_col=0)
        logger.info(f"Data loaded successfully from {input_csv}. Shape: {df.shape}")

        sarima_predictions = pd.read_csv(sarima_predictions_csv, parse_dates=True, index_col=0)
        logger.info(f"SARIMA predictions loaded successfully from {sarima_predictions_csv}. Shape: {sarima_predictions.shape}")

        check_datetime_index(df, "Initial CSV DataFrame")

        df = preprocess_data(df)
        check_datetime_index(df, "After Preprocessing")

        required_features = [
            'fourier_sin_1', 'fourier_cos_1', 'day_of_week', 'month', 'quarter', 'year',
            'RandomWalk_L2', 'CIR', 'GBM', 'JumpDiffusion'
        ]

        check_required_features(df, required_features)

        df_final = df
        logger.debug(f"Columns in df_final after preprocessing: {df_final.columns}")

        logger.info("Starting hyperparameter optimization...")
        best_params = hyperparameter_optimization(df_final, sarima_predictions, k_folds=k_folds, n_calls=n_calls)

        if best_params is not None:
            try:
                np.save(output_results, best_params)
                logger.info(f"Best parameters saved to {output_results}")
            except Exception as e:
                logger.error(f"An error occurred while saving the results: {e}")

        if not processed_dir.exists():
            logger.info(f"Processed directory does not exist. Creating: {processed_dir}")
            processed_dir.mkdir(parents=True, exist_ok=True)

        learning_rate, epochs, hidden_dim, weight_decay, num_layers, dropout_rate, bidirectional = best_params.values()
        
        model = LiquidNeuralNetwork(
            input_dim=len(required_features) + 1,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            bidirectional=bidirectional
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        X_train = torch.tensor(df_final[required_features].values.reshape(df_final.shape[0], 1, len(required_features)), dtype=torch.float32)
        y_train = torch.tensor(df_final['TrueValues'].values, dtype=torch.float32).squeeze(-1)
        train_liquid_nn(model, optimizer, criterion, X_train, y_train, epochs=epochs)

    except FileNotFoundError:
        logger.error(f"File not found: {input_csv}. Please ensure the file exists.")
        return
    except Exception as e:
        logger.error(f"An error occurred while loading data: {e}")
        traceback.print_exc()
        return

if __name__ == "__main__":
    app()

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from arch import arch_model

class LiquidNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LiquidNeuralNetwork, self).__init__()
        self.recurrent_layer = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc_layer = nn.Sequential(
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x, _ = self.recurrent_layer(x)
        return self.fc_layer(x[:, -1, :])

def calculate_garch_volatility(df):
    returns = df['TrueValues'].pct_change().fillna(method='ffill').fillna(method='bfill')
    returns = returns.replace([np.inf, -np.inf], 0).fillna(0)

    model = arch_model(returns, vol='Garch', p=1, q=1)
    model_fit = model.fit(disp='off')
    volatility_filled = model_fit.conditional_volatility.reindex(df.index).fillna(method='ffill').fillna(0)
    
    return volatility_filled

def create_differenced_feature(df, column):
    return df[column].diff().fillna(0)

def add_fourier_terms(df, period, n_terms=2):
    time = np.arange(len(df))
    for i in range(1, n_terms + 1):
        df[f'fourier_sin_{i}'] = np.sin(2 * np.pi * i * time / period)
        df[f'fourier_cos_{i}'] = np.cos(2 * np.pi * i * time / period)
    return df

def add_time_based_features(df):
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['day_of_month'] = df.index.day
    df['week_of_year'] = df.index.isocalendar().week
    return df

def train_liquid_nn(model, optimizer, criterion, X_train, y_train, epochs=100):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(X_train)
        loss = criterion(predictions, y_train)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{epochs}: Loss = {loss.item()}')

def ensemble_predictions_liquid_nn(df):
    available_models = ['RandomWalk_L2', 'CIR', 'GBM', 'JumpDiffusion']
    selected_models = [model for model in available_models if input(f"Include {model}? (y/n): ").strip().lower() == 'y']

    if not selected_models:
        raise ValueError("No models selected for the ensemble.")

    learning_rate_input = input(f"Enter learning rate (default 0.001): ").strip()
    learning_rate = float(learning_rate_input) if learning_rate_input else 0.001

    # Add GARCH volatility as a feature
    garch_volatility = calculate_garch_volatility(df).values

    # Add differenced and Fourier features
    for model in selected_models:
        df[f'{model}_diff'] = create_differenced_feature(df, model)

    # Add time-based features
    df = add_time_based_features(df)

    # Add Fourier terms (example: yearly seasonality for daily data)
    df = add_fourier_terms(df, period=365, n_terms=2)

    # Prepare input data
    features = selected_models + [f'{model}_diff' for model in selected_models] + \
               ['day_of_week', 'month', 'day_of_month', 'week_of_year'] + \
               [f'fourier_sin_{i}' for i in range(1, 3)] + \
               [f'fourier_cos_{i}' for i in range(1, 3)] + ['GARCH_Volatility']

    df['GARCH_Volatility'] = garch_volatility
    X = df[features].values
    y = df['TrueValues'].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train = torch.tensor(X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1]), dtype=torch.float32)
    y_train = torch.tensor(y, dtype=torch.float32)

    input_dim = X_train.shape[2]
    hidden_dim = 64
    output_dim = 1
    model = LiquidNeuralNetwork(input_dim, hidden_dim, output_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    train_liquid_nn(model, optimizer, criterion, X_train, y_train, epochs=100)

    model.eval()
    with torch.no_grad():
        ensemble_predictions = model(X_train).numpy().flatten()

    return ensemble_predictions

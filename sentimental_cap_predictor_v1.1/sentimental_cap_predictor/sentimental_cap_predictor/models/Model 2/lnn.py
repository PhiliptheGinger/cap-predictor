import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from arch import arch_model

class LiquidNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LiquidNeuralNetwork, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layer(x)

def calculate_garch_volatility(df):
    """Calculate GARCH(1,1) volatility."""
    # Calculate returns as percentage change of TrueValues
    returns = df['TrueValues'].pct_change()

    # Fill any NaNs or infinite values that result from pct_change()
    returns = returns.fillna(method='ffill').fillna(method='bfill')

    # If returns still contain NaNs or infinite values, replace them with zeros
    if not np.all(np.isfinite(returns)):
        returns = returns.replace([np.inf, -np.inf], 0).fillna(0)

    # Ensure returns have valid finite values before passing to GARCH
    if not np.all(np.isfinite(returns)):
        raise ValueError("Returns still contain NaN or infinite values after processing.")

    # Initialize and fit the GARCH model
    model = arch_model(returns, vol='Garch', p=1, q=1)
    model_fit = model.fit(disp='off')

    # Get conditional volatility
    volatility = model_fit.conditional_volatility

    # Reindex to match the original DataFrame and fill missing values
    volatility_filled = volatility.reindex(df.index).fillna(method='ffill').fillna(0)
    
    return volatility_filled

def train_liquid_nn(model, optimizer, criterion, X_train, y_train, epochs=100):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(X_train)
        loss = criterion(predictions, y_train)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Loss = {loss.item()}')

def ensemble_predictions_liquid_nn(df):
    # Define available models and prompt the user for inclusion
    available_models = ['RandomWalk_L2', 'CIR', 'GBM', 'JumpDiffusion']
    selected_models = []

    for model in available_models:
        user_input = input(f"Do you want to include the {model} model in the ensemble? (y/n): ").strip().lower()
        if user_input == 'y':
            selected_models.append(model)

    # Prepare the inputs based on selected models
    if selected_models:
        X = torch.tensor(df[selected_models].values, dtype=torch.float32)
    else:
        raise ValueError("No models selected for the ensemble.")

    # Calculate and add GARCH volatility as an additional feature
    garch_volatility = calculate_garch_volatility(df).values
    garch_volatility_feature = torch.tensor(garch_volatility, dtype=torch.float32).view(-1, 1)
    X = torch.cat([X, garch_volatility_feature], dim=1)

    y = torch.tensor(df['TrueValues'].values, dtype=torch.float32).view(-1, 1)

    # Define model
    input_dim = X.shape[1]  # Updated input dimension to account for selected models and GARCH volatility
    hidden_dim = 64  # You can adjust this
    output_dim = 1
    model = LiquidNeuralNetwork(input_dim, hidden_dim, output_dim)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Train model
    train_liquid_nn(model, optimizer, criterion, X, y)

    # Generate ensemble predictions
    model.eval()
    with torch.no_grad():
        ensemble_predictions = model(X).numpy().flatten()

    return ensemble_predictions  # Ensure this is a NumPy array or a pandas Series

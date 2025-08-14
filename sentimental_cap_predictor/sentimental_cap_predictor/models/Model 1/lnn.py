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
    returns = df['TrueValues'].pct_change().dropna()
    model = arch_model(returns, vol='Garch', p=1, q=1)
    model_fit = model.fit(disp='off')
    volatility = model_fit.conditional_volatility
    volatility_filled = volatility.reindex(df.index).fillna(method='ffill').fillna(0)  # Forward fill and backfill missing values
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
    # Prepare the inputs (stack the different model predictions as features)
    X = torch.tensor(df[['SARIMA', 'RandomWalk_L2', 'MarkovChain', 'CIR', 'GBM', 'JumpDiffusion']].values, dtype=torch.float32)

    # Calculate and add GARCH volatility as an additional feature
    garch_volatility = calculate_garch_volatility(df).values
    garch_volatility_feature = torch.tensor(garch_volatility, dtype=torch.float32).view(-1, 1)
    X = torch.cat([X, garch_volatility_feature], dim=1)

    y = torch.tensor(df['TrueValues'].values, dtype=torch.float32).view(-1, 1)

    # Define model
    input_dim = X.shape[1]  # Updated input dimension to account for GARCH volatility
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

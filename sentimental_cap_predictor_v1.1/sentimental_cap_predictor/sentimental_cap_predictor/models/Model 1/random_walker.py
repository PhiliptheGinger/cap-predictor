import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import Ridge
import numpy as np
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fit_sarima(data, order, seasonal_order, trend, enforce_stationarity, enforce_invertibility):
    """Fits a SARIMA model to the data."""
    model = SARIMAX(
        data,
        order=order,
        seasonal_order=seasonal_order,
        trend=trend,
        enforce_stationarity=enforce_stationarity,
        enforce_invertibility=enforce_invertibility
    )
    fitted_model = model.fit(disp=False, maxiter=1000, method='powell', xtol=1e-5, ftol=1e-5)
    return fitted_model

def generate_random_walks(model, steps, num_simulations=100):
    """Generates random walks (simulated future paths) using the SARIMA model."""
    simulations = []
    
    # Use tqdm to track the progress of simulations
    for _ in tqdm(range(num_simulations), desc="Generating random walks"):
        sim = model.simulate(steps=steps, nsimulations=steps)  # Add nsimulations argument as steps
        simulations.append(sim)
    
    return np.array(simulations)

def apply_l2_regularization(simulations, true_values=None):
    """
    Applies L2 regularization (Ridge regression) to find the optimal prediction based on the simulations.

    Args:
        simulations (np.ndarray): The simulated future paths.
        true_values (pd.Series): Optional, true future values for fitting (used for test data if available).
    
    Returns:
        np.ndarray: The optimal prediction.
    """
    # Reshape simulations for regression
    X = simulations.T  # Shape: (num_simulations, steps)
    
    if true_values is not None:
        # Use true values to fit the model (when available)
        y = true_values.values
    else:
        # If true values aren't available, we can use the average simulation as a proxy
        y = np.mean(X, axis=0)
    
    # Adjust X and y to have the same number of samples if they differ
    if X.shape[0] > y.shape[0]:
        X = X[:y.shape[0], :]
    elif y.shape[0] > X.shape[0]:
        y = y[:X.shape[0]]
    
    # Apply Ridge regression (L2 regularization)
    ridge_model = Ridge(alpha=1.0)  # Alpha controls the strength of regularization
    
    # Track progress of fitting the ridge model
    with tqdm(total=1, desc="Fitting Ridge regression") as pbar:
        ridge_model.fit(X, y)
        pbar.update(1)
    
    # Predict the optimal path
    optimal_prediction = ridge_model.predict(X)
    
    return optimal_prediction

def main():
    # Example of how to use the functions with your data
    data = pd.Series(...)  # Replace with your actual time series data
    
    # Split the data into train and test sets (80% train, 20% test)
    split_index = int(len(data) * 0.8)
    train_data = data[:split_index]
    test_data = data[split_index:]
    
    # Assuming best_params is the result of your hyperparameter optimization
    best_params = {
        'p': 1, 'd': 1, 'q': 1,
        'P': 1, 'D': 1, 'Q': 1, 'S': 12,
        'trend': 'c',
        'enforce_stationarity': True,
        'enforce_invertibility': True
    }

    # Fit the model with the best parameters
    model = fit_sarima(train_data, 
                       (best_params['p'], best_params['d'], best_params['q']),
                       (best_params['P'], best_params['D'], best_params['Q'], best_params['S']),
                       best_params['trend'],
                       best_params['enforce_stationarity'],
                       best_params['enforce_invertibility'])
    
    # Generate SARIMA predictions for the test data
    predicted_seasons = model.forecast(steps=len(test_data))
    
    # Generate random walks (simulations) based on the fitted model
    steps = len(test_data)  # Set the number of steps to match the length of the test data
    num_simulations = 100  # Number of simulations
    simulations = generate_random_walks(model, steps, num_simulations)
    
    # Apply L2 regularization to find the optimal prediction
    optimal_prediction = apply_l2_regularization(simulations, true_values=test_data)
    
    # Save Predictions and Results
    df_test = pd.DataFrame({
        'TrueValues': test_data,
        'SARIMA': predicted_seasons,
        'OptimalPrediction': optimal_prediction,
        'BrownianMotion': test_data,
    })
    
    # Return the generated simulations and the optimal prediction
    return {
        'simulations': simulations,
        'optimal_prediction': optimal_prediction,
        'train_data': train_data,
        'test_data': test_data,
        'best_params': best_params,
        'df_test': df_test
    }

if __name__ == "__main__":
    result = main()
    # The result dictionary now contains the simulations, optimal prediction, train_data, test_data, df_test, and best_params

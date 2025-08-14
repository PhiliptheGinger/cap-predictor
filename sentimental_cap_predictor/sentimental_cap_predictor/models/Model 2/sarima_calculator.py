import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import optuna
from tqdm import tqdm
import itertools
import warnings
import joblib
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fit_sarima(data, order, seasonal_order, trend, enforce_stationarity, enforce_invertibility):
    """Fits a SARIMA model to the data."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model = SARIMAX(
            data,
            order=order,
            seasonal_order=seasonal_order,
            trend=trend,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility
        )
        try:
            fitted_model = model.fit(disp=False, maxiter=1000, method='powell', xtol=1e-5, ftol=1e-5)
        except Exception as e:
            logging.error(f"Error fitting SARIMA model with parameters: {order}, {seasonal_order}, {trend}. Error: {e}")
            raise e
        
        # Handle frequency warnings and value warnings
        for warning in w:
            if issubclass(warning.category, UserWarning):
                if "inferred frequency" in str(warning.message):
                    logging.warning(f"Warning: {warning.message}")
            if issubclass(warning.category, ValueWarning):
                logging.warning(f"ValueWarning: {warning.message}")

    return fitted_model

def predict_seasons_with_intervals(model, steps):
    """Predicts future values with confidence intervals using the SARIMA model."""
    forecast = model.get_forecast(steps=steps)
    mean_forecast = forecast.predicted_mean
    conf_int = forecast.conf_int()
    return mean_forecast, conf_int

def evaluate_sarima(params, train_data, test_data):
    """
    Evaluates a SARIMA model with the given parameters.

    Args:
        params (dict): Dictionary containing SARIMA parameters including 'p', 'd', 'q', 'P', 'D', 'Q', 'S', 'trend',
                       'enforce_stationarity', 'enforce_invertibility'.
        train_data (pd.Series): The training data.
        test_data (pd.Series): The testing data.
    
    Returns:
        float: The mean squared error of the SARIMA model's predictions.
    """
    # Runtime check for params dictionary
    logging.info(f"Evaluating SARIMA with parameters: {params}")
    assert isinstance(params, dict), f"Expected a dictionary for parameters, got {type(params)}"
    
    order = (params['p'], params['d'], params['q'])
    seasonal_order = (params['P'], params['D'], params['Q'], params['S'])
    
    model = fit_sarima(train_data, order, seasonal_order, params['trend'], 
                       params['enforce_stationarity'], params['enforce_invertibility'])
    
    predictions, _ = predict_seasons_with_intervals(model, len(test_data))
    mse = mean_squared_error(test_data, predictions)
    
    # Validate that the MSE is within reasonable bounds
    assert not pd.isnull(mse), "MSE is NaN"
    assert mse < 1e6, f"Unusually high Mean Squared Error: {mse}"
    
    return mse

def dynamic_param_selection(train_data):
    """Dynamically selects the SARIMA parameters based on the data size and characteristics."""
    n = len(train_data)
    
    # Set p, d, q based on data size and autocorrelation
    p = min(3, n // 10)  # Restrict p to a reasonable size
    d = 1  # Differencing typically needed for non-stationary data
    q = min(3, n // 10)  # Restrict q similarly
    
    # Set P, D, Q, S for seasonal components only if data length supports it
    S = 12 if n > 365 else 0  # Set seasonal period based on data length; set to 0 if not enough data
    P = 1 if S > 1 else 0
    D = 1 if S > 1 else 0
    Q = 1 if S > 1 else 0
    
    # Adjust S to avoid invalid configurations
    if S <= 1:
        P = D = Q = S = 0
    
    return {
        'p': range(p + 1),
        'd': [d],
        'q': range(q + 1),
        'P': [P],
        'D': [D],
        'Q': [Q],
        'S': [S],
        'trend': ['c', 't', None],
        'enforce_stationarity': [True, False],
        'enforce_invertibility': [True, False]
    }

def moderate_expand_search_space(best_params, factor=0.7):
    """Moderately expands the search space around the best parameters found during the grid search."""
    def expand_list(param, range_factor=0.7):
        return [
            max(0, int(param - range_factor * param)),
            param,
            min(param + int(range_factor * param), param + 2)  # Expands the range more generously
        ]
    
    return {
        'p': expand_list(best_params['p']),
        'd': expand_list(best_params['d']),
        'q': expand_list(best_params['q']),
        'P': expand_list(best_params['P']),
        'D': expand_list(best_params['D']),
        'Q': expand_list(best_params['Q']),
        'S': expand_list(best_params['S'], range_factor=factor),
        'trend': [best_params['trend']],  # May consider allowing variability
        'enforce_stationarity': [best_params['enforce_stationarity']],  # Also consider some variability here
        'enforce_invertibility': [best_params['enforce_invertibility']]  # Same as above
    }

def perform_hyperparameter_optimization(train_data, test_data, timeout_minutes, cache_dir=None, early_stopping_rounds=20):
    """Performs hyperparameter optimization for SARIMA model using Optuna with expanded search space and early stopping."""
    
    sarima_params = dynamic_param_selection(train_data)
    best_params = None
    best_grid_params = None
    best_grid_score = float('inf')

    # Coarse Grid Search with tqdm progress bar
    logging.info("Starting coarse grid search...")
    param_combinations = list(itertools.product(
        sarima_params['p'], sarima_params['d'], sarima_params['q'],
        sarima_params['P'], sarima_params['D'], sarima_params['Q'], sarima_params['S'],
        sarima_params['trend'], sarima_params['enforce_stationarity'], sarima_params['enforce_invertibility']
    ))
    
    valid_parameters = []

    for p, d, q, P, D, Q, S, trend, enforce_stationarity, enforce_invertibility in tqdm(param_combinations, total=len(param_combinations)):
        order = (p, d, q)
        seasonal_order = (P, D, Q, S)
        try:
            model = fit_sarima(train_data, order, seasonal_order, trend, enforce_stationarity, enforce_invertibility)
            predictions, _ = predict_seasons_with_intervals(model, len(test_data))
            mse = mean_squared_error(test_data, predictions)
            if mse < best_grid_score:
                best_grid_score = mse
                best_grid_params = {
                    'p': p, 'd': d, 'q': q,
                    'P': P, 'D': D, 'Q': Q, 'S': S,
                    'trend': trend,
                    'enforce_stationarity': enforce_stationarity,
                    'enforce_invertibility': enforce_invertibility
                }
            valid_parameters.append({
                'p': p, 'd': d, 'q': q,
                'P': P, 'D': D, 'Q': Q, 'S': S,
                'trend': trend,
                'enforce_stationarity': enforce_stationarity,
                'enforce_invertibility': enforce_invertibility
            })
        except Exception as e:
            logging.error(f"Skipping parameter set due to error: {e}")
            continue

    if not valid_parameters:
        logging.error("No valid parameters found during the grid search.")
        return []  # Return an empty list instead of None

    return valid_parameters


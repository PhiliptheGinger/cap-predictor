import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
import optuna
from tqdm import tqdm
import itertools
import warnings
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fit_sarima(data, order, seasonal_order, trend, enforce_stationarity, enforce_invertibility):
    """Fits a SARIMA model to the data."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")  # Catch all warnings
        model = SARIMAX(
            data,
            order=order,
            seasonal_order=seasonal_order,
            trend=trend,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility
        )
        try:
            fitted_model = model.fit(disp=False, maxiter=5000, method='lbfgs', xtol=1e-5, ftol=1e-5)
        except Exception as e:
            logging.error(f"Error fitting SARIMA model with parameters: {order}, {seasonal_order}, {trend}. Error: {e}")
            raise e

        # Handle any warnings caught during fitting
        for warning in w:
            if issubclass(warning.category, Warning):
                logging.warning(f"Warning captured: {warning.message}")

    return fitted_model

def predict_seasons_with_intervals(model, steps):
    """Predicts future values with confidence intervals using the SARIMA model."""
    forecast = model.get_forecast(steps=steps)
    mean_forecast = forecast.predicted_mean
    conf_int = forecast.conf_int()
    return mean_forecast, conf_int

def evaluate_sarima(params, train_data, test_data, dominant_frequency=None):
    """
    Evaluates a SARIMA model with the given parameters using direct forecasting.

    Args:
        params (dict): Dictionary containing SARIMA parameters including 'p', 'd', 'q', 'P', 'D', 'Q', 'S', 'trend',
                       'enforce_stationarity', 'enforce_invertibility'.
        train_data (pd.Series): The training data.
        test_data (pd.Series): The testing data.
        dominant_frequency (int): The dominant frequency extracted from FFT/ACF analysis (used as seasonal period S).
    
    Returns:
        float: The mean squared error of the SARIMA model's predictions.
    """
    logging.info(f"Evaluating SARIMA with parameters: {params}")
    assert isinstance(params, dict), f"Expected a dictionary for parameters, got {type(params)}"
    
    order = (params['p'], params['d'], params['q'])
    
    # Use the dominant frequency as the seasonal period if provided
    if dominant_frequency:
        seasonal_order = (params['P'], params['D'], params['Q'], dominant_frequency)
    else:
        seasonal_order = (params['P'], params['D'], params['Q'], params['S'])
    
    # Fit SARIMA model on the entire training data
    model = fit_sarima(train_data, order, seasonal_order, params['trend'], params['enforce_stationarity'], params['enforce_invertibility'])
    
    # Forecast on the test data
    predictions, _ = predict_seasons_with_intervals(model, len(test_data))
    
    # Calculate the mean squared error
    mse = mean_squared_error(test_data, predictions)
    
    assert not pd.isnull(mse), "MSE is NaN"
    assert mse < 1e6, f"Unusually high Mean Squared Error: {mse}"
    
    return mse

def check_stationarity(data, significance=0.05):
    """Performs the Augmented Dickey-Fuller test to check for stationarity."""
    result = adfuller(data)
    p_value = result[1]
    return p_value < significance  # Returns True if the series is stationary

def dynamic_param_selection(train_data):
    """Dynamically selects the SARIMA parameters based on the data size and characteristics."""
    n = len(train_data)
    
    # Perform stationarity check for d
    is_stationary = check_stationarity(train_data)
    d = 0 if is_stationary else 1  # Set d=0 if stationary, otherwise 1
    
    # Set p, q based on data size and autocorrelation
    p = min(5, n // 10)
    q = min(5, n // 10)
    
    # Set P, D, Q, S for seasonal components only if data length supports it
    S = 12 if n > 365 else 0  # Set seasonal period based on data length
    if S > 1:
        seasonal_diff_series = train_data.diff(S).dropna()
        is_seasonal_stationary = check_stationarity(seasonal_diff_series)
        D = 0 if is_seasonal_stationary else 1  # Set D=0 if seasonal part is stationary, otherwise 1
        P = Q = 2
    else:
        P = D = Q = S = 0  # No seasonal component if data length is insufficient
    
    return {
        'p': range(p + 1),
        'd': [d],
        'q': range(q + 1),
        'P': range(P + 1),
        'D': [D],
        'Q': range(Q + 1),
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
        'trend': [best_params['trend']],
        'enforce_stationarity': [best_params['enforce_stationarity']],
        'enforce_invertibility': [best_params['enforce_invertibility']]
    }

def objective(trial, train_data, test_data, dominant_frequency=None):
    """Defines the objective function for Optuna optimization."""
    p = trial.suggest_int('p', 0, 5)
    d = trial.suggest_int('d', 0, 1)
    q = trial.suggest_int('q', 0, 5)
    P = trial.suggest_int('P', 0, 2)
    D = trial.suggest_int('D', 0, 1)
    Q = trial.suggest_int('Q', 0, 2)
    
    # If we have a dominant frequency, suggest seasonal order using that frequency
    if dominant_frequency:
        S = trial.suggest_int('S', dominant_frequency, dominant_frequency)
    else:
        S = trial.suggest_int('S', 0, 12)
    
    trend = trial.suggest_categorical('trend', ['c', 't', None])
    enforce_stationarity = trial.suggest_categorical('enforce_stationarity', [True, False])
    enforce_invertibility = trial.suggest_categorical('enforce_invertibility', [True, False])
    
    params = {
        'p': p, 'd': d, 'q': q,
        'P': P, 'D': D, 'Q': Q, 'S': S,
        'trend': trend,
        'enforce_stationarity': enforce_stationarity,
        'enforce_invertibility': enforce_invertibility
    }
    
    mse = evaluate_sarima(params, train_data, test_data, dominant_frequency=dominant_frequency)
    return mse

def perform_hyperparameter_optimization(train_data, test_data, timeout_minutes, dominant_frequency=None, cache_dir=None, early_stopping_rounds=20):
    """Performs hyperparameter optimization for SARIMA model using Optuna with expanded search space and early stopping."""
    
    sarima_params = dynamic_param_selection(train_data)
    best_params = None
    best_grid_params = None
    best_grid_score = float('inf')

    logging.info("Starting coarse grid search...")
    param_combinations = list(itertools.product(
        sarima_params['p'], sarima_params['d'], sarima_params['q'],
        sarima_params['P'], sarima_params['D'], sarima_params['Q'], sarima_params['S'],
        sarima_params['trend'], sarima_params['enforce_stationarity'], sarima_params['enforce_invertibility']
    ))
    
    valid_parameters = []

    # Add progress bar here to track the coarse grid search
    with tqdm(total=len(param_combinations), desc="SARIMA Grid Search Progress", ncols=100) as pbar:
        for p, d, q, P, D, Q, S, trend, enforce_stationarity, enforce_invertibility in param_combinations:
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
            
            pbar.update(1)  # Update the progress bar after each iteration

    if not valid_parameters:
        logging.error("No valid parameters found during the grid search.")
        return []  # Return an empty list instead of None

    logging.info("Starting Optuna optimization...")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, train_data, test_data, dominant_frequency=dominant_frequency), timeout=timeout_minutes * 60)
    
    best_params = study.best_params
    logging.info(f"Best parameters found by Optuna: {best_params}")

    return best_params

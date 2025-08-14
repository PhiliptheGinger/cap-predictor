import numpy as np
import logging
import traceback
from itertools import product
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# CIR model simulation with truncation and length check
def simulate_CIR(kappa, theta, sigma, X0, T=1, N=100):
    """Simulates a CIR process."""
    if N <= 0:
        raise ValueError("N must be a positive integer.")
        
    dt = T / N
    X = np.zeros(N)
    X[0] = X0
    
    for i in range(1, N):
        try:
            dX = kappa * (theta - X[i-1]) * dt + sigma * np.sqrt(max(X[i-1], 0)) * np.random.normal()
            X[i] = X[i-1] + dX
            X[i] = max(X[i], 0)  # Ensure non-negative values
            logging.debug(f"CIR: Index {i}, Value {X[i]}")  # Track the index and value
        except Exception as e:
            logging.error(f"Error in CIR simulation at index {i}: {e}")
            return None
    
    return X

# GBM model simulation with truncation and length check
def simulate_GBM(mu, sigma, S0, T=1, N=100):
    """Simulates a Geometric Brownian Motion process."""
    if N <= 0:
        raise ValueError("N must be a positive integer.")
        
    dt = T / N
    S = np.zeros(N)
    S[0] = S0
    
    for i in range(1, N):
        try:
            dS = mu * S[i-1] * dt + sigma * S[i-1] * np.random.normal()
            S[i] = S[i-1] + dS
            logging.debug(f"GBM: Index {i}, Value {S[i]}")  # Track the index and value
        except Exception as e:
            logging.error(f"Error in GBM simulation at index {i}: {e}")
            return None
    
    return S

# Jump Diffusion model simulation with truncation and length check
def simulate_JumpDiffusion(lambda_, muJ, sigmaJ, S0, T=1, N=100):
    """Simulates a Jump Diffusion process."""
    if N <= 0:
        raise ValueError("N must be a positive integer.")
        
    dt = T / N
    S = np.zeros(N)
    S[0] = S0
    
    for i in range(1, N):
        try:
            jump = np.random.poisson(lambda_) * (muJ + sigmaJ * np.random.normal())
            dS = S[i-1] * dt + jump
            S[i] = S[i-1] + dS
            logging.debug(f"JumpDiffusion: Index {i}, Value {S[i]}")  # Track the index and value
        except Exception as e:
            logging.error(f"Error in Jump Diffusion simulation at index {i}: {e}")
            return None
    
    return S

# Function to calculate differencing
def calculate_diff(series):
    """Calculates the differencing of a time series."""
    if series is None or len(series) == 0:
        logging.error("Cannot calculate differencing on an empty series.")
        return None
    diff = np.diff(series)
    logging.debug(f"Differences: {diff}")  # Track the differencing
    return diff

# Rough grid search parameters
cir_params = list(product([0.1, 0.5, 1.0], [0.01, 0.05, 0.1], [0.01, 0.05, 0.1], [0.01, 0.1, 0.5]))
gbm_params = list(product([0.01, 0.05, 0.1], [0.01, 0.05, 0.1], [1, 10, 50]))
jump_params = list(product([0.1, 0.5, 1.0], [0.01, 0.05, 0.1], [0.01, 0.05, 0.1], [1, 10, 50]))

# Function to perform grid search and return distributions and errors
def perform_grid_search():
    """Performs a grid search for the best parameters for each model and returns the distributions."""
    best_cir_params = None
    best_gbm_params = None
    best_jump_params = None

    min_error_cir = float('inf')
    min_error_gbm = float('inf')
    min_error_jump = float('inf')

    best_cir_dist = None
    best_gbm_dist = None
    best_jump_dist = None

    # Evaluate CIR model
    logging.info("Starting grid search for CIR model.")
    for kappa, theta, sigma, X0 in cir_params:
        try:
            X = simulate_CIR(kappa, theta, sigma, X0)
            if X is None or len(X) == 0:
                logging.error(f"Invalid CIR distribution for params {kappa, theta, sigma, X0}")
                continue
            X_diff = calculate_diff(X)
            if X_diff is None:
                continue
            error = np.mean(np.square(np.insert(X_diff, 0, 0)))  # Padding with 0
            logging.info(f"CIR Params: {kappa, theta, sigma, X0}, Error: {error}")
            if error < min_error_cir:
                min_error_cir = error
                best_cir_params = (kappa, theta, sigma, X0)
                best_cir_dist = X
        except Exception as e:
            logging.error(f"Error simulating CIR model with params {kappa, theta, sigma, X0}: {e}")
            logging.error(traceback.format_exc())

    logging.info(f"Best CIR parameters: {best_cir_params} with error: {min_error_cir}")

    # Evaluate GBM model
    logging.info("Starting grid search for GBM model.")
    for mu, sigma, S0 in gbm_params:
        try:
            S = simulate_GBM(mu, sigma, S0)
            if S is None or len(S) == 0:
                logging.error(f"Invalid GBM distribution for params {mu, sigma, S0}")
                continue
            S_diff = calculate_diff(S)
            if S_diff is None:
                continue
            error = np.mean(np.square(np.insert(S_diff, 0, 0)))  # Padding with 0
            logging.info(f"GBM Params: {mu, sigma, S0}, Error: {error}")
            if error < min_error_gbm:
                min_error_gbm = error
                best_gbm_params = (mu, sigma, S0)
                best_gbm_dist = S
        except Exception as e:
            logging.error(f"Error simulating GBM model with params {mu, sigma, S0}: {e}")
            logging.error(traceback.format_exc())

    logging.info(f"Best GBM parameters: {best_gbm_params} with error: {min_error_gbm}")

    # Evaluate Jump Diffusion model
    logging.info("Starting grid search for Jump Diffusion model.")
    for lambda_, muJ, sigmaJ, S0 in jump_params:
        try:
            S = simulate_JumpDiffusion(lambda_, muJ, sigmaJ, S0)
            if S is None or len(S) == 0:
                logging.error(f"Invalid Jump Diffusion distribution for params {lambda_, muJ, sigmaJ, S0}")
                continue
            S_diff = calculate_diff(S)
            if S_diff is None:
                continue
            error = np.mean(np.square(np.insert(S_diff, 0, 0)))  # Padding with 0
            logging.info(f"Jump Diffusion Params: {lambda_, muJ, sigmaJ, S0}, Error: {error}")
            if error < min_error_jump:
                min_error_jump = error
                best_jump_params = (lambda_, muJ, sigmaJ, S0)
                best_jump_dist = S
        except Exception as e:
            logging.error(f"Error simulating Jump Diffusion model with params {lambda_, muJ, sigmaJ, S0}: {e}")
            logging.error(traceback.format_exc())

    logging.info(f"Best Jump Diffusion parameters: {best_jump_params} with error: {min_error_jump}")

    # Ensure the best distributions are valid before creating the DataFrame
    if best_cir_dist is None or len(best_cir_dist) == 0:
        logging.error("Best CIR distribution is invalid.")
    if best_gbm_dist is None or len(best_gbm_dist) == 0:
        logging.error("Best GBM distribution is invalid.")
    if best_jump_dist is None or len(best_jump_dist) == 0:
        logging.error("Best Jump Diffusion distribution is invalid.")

    # Create DataFrame from best distributions
    df = pd.DataFrame({
        'CIR': best_cir_dist,
        'GBM': best_gbm_dist,
        'JumpDiffusion': best_jump_dist
    })

    return df, best_cir_params, best_gbm_params, best_jump_params

# Define the skopt search spaces
cir_space = [Real(0.1, 1.0, name='kappa'), Real(0.01, 0.1, name='theta'), Real(0.01, 0.1, name='sigma'), Real(0.01, 0.5, name='X0')]
gbm_space = [Real(0.01, 0.1, name='mu'), Real(0.01, 0.1, name='sigma'), Real(1, 50, name='S0')]
jump_space = [Real(0.1, 1.0, name='lambda_'), Real(0.01, 0.1, name='muJ'), Real(0.01, 0.1, name='sigmaJ'), Real(1, 50, name='S0')]

# Objective functions for skopt
@use_named_args(cir_space)
def cir_objective(**params):
    try:
        X = simulate_CIR(**params)
        if X is None or len(X) == 0:
            logging.error("CIR objective returned invalid series.")
            return float('inf')
        X_diff = calculate_diff(X)
        if X_diff is None:
            return float('inf')
        error = np.mean(np.square(np.insert(X_diff, 0, 0)))  # Padding with 0
        return error
    except Exception as e:
        logging.error(f"Error in CIR optimization: {e}")
        return float('inf')

@use_named_args(gbm_space)
def gbm_objective(**params):
    try:
        S = simulate_GBM(**params)
        if S is None or len(S) == 0:
            logging.error("GBM objective returned invalid series.")
            return float('inf')
        S_diff = calculate_diff(S)
        if S_diff is None:
            return float('inf')
        error = np.mean(np.square(np.insert(S_diff, 0, 0)))  # Padding with 0
        return error
    except Exception as e:
        logging.error(f"Error in GBM optimization: {e}")
        return float('inf')

@use_named_args(jump_space)
def jump_objective(**params):
    try:
        S = simulate_JumpDiffusion(**params)
        if S is None or len(S) == 0:
            logging.error("Jump Diffusion objective returned invalid series.")
            return float('inf')
        S_diff = calculate_diff(S)
        if S_diff is None:
            return float('inf')
        error = np.mean(np.square(np.insert(S_diff, 0, 0)))  # Padding with 0
        return error
    except Exception as e:
        logging.error(f"Error in Jump Diffusion optimization: {e}")
        return float('inf')

if __name__ == "__main__":
    # Perform rough grid search
    results_df, best_cir_params, best_gbm_params, best_jump_params = perform_grid_search()
    logging.info("Rough grid search completed.")
    
    # Perform skopt optimization based on rough search results
    logging.info("Starting skopt optimization for CIR.")
    cir_result = gp_minimize(cir_objective, cir_space, n_calls=50, n_random_starts=10)
    
    logging.info("Starting skopt optimization for GBM.")
    gbm_result = gp_minimize(gbm_objective, gbm_space, n_calls=50, n_random_starts=10)
    
    logging.info("Starting skopt optimization for Jump Diffusion.")
    jump_result = gp_minimize(jump_objective, jump_space, n_calls=50, n_random_starts=10)
    
    # Log the results
    logging.info(f"Optimized CIR parameters: {cir_result.x}")
    logging.info(f"Optimized GBM parameters: {gbm_result.x}")
    logging.info(f"Optimized Jump Diffusion parameters: {jump_result.x}")
    
    logging.info("Optimization completed. Final Results DataFrame:")
    logging.info(results_df.head())

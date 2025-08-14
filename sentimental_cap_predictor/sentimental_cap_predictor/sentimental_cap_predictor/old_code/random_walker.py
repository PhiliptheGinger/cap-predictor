import sys
from pathlib import Path
import typer
from loguru import logger
import pandas as pd
import joblib
import numpy as np
from hashlib import md5
from tqdm import tqdm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import Ridge
from sentimental_cap_predictor.sentimental_cap_predictor.sentimental_cap_predictor.config import PROCESSED_DATA_DIR, INTERIM_DATA_DIR, RAW_DATA_DIR, MODELING_DIR

# Add MODELING_DIR to the Python path
sys.path.append(str(MODELING_DIR))

# Import necessary functions
try:
    from stochastic_grid_search import simulate_GBM, simulate_JumpDiffusion, simulate_CIR
except ImportError as e:
    logger.error(f"ImportError: {e}")
    sys.exit(1)

app = typer.Typer()

# Initial Setup: Define train ratio and default frequency
TRAIN_RATIO = 0.8  # Ratio of data to be used for training
DEFAULT_FREQUENCY = 'D'  # Default frequency set to daily

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

def generate_random_walks(model, steps, num_simulations=100, model_type='sarima', **kwargs):
    """Generates random walks (simulated future paths) using the specified model."""
    simulations = []
    
    for _ in tqdm(range(num_simulations), desc=f"Generating random walks for {model_type}"):
        if model_type == 'sarima':
            sim = model.simulate(steps=steps, nsimulations=steps)
        elif model_type == 'cir':
            sim = simulate_CIR(kwargs['kappa'], kwargs['theta'], kwargs['sigma'], kwargs['x0'], T=1, N=steps)
        elif model_type == 'gbm':
            sim = simulate_GBM(kwargs['mu'], kwargs['sigma'], kwargs['S0'], T=1, N=steps)
        elif model_type == 'jump':
            sim = simulate_JumpDiffusion(kwargs['lambda_'], kwargs['muJ'], kwargs['sigmaJ'], kwargs['S0'], T=1, N=steps)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Adjust the length of the simulation if it doesn't match the expected steps
        if len(sim) < steps:
            padding = np.full(steps - len(sim), np.nan)
            sim = np.concatenate([sim, padding])
        elif len(sim) > steps:
            sim = sim[:steps]
        
        simulations.append(sim)
    
    simulations = np.array(simulations)
    
    # Calculate the difference (diff) of the simulations
    simulations_diff = np.diff(simulations, axis=1)
    
    # Handle NaN values in the diff array
    simulations_diff = np.nan_to_num(simulations_diff)
    
    return simulations_diff

def apply_l2_regularization(simulations_diff, true_values=None):
    """Applies L2 regularization (Ridge regression) to find the optimal prediction based on the simulations."""
    X = simulations_diff.T  # Shape: (num_simulations, steps)
    
    if true_values is not None:
        y = true_values.values
    else:
        y = np.mean(X, axis=0)
    
    if X.shape[0] > y.shape[0]:
        X = X[:y.shape[0], :]
    elif y.shape[0] > X.shape[0]:
        y = y[:X.shape[0]]
    
    ridge_model = Ridge(alpha=1.0)
    
    with tqdm(total=1, desc="Fitting Ridge regression") as pbar:
        ridge_model.fit(X, y)
        pbar.update(1)
    
    optimal_prediction = ridge_model.predict(X)
    return optimal_prediction

@app.command()
def main(
    interim_dir: Path = typer.Option(INTERIM_DATA_DIR, help="Directory to save intermediate steps and models."),
    processed_dir: Path = typer.Option(PROCESSED_DATA_DIR, help="Directory to save the final features and results."),
):
    logger.info("Script started. Prompting for ticker symbol.")
    
    ticker = input(f"{typer.style('Please enter the ticker symbol:', fg=typer.colors.YELLOW)} ").strip().upper()

    if not ticker:
        logger.error("No ticker symbol provided. Exiting the script.")
        return

    logger.info(f"Ticker symbol entered: {ticker}")

    input_feather = RAW_DATA_DIR / f"{ticker}.feather"
    logger.info(f"Loading data from Feather file: {input_feather}")
    
    try:
        df = pd.read_feather(input_feather)
        logger.info(f"Data loaded successfully with shape: {df.shape}")
    except FileNotFoundError:
        logger.error(f"File not found: {input_feather}. Please ensure the file exists in the RAW_DATA_DIR.")
        return
    except Exception as e:
        logger.error(f"An error occurred while loading data: {e}")
        return

    if df.empty:
        logger.error("Loaded data is empty. Exiting the script.")
        return

    logger.info("Ensuring DatetimeIndex and setting frequency.")
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    if df.index.freq is None:
        df = df.asfreq(DEFAULT_FREQUENCY)
        logger.info(f"No frequency information was provided. Set default frequency to {DEFAULT_FREQUENCY}.")

    logger.info("Splitting data into train and test sets.")
    train_size = int(len(df) * TRAIN_RATIO)
    train_data = df[:train_size]['Close']
    test_data = df[train_size:]['Close']

    if len(train_data) == 0:
        logger.error("Training data is empty. Unable to proceed with model fitting.")
        return

    # Example parameters for SARIMA
    best_params = {
        'p': 1, 'd': 1, 'q': 1,
        'P': 1, 'D': 1, 'Q': 1, 'S': 12,
        'trend': 'c',
        'enforce_stationarity': True,
        'enforce_invertibility': True
    }

    logger.info("Fitting the SARIMA model and generating random walks.")
    sarima_model = fit_sarima(train_data, 
                              (best_params['p'], best_params['d'], best_params['q']),
                              (best_params['P'], best_params['D'], best_params['Q'], best_params['S']),
                              best_params['trend'],
                              best_params['enforce_stationarity'],
                              best_params['enforce_invertibility'])

    sarima_predictions = sarima_model.predict(start=len(train_data), end=len(df) - 1)

    steps = len(test_data)
    num_simulations = 100
    sarima_simulations_diff = generate_random_walks(sarima_model, steps, num_simulations, model_type='sarima')

    logger.info("Applying L2 regularization to SARIMA simulations.")
    sarima_optimal_prediction = apply_l2_regularization(sarima_simulations_diff, true_values=test_data)

    # Perform simulations for other models
    logger.info("Performing CIR simulations.")
    cir_params = {'kappa': 0.1, 'theta': 0.2, 'sigma': 0.3, 'x0': 0.05}
    cir_simulations_diff = generate_random_walks(None, steps, num_simulations, model_type='cir', **cir_params)

    logger.info("Performing GBM simulations.")
    gbm_params = {'mu': 0.1, 'sigma': 0.2, 'S0': 1.0}
    gbm_simulations_diff = generate_random_walks(None, steps, num_simulations, model_type='gbm', **gbm_params)

    logger.info("Performing Jump Diffusion simulations.")
    jump_params = {'lambda_': 0.1, 'muJ': 0.1, 'sigmaJ': 0.2, 'S0': 1.0}
    jump_simulations_diff = generate_random_walks(None, steps, num_simulations, model_type='jump', **jump_params)

    # Apply L2 regularization to find optimal predictions for CIR, GBM, and Jump Diffusion
    logger.info("Applying L2 regularization to CIR simulations.")
    cir_optimal_prediction = apply_l2_regularization(cir_simulations_diff, true_values=test_data)

    logger.info("Applying L2 regularization to GBM simulations.")
    gbm_optimal_prediction = apply_l2_regularization(gbm_simulations_diff, true_values=test_data)

    logger.info("Applying L2 regularization to Jump Diffusion simulations.")
    jump_optimal_prediction = apply_l2_regularization(jump_simulations_diff, true_values=test_data)

    # Ensure all arrays have the same length
    min_length = min(len(test_data), len(sarima_predictions), len(cir_optimal_prediction),
                     len(gbm_optimal_prediction), len(jump_optimal_prediction))

    df_test = pd.DataFrame({
        'TrueValues': test_data[-min_length:],  # Truncate to min_length
        'SARIMA': sarima_predictions[-min_length:],  # Truncate to min_length
        'Optimal_SARIMA': sarima_optimal_prediction[-min_length:],  # Truncate to min_length
        'Optimal_CIR': cir_optimal_prediction[-min_length:],  # Truncate to min_length
        'Optimal_GBM': gbm_optimal_prediction[-min_length:],  # Truncate to min_length
        'Optimal_Jump': jump_optimal_prediction[-min_length:],  # Truncate to min_length
    })

    logger.info("Saving the final results to a CSV file.")
    output_csv_path = processed_dir / f"{ticker}_final_predictions.csv"
    df_test.to_csv(output_csv_path, index=True)

    logger.info(f"Results saved to {output_csv_path}.")

if __name__ == "__main__":
    app()

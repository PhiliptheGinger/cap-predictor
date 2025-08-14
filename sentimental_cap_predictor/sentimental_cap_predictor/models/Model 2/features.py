import sys
from pathlib import Path
import typer
from loguru import logger
import pandas as pd
import joblib
import numpy as np
from hashlib import md5
from tqdm import tqdm
from sentimental_cap_predictor.sentimental_cap_predictor.sentimental_cap_predictor.config import PROCESSED_DATA_DIR, INTERIM_DATA_DIR, RAW_DATA_DIR, MODELING_DIR

# Add MODELING_DIR to the Python path
sys.path.append(str(MODELING_DIR))

# Import necessary functions
try:
    from sarima_calculator import fit_sarima, perform_hyperparameter_optimization, evaluate_sarima
    from random_walker import generate_random_walks, apply_l2_regularization
    from stochastic_grid_search import simulate_CIR, simulate_GBM, simulate_JumpDiffusion, cir_params, gbm_params, jump_params
    from lnn import ensemble_predictions_liquid_nn  # Import LNN functions
except ImportError as e:
    logger.error(f"ImportError: {e}")
    sys.exit(1)

app = typer.Typer()

# Initial Setup: Define train ratio and default frequency
TRAIN_RATIO = 0.8  # Ratio of data to be used for training
DEFAULT_FREQUENCY = 'D'  # Default frequency set to daily

#=-~= Data Augmentation Functions Start =-~=

def add_gaussian_noise(data, magnitude=0.01):
    """Add Gaussian noise to the data."""
    noise = np.random.normal(0, magnitude, data.shape)
    return data + noise

def jitter(data, magnitude=0.01):
    """Apply jittering by adding small random noise."""
    jittered_data = data + magnitude * np.random.randn(*data.shape)
    return jittered_data

def temporal_shift(data, shift_magnitude=1):
    """Shift the time series data forward or backward by a certain number of steps."""
    return data.shift(shift_magnitude).fillna(method='bfill' if shift_magnitude > 0 else 'ffill')

def first_order_differencing(data):
    """Apply first-order differencing to remove trends."""
    return data.diff().fillna(0)

def seasonal_differencing(data, period=12):
    """Apply seasonal differencing to remove seasonal patterns."""
    return data.diff(periods=period).fillna(0)

#=-~= Data Augmentation Functions End =-~=


def generate_filename(order=None, seasonal_order=None, trend=None, enforce_stationarity=None, enforce_invertibility=None, prefix="sarima_model"):
    """Generates a unique filename based on model parameters."""
    if order and seasonal_order:
        param_str = f"{order}_{seasonal_order}_{trend}_{enforce_stationarity}_{enforce_invertibility}"
        hashed_params = md5(param_str.encode('utf-8')).hexdigest()
    else:
        hashed_params = "intermediate"
    return f"{prefix}_{hashed_params}.pkl"

def save_grid_search_progress(parameters, scores, save_path):
    """Saves the current grid search parameters and their scores to a file."""
    progress_data = {
        'parameters': parameters,
        'scores': scores
    }
    joblib.dump(progress_data, save_path)

def load_grid_search_progress(save_path):
    """Loads the grid search progress from a file."""
    if save_path.exists():
        return joblib.load(save_path)
    else:
        return {'parameters': [], 'scores': []}

def min_max_scale(df):
    """Min-Max scale the dataframe columns, excluding non-numeric columns like 'Date'."""
    from sklearn.preprocessing import MinMaxScaler
    
    numeric_df = df.select_dtypes(include=[np.number])
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(numeric_df)
    scaled_df = pd.DataFrame(scaled_data, columns=numeric_df.columns, index=df.index)
    return scaled_df, scaler

def adjust_y_intercept(df, columns, last_true_value):
    """Adjusts the y-intercept of each prediction series to match the last true value."""
    for column in columns:
        df[column] += last_true_value - df[column].iloc[0]
    return df

@app.command()
def main(
    interim_dir: Path = typer.Option(INTERIM_DATA_DIR, help="Directory to save intermediate steps and models."),
    processed_dir: Path = typer.Option(PROCESSED_DATA_DIR, help="Directory to save the final features and results."),
    timeout_minutes: int = typer.Option(60, help="Timeout for hyperparameter optimization in minutes."),
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

    logger.info("Scaling data for models.")
    try:
        scaled_data, scaler = min_max_scale(df[['Close']])
        logger.info(f"Data scaling successful. Scaled data shape: {scaled_data.shape}")
    except Exception as e:
        logger.error(f"An error occurred during data scaling: {e}")
        return

    if scaled_data.empty:
        logger.error("Scaled data is empty. Exiting the script.")
        return

    # -=~= Prompting User for Augmentation Choices Start =-~=
    
    # Gaussian Noise
    add_noise = input("Would you like to add Gaussian noise to the data? (y/n): ").strip().lower()
    if add_noise == 'y':
        noise_magnitude = float(input("Enter the magnitude for Gaussian noise (e.g., 0.01): ").strip())
        scaled_data['Close'] = add_gaussian_noise(scaled_data['Close'], magnitude=noise_magnitude)
        logger.info(f"Gaussian noise added with magnitude {noise_magnitude}.")
    
    # Jittering
    add_jitter = input("Would you like to apply jittering to the data? (y/n): ").strip().lower()
    if add_jitter == 'y':
        jitter_magnitude = float(input("Enter the magnitude for jittering (e.g., 0.01): ").strip())
        scaled_data['Close'] = jitter(scaled_data['Close'], magnitude=jitter_magnitude)
        logger.info(f"Jittering applied with magnitude {jitter_magnitude}.")
    
    # Temporal Shifts
    apply_shift = input("Would you like to apply temporal shifts to the data? (y/n): ").strip().lower()
    if apply_shift == 'y':
        shift_magnitude = int(input("Enter the shift magnitude (e.g., 1 for one step forward): ").strip())
        scaled_data['Close'] = temporal_shift(scaled_data['Close'], shift_magnitude=shift_magnitude)
        logger.info(f"Temporal shifts applied with shift magnitude {shift_magnitude}.")
    
    # First-order Differencing
    apply_diff = input("Would you like to apply first-order differencing? (y/n): ").strip().lower()
    if apply_diff == 'y':
        scaled_data['Close'] = first_order_differencing(scaled_data['Close'])
        logger.info("First-order differencing applied.")
    
    # Seasonal Differencing
    apply_seasonal_diff = input("Would you like to apply seasonal differencing? (y/n): ").strip().lower()
    if apply_seasonal_diff == 'y':
        seasonal_period = int(input("Enter the seasonal period (e.g., 12): ").strip())
        scaled_data['Close'] = seasonal_differencing(scaled_data['Close'], period=seasonal_period)
        logger.info(f"Seasonal differencing applied with period {seasonal_period}.")

    # -=~= Prompting User for Augmentation Choices End =-~=

    logger.info("Splitting data into train and test sets.")
    train_size = int(len(scaled_data) * TRAIN_RATIO)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]
    logger.info(f"Training data size: {len(train_data)}, Testing data size: {len(test_data)}")

    if len(train_data) == 0:
        logger.error("Training data is empty. Unable to proceed with model fitting.")
        return

    # Retrieve the last true value from the training data before predictions start
    last_true_value = train_data['Close'].iloc[-1]

    # SARIMA Model: Hyperparameter Optimization and Prediction
    logger.info("Starting hyperparameter optimization for SARIMA model.")
    
    best_params_path = interim_dir / f"{ticker}_best_sarima_params.pkl"
    grid_search_save_path = interim_dir / f"{ticker}_grid_search_progress.pkl"
    
    if best_params_path.exists():
        logger.info(f"Loading previously found best parameters from {best_params_path}")
        best_params = joblib.load(best_params_path)
    else:
        logger.info("Performing hyperparameter optimization...")
        progress = load_grid_search_progress(grid_search_save_path)
        parameters_tried = progress['parameters']
        scores = progress['scores']

        with tqdm(total=100, desc="Optimizing SARIMA Hyperparameters", ncols=100) as pbar:
            param_sets = perform_hyperparameter_optimization(train_data['Close'], test_data['Close'], timeout_minutes)
            if not param_sets:
                logger.error("No valid parameter sets returned during optimization.")
                return

            for param_set in param_sets:
                score = evaluate_sarima(param_set, train_data['Close'], test_data['Close'])
                parameters_tried.append(param_set)
                scores.append(score)
                save_grid_search_progress(parameters_tried, scores, grid_search_save_path)
                pbar.update(1)

            best_index = scores.index(min(scores))
            best_params = parameters_tried[best_index]

        joblib.dump(best_params, best_params_path)
        logger.info(f"Saved best parameters to {best_params_path}")

    logger.info("Fitting the best SARIMA model and making predictions.")
    model_filename = generate_filename(
        order=(best_params['p'], best_params['d'], best_params['q']), 
        seasonal_order=(best_params['P'], best_params['D'], best_params['Q'], best_params['S']),
        trend=best_params['trend'],
        enforce_stationarity=best_params['enforce_stationarity'],
        enforce_invertibility=best_params['enforce_invertibility']
    )
    
    model_path = interim_dir / model_filename
    
    try:
        if model_path.exists():
            logger.info(f"Loading existing SARIMA model from {model_path}")
            best_model = joblib.load(model_path)
        else:
            best_model = fit_sarima(train_data['Close'], (best_params['p'], best_params['d'], best_params['q']), 
                                    (best_params['P'], best_params['D'], best_params['Q'], best_params['S']), 
                                    best_params['trend'], best_params['enforce_stationarity'], 
                                    best_params['enforce_invertibility'])
            joblib.dump(best_model, model_path)
            logger.info(f"Saved SARIMA model to {model_path}")
    except (ModuleNotFoundError, ImportError) as e:
        if "numpy._core" in str(e):
            logger.error("Error loading model due to NumPy version incompatibility. Retraining the model...")
            best_model = fit_sarima(train_data['Close'], (best_params['p'], best_params['d'], best_params['q']), 
                                    (best_params['P'], best_params['D'], best_params['Q'], best_params['S']), 
                                    best_params['trend'], best_params['enforce_stationarity'], 
                                    best_params['enforce_invertibility'])
            joblib.dump(best_model, model_path)
            logger.info(f"Re-saved SARIMA model to {model_path}")
        else:
            raise e

    logger.info("Saving SARIMA predictions for further use.")
    predicted_seasons = best_model.predict(start=len(train_data), end=len(scaled_data) - 1)
    # Adjust SARIMA predictions to start from the last known true value
    sarima_predictions = pd.Series(predicted_seasons, index=test_data.index, name="SARIMA")
    sarima_predictions += last_true_value - sarima_predictions.iloc[0]  # Adjust to make contiguous

    logger.info("Generating random walks using the SARIMA model.")
    steps = len(test_data)
    num_simulations = 100
    simulations = generate_random_walks(best_model, steps, num_simulations)
    simulations += last_true_value - simulations[:, 0].reshape(-1, 1)  # Adjust to make contiguous

    logger.info("Applying L2 regularization to the simulations.")
    optimal_prediction = apply_l2_regularization(simulations, true_values=test_data['Close'])
    optimal_prediction_series = pd.Series(optimal_prediction, index=test_data.index, name="RandomWalk_L2")

    # Stochastic Processes: CIR, GBM, Jump Diffusion
    logger.info("Performing grid search for stochastic processes.")

    # Perform grid search for CIR
    best_cir_params = None
    min_error_cir = float('inf')
    for kappa, theta, sigma, X0 in cir_params:
        X = simulate_CIR(kappa, theta, sigma, X0, T=1, N=steps)
        X += last_true_value - X[0]  # Adjust to make contiguous
        error = np.mean(np.square(X - test_data['Close']))  # Example error metric
        if error < min_error_cir:
            min_error_cir = error
            best_cir_params = (kappa, theta, sigma, X0)
    
    # Perform grid search for GBM
    best_gbm_params = None
    min_error_gbm = float('inf')
    for mu, sigma, S0 in gbm_params:
        S = simulate_GBM(mu, sigma, S0, T=1, N=steps)
        S += last_true_value - S[0]  # Adjust to make contiguous
        error = np.mean(np.square(S - test_data['Close']))  # Example error metric
        if error < min_error_gbm:
            min_error_gbm = error
            best_gbm_params = (mu, sigma, S0)
    
    # Perform grid search for Jump Diffusion
    best_jump_params = None
    min_error_jump = float('inf')
    for lambda_, muJ, sigmaJ, S0 in jump_params:
        S = simulate_JumpDiffusion(lambda_, muJ, sigmaJ, S0, T=1, N=steps)
        S += last_true_value - S[0]  # Adjust to make contiguous
        error = np.mean(np.square(S - test_data['Close']))  # Example error metric
        if error < min_error_jump:
            min_error_jump = error
            best_jump_params = (lambda_, muJ, sigmaJ, S0)

    logger.info(f"Best CIR parameters: {best_cir_params}")
    logger.info(f"Best GBM parameters: {best_gbm_params}")
    logger.info(f"Best Jump Diffusion parameters: {best_jump_params}")

    logger.info("Simulating CIR, GBM, and Jump Diffusion processes with optimized parameters.")
    cir_values = simulate_CIR(*best_cir_params, T=1, N=steps)
    cir_values += last_true_value - cir_values[0]  # Adjust to make contiguous
    gbm_values = simulate_GBM(*best_gbm_params, T=1, N=steps)
    gbm_values += last_true_value - gbm_values[0]  # Adjust to make contiguous
    jump_values = simulate_JumpDiffusion(*best_jump_params, T=1, N=steps)
    jump_values += last_true_value - jump_values[0]  # Adjust to make contiguous

    # Convert CIR, GBM, and Jump Diffusion values to Series
    cir_series = pd.Series(cir_values, index=test_data.index, name="CIR")
    gbm_series = pd.Series(gbm_values, index=test_data.index, name="GBM")
    jump_series = pd.Series(jump_values, index=test_data.index, name="JumpDiffusion")

    # Combine all predictions and true values into a single DataFrame
    df_final = pd.DataFrame({
        'TrueValues': test_data['Close'],
        'SARIMA': sarima_predictions,
        'RandomWalk_L2': optimal_prediction_series,
        'CIR': cir_series,
        'GBM': gbm_series,
        'JumpDiffusion': jump_series
    })

    # Adjust y-intercepts of all predictions to match the last true value in training data
    df_final = adjust_y_intercept(df_final, ['SARIMA', 'RandomWalk_L2', 'CIR', 'GBM', 'JumpDiffusion'], last_true_value)

    # Ensure no NaN values before calculating errors
    df_final = df_final.fillna(method='ffill')

    # Handle edge cases where forward-filling might not suffice (e.g., initial NaNs)
    df_final = df_final.fillna(0)  # Replace any remaining NaNs with zeros

    # Debug Logging to Ensure `df_final` is Defined
    logger.info(f"df_final is defined with shape {df_final.shape}")

    # Generate ensemble predictions using Liquid Neural Network
    logger.info("Generating ensemble predictions using Liquid Neural Network.")
    df_final['LiquidEnsemble'] = ensemble_predictions_liquid_nn(df_final)

    output_csv_path = processed_dir / f"{ticker}_final_predictions.csv"
    logger.info(f"Saving final predictions and features to {output_csv_path}.")
    df_final.to_csv(output_csv_path, index=True)

if __name__ == "__main__":
    app()

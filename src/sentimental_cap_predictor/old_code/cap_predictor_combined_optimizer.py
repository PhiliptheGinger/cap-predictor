import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
import logging
import joblib
from collections import defaultdict
from datetime import datetime, timedelta
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import optuna
from tqdm import tqdm
import os
import traceback

# Ensure interactive backend if possible
if plt.get_backend() != 'Qt5Agg' and 'DISPLAY' in os.environ:
    plt.switch_backend('Qt5Agg')

# Set up logging
LOG_FILE_PATH = 'D:\\Programming Projects\\CAP\\CAP_PREDICTOR\\stable\\cap_predictor_combined_optimizer_vs2.3.py'
LOG_OUTPUT_PATH = 'D:\\Programming Projects\\CAP\\CAP_PREDICTOR\\stable\\cap_predictor_log.log'

output_graph_path = r'D:\Programming Projects\CAP\CAP_PREDICTOR\output_graph.png'
learning_curve_path = r'D:\Programming Projects\CAP\CAP_PREDICTOR\learning_curve.png'

# Ensure the log directory exists
if not os.path.exists(os.path.dirname(LOG_OUTPUT_PATH)):
    os.makedirs(os.path.dirname(LOG_OUTPUT_PATH))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler(LOG_OUTPUT_PATH),
    logging.StreamHandler()
])
logger = logging.getLogger(__name__)

def log_error_with_file_path(msg, exc):
    logger.error(f"{LOG_FILE_PATH} - {msg}: {exc}")
    logger.error(traceback.format_exc())

# Define the threshold for an acceptable confidence interval width
CONFIDENCE_INTERVAL_THRESHOLD = 0.10  # 10%

def load_data_from_yfinance(ticker, start_date, end_date):
    logger.info(f"Entering load_data_from_yfinance with ticker={ticker}, start_date={start_date}, end_date={end_date}")
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        data.index = pd.to_datetime(data.index)
        inferred_freq = pd.infer_freq(data.index)
        if inferred_freq:
            data = data.asfreq(inferred_freq)
        else:
            data = data.asfreq('B')  # Assuming business day frequency if no frequency is inferred
        logger.info(f"Loaded data from yfinance: {data.head()}")
        return data
    except Exception as e:
        log_error_with_file_path("Error loading data from yfinance", e)
        raise

def generate_brownian_motion(length, start_price=100, drift=0.0001, volatility=0.01):
    logger.info(f"Entering generate_brownian_motion with length={length}, start_price={start_price}, drift={drift}, volatility={volatility}")
    try:
        np.random.seed(0)
        returns = np.random.normal(loc=drift, scale=volatility, size=length)
        price = start_price * np.exp(np.cumsum(returns))
        logger.info(f"Generated Brownian motion data: {price[:5]}")
        return price
    except Exception as e:
        log_error_with_file_path("Error generating Brownian motion data", e)
        raise

def min_max_scale(data, feature_range=(0, 1)):
    logger.info(f"Entering min_max_scale with data shape={data.shape}, feature_range={feature_range}")
    try:
        scaler = MinMaxScaler(feature_range=feature_range)
        scaled_data = scaler.fit_transform(data)
        logger.info(f"Scaled data shape={scaled_data.shape}")
        return pd.DataFrame(scaled_data, index=data.index, columns=data.columns), scaler
    except Exception as e:
        log_error_with_file_path("Error scaling data", e)
        raise

def fit_sarima(data, order, seasonal_order):
    logger.info(f"Entering fit_sarima with data shape={data.shape}, order={order}, seasonal_order={seasonal_order}")
    try:
        if data.index.freq is None:
            inferred_freq = pd.infer_freq(data.index)
            if inferred_freq:
                data = data.asfreq(inferred_freq)
            else:
                data = data.asfreq('B')
        
        logger.info(f"Data frequency: {data.index.freq}")

        model = SARIMAX(data, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        fitted_model = model.fit(disp=False)
        logger.info(f"Fitted SARIMA model summary: {fitted_model.summary()}")
        return fitted_model
    except Exception as e:
        log_error_with_file_path("Error fitting SARIMA model", e)
        raise

def predict_seasons_with_intervals(model, steps):
    logger.info(f"Entering predict_seasons_with_intervals with steps={steps}")
    try:
        forecast = model.get_forecast(steps=steps)
        mean_forecast = forecast.predicted_mean
        conf_int = forecast.conf_int()
        logger.info(f"Predicted Mean: {mean_forecast.head()}, Confidence Intervals: {conf_int.head()}")
        return mean_forecast, conf_int
    except Exception as e:
        log_error_with_file_path("Error predicting seasons with intervals", e)
        raise

def track_state_transitions_and_averages(data, window=4):
    logger.info(f"Entering track_state_transitions_and_averages with data shape={data.shape}, window={window}")
    try:
        state_patterns = defaultdict(lambda: {'count': 0, 'sum': 0.0, 'price_points': []})
        
        for i in range(len(data) - window):
            segment = data.iloc[i:i+window]
            mean = np.mean(segment)
            median = np.median(segment)
            pattern = f"mean_{mean:.2f}_median_{median:.2f}"
            
            state_patterns[pattern]['count'] += 1
            state_patterns[pattern]['sum'] += data.iloc[i+window]
            state_patterns[pattern]['price_points'].append(data.iloc[i+window])

        averages = {pattern: info['sum'] / info['count'] for pattern, info in state_patterns.items()}
        logger.info("State transitions and averages tracked successfully.")
        return state_patterns, averages
    except Exception as e:
        log_error_with_file_path("Error tracking state transitions and averages", e)
        raise

def calculate_transition_matrix(state_patterns):
    logger.info(f"Entering calculate_transition_matrix with state_patterns keys={list(state_patterns.keys())}")
    try:
        patterns = list(state_patterns.keys())
        n = len(patterns)
        transition_matrix = np.zeros((n, n))

        for i, pattern_i in enumerate(patterns):
            total_transitions = sum(state_patterns[pattern_i]['count'] for j, pattern_j in enumerate(patterns) if i != j)
            if total_transitions > 0:
                for j, pattern_j in enumerate(patterns):
                    if i != j:
                        transition_matrix[i, j] = state_patterns[pattern_j]['count'] / total_transitions
                transition_matrix[i, :] /= transition_matrix[i, :].sum()
            else:
                transition_matrix[i, :] = 1 / n

        logger.info(f"Transition matrix calculated successfully. Matrix shape: {transition_matrix.shape}")
        return transition_matrix, patterns
    except Exception as e:
        log_error_with_file_path("Error calculating transition matrix", e)
        raise

def simulate_markov_chain(P, π, steps=100):
    logger.info(f"Entering simulate_markov_chain with P shape={P.shape}, π={π}, steps={steps}")
    try:
        state_space = np.arange(P.shape[0])
        current_state = np.random.choice(state_space, p=π)
        states = [current_state]
        
        for _ in range(steps - 1):  # Adjust to generate exactly 'steps' states
            current_state = np.random.choice(state_space, p=P[current_state])
            states.append(current_state)
            
        logger.info(f"Simulated Markov chain states: {states[:5]}")
        return states
    except Exception as e:
        log_error_with_file_path("Error simulating Markov chain", e)
        raise

def perform_hyperparameter_optimization(train_data, test_data, sarima_params, timeout_minutes, max_retries=3):
    logger.info(f"Entering perform_hyperparameter_optimization with train_data shape={train_data.shape}, test_data shape={test_data.shape}, sarima_params={sarima_params}, timeout_minutes={timeout_minutes}")
    try:
        # Validate sarima_params structure
        assert isinstance(sarima_params, dict), "sarima_params should be a dictionary"
        expected_keys = ['p', 'd', 'q', 'P', 'D', 'Q', 'S']
        for key in expected_keys:
            assert key in sarima_params, f"Missing key in sarima_params: {key}"
            assert isinstance(sarima_params[key], list), f"Value for {key} should be a list, got {type(sarima_params[key])}"
            assert all(isinstance(i, int) for i in sarima_params[key]), f"All values for {key} should be integers, got {sarima_params[key]}"

        retries = 0
        best_params = None
        while retries < max_retries and best_params is None:
            try:
                study = optuna.create_study(direction='minimize')
                pbar = tqdm(total=180, desc="Optuna Trials", bar_format="{desc} {percentage:3.0f}%|{bar}| Trial {n_fmt}/{total_fmt}")

                def objective_with_pbar(trial):
                    return objective_with_pbar_inner(trial, train_data, test_data, sarima_params, study, pbar)

                study.optimize(objective_with_pbar, timeout=timeout_minutes * 60, n_trials=180)
                pbar.close()

                if len(study.trials) == 0:
                    logger.error("No trials were completed successfully.")
                else:
                    best_params = study.best_params
                    logger.info(f"Best parameters found: {best_params}")
            except Exception as e:
                log_error_with_file_path(f"Error during hyperparameter optimization attempt {retries + 1}", e)
                retries += 1
                if retries < max_retries:
                    logger.info(f"Retrying hyperparameter optimization ({retries}/{max_retries})")

        if not best_params:
            logger.error("Hyperparameter optimization failed after maximum retries.")
        return best_params
    except Exception as e:
        log_error_with_file_path("Error during hyperparameter optimization", e)
        return None

def objective_with_pbar_inner(trial, train_data, test_data, sarima_params, study, pbar):
    logger.info(f"Entering objective_with_pbar_inner with trial number={trial.number}")
    try:
        logger.info(f"sarima_params: {sarima_params}")

        keys_to_check = ['p', 'd', 'q', 'P', 'D', 'Q', 'S']
        for key in keys_to_check:
            logger.info(f"Checking key '{key}' in sarima_params...")
            if key not in sarima_params:
                logger.error(f"Key '{key}' is missing from sarima_params")
                raise ValueError(f"Key '{key}' is missing from sarima_params")
            elif not isinstance(sarima_params[key], list):
                logger.error(f"Value for key '{key}' should be a list, but got {type(sarima_params[key])}")
                raise TypeError(f"Value for key '{key}' should be a list, but got {type(sarima_params[key])}")
            elif not all(isinstance(item, int) for item in sarima_params[key]):
                logger.error(f"All elements for key '{key}' should be integers, but got {sarima_params[key]}")
                raise ValueError(f"All elements for key '{key}' should be integers, but got {sarima_params[key]}")

        logger.info("Suggesting parameters...")
        p = trial.suggest_categorical('p', sarima_params['p'])
        logger.info(f"Suggested p: {p}")
        d = trial.suggest_categorical('d', sarima_params['d'])
        logger.info(f"Suggested d: {d}")
        q = trial.suggest_categorical('q', sarima_params['q'])
        logger.info(f"Suggested q: {q}")
        P = trial.suggest_categorical('P', sarima_params['P'])
        logger.info(f"Suggested P: {P}")
        D = trial.suggest_categorical('D', sarima_params['D'])
        logger.info(f"Suggested D: {D}")
        Q = trial.suggest_categorical('Q', sarima_params['Q'])
        logger.info(f"Suggested Q: {Q}")
        S = trial.suggest_categorical('S', sarima_params['S'])
        logger.info(f"Suggested S: {S}")

        order = (p, d, q)
        seasonal_order = (P, D, Q, S)

        logger.info(f"Order: {order}, Seasonal Order: {seasonal_order}")

        model = fit_sarima(train_data, order, seasonal_order)
        predictions, _ = predict_seasons_with_intervals(model, len(test_data))

        logger.info(f"Predictions: {predictions.head()}")

        mse = mean_squared_error(test_data, predictions)

        logger.info(f"MSE: {mse}")

        try:
            best_value = study.best_value
        except ValueError:
            best_value = np.nan

        pbar.set_postfix(Best_Trial_MSE=best_value, Current_Trial_MSE=mse)
        pbar.update(1)

        return mse
    except Exception as e:
        logger.error(f"Error in objective function: {e}")
        raise

def fit_predict_lnn(train_data, test_data, epochs=50):
    logger.info(f"Entering fit_predict_lnn with train_data shape={train_data.shape}, test_data shape={test_data.shape}, epochs={epochs}")
    try:
        input_dim = 1
        hidden_dim = 10
        output_dim = 1
        lnn = LiquidNeuralNetwork(input_dim, hidden_dim, output_dim)
        
        # Training loop
        for epoch in range(epochs):
            for t in range(len(train_data) - 1):
                x_t = np.array([[train_data.iloc[t]]])
                y_true = np.array([[train_data.iloc[t + 1]]])
                y_pred = lnn.forward(x_t)
                lnn.update_weights(x_t, y_true, y_pred)
        
        # Prediction loop
        predictions = []
        for t in range(len(test_data)):
            x_t = np.array([[test_data.iloc[t]]])
            y_pred = lnn.forward(x_t)
            predictions.append(y_pred.item())
        
        logger.info(f"LNN Predictions: {predictions[:5]}")
        return pd.Series(predictions, index=test_data.index)
    except Exception as e:
        log_error_with_file_path("Error fitting LNN", e)
        raise

class LiquidNeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim, alpha=0.01):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.alpha = alpha
        self.W_ih = np.random.randn(hidden_dim, input_dim) * np.sqrt(2 / input_dim)  # He initialization
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2 / hidden_dim)  # He initialization
        self.W_ho = np.random.randn(output_dim, hidden_dim) * np.sqrt(2 / hidden_dim)  # He initialization
        self.b_h = np.zeros((hidden_dim, 1))
        self.b_o = np.zeros((output_dim, 1))
        self.h_t = np.zeros((hidden_dim, 1))

    def forward(self, x_t):
        self.h_t = np.tanh(np.dot(self.W_ih, x_t) + np.dot(self.W_hh, self.h_t) + self.b_h)
        y_t = np.dot(self.W_ho, self.h_t) + self.b_o
        return y_t

    def update_weights(self, x_t, y_true, y_pred):
        error = y_true - y_pred
        dW_ho = self.alpha * np.dot(error, self.h_t.T)
        db_o = self.alpha * error
        dW_hh = self.alpha * np.dot(np.dot(self.W_ho.T, error) * (1 - self.h_t ** 2), self.h_t.T)
        dW_ih = self.alpha * np.dot(np.dot(self.W_ho.T, error) * (1 - self.h_t ** 2), x_t.T)
        db_h = self.alpha * np.dot(self.W_ho.T, error) * (1 - self.h_t ** 2)
        self.W_ho += dW_ho
        self.b_o += db_o
        self.W_hh += dW_hh
        self.W_ih += dW_ih
        self.b_h += db_h

def plot_results_with_intervals(train_data, test_data, predicted_seasons, conf_int, scaled_markov_states, brownian_motion_model, transformed_brownian_motion, combined_model, lnn_predictions, transformed_lnn_predictions, output_graph_path):
    logger.info(f"Entering plot_results_with_intervals with output_graph_path={output_graph_path}")
    try:
        plt.figure(figsize=(14, 7))
        plt.plot(train_data, label="Train Data")
        plt.plot(test_data, label="Test Data")
        plt.plot(predicted_seasons, label="SARIMA Predictions")
        plt.plot(scaled_markov_states, label="Scaled Markov States")
        plt.plot(brownian_motion_model, label="Brownian Motion Model")
        plt.plot(transformed_brownian_motion, label="Transformed Brownian Motion")
        plt.plot(combined_model, label="Combined Model")
        plt.plot(lnn_predictions, label="LNN Predictions")
        plt.plot(transformed_lnn_predictions, label="Transformed LNN Predictions")

        plt.fill_between(test_data.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='grey', alpha=0.3, label="Confidence Interval")
        
        plt.legend()
        
        # Try-catch for saving plot
        try:
            plt.savefig(output_graph_path)
            logger.info(f"Plot saved to {output_graph_path}")
        except Exception as e:
            log_error_with_file_path("Error saving results plot", e)

        plt.show()

    except Exception as e:
        log_error_with_file_path("Error plotting results", e)
        raise

def get_errors_with_learning_curve(true_values, sarima_pred, brownian_pred, transformed_brownian_pred, combined_pred, lnn_pred, transformed_lnn_pred, train_data, model_func, test_data):
    logger.info(f"Entering get_errors_with_learning_curve with true_values shape={true_values.shape}")
    try:
        errors = {
            'MSE SARIMA': mean_squared_error(true_values, sarima_pred),
            'MSE Brownian Motion': mean_squared_error(true_values, brownian_pred),
            'MSE Transformed Brownian Motion': mean_squared_error(true_values, transformed_brownian_pred),
            'MSE Combined Model': mean_squared_error(true_values, combined_pred),
            'MSE LNN': mean_squared_error(true_values, lnn_pred),
            'MSE Transformed LNN': mean_squared_error(true_values, transformed_lnn_pred),
            'MAE SARIMA': mean_absolute_error(true_values, sarima_pred),
            'MAE Brownian Motion': mean_absolute_error(true_values, brownian_pred),
            'MAE Transformed Brownian Motion': mean_absolute_error(true_values, transformed_brownian_pred),
            'MAE Combined Model': mean_absolute_error(true_values, combined_pred),
            'MAE LNN': mean_absolute_error(true_values, lnn_pred),
            'MAE Transformed LNN': mean_absolute_error(true_values, transformed_lnn_pred),
            'R2 SARIMA': r2_score(true_values, sarima_pred),
            'R2 Brownian Motion': r2_score(true_values, brownian_pred),
            'R2 Transformed Brownian Motion': r2_score(true_values, transformed_brownian_pred),
            'R2 Combined Model': r2_score(true_values, combined_pred),
            'R2 LNN': r2_score(true_values, lnn_pred),
            'R2 Transformed LNN': r2_score(true_values, transformed_lnn_pred),
            'MAPE SARIMA': mean_absolute_percentage_error(true_values, sarima_pred),
            'MAPE Brownian Motion': mean_absolute_percentage_error(true_values, brownian_pred),
            'MAPE Transformed Brownian Motion': mean_absolute_percentage_error(true_values, transformed_brownian_pred),
            'MAPE Combined Model': mean_absolute_percentage_error(true_values, combined_pred),
            'MAPE LNN': mean_absolute_percentage_error(true_values, lnn_pred),
            'MAPE Transformed LNN': mean_absolute_percentage_error(true_values, transformed_lnn_pred)
        }

        train_sizes = np.linspace(0.1, 1.0, 5)
        train_errors = []
        validation_errors = []

        for train_size in train_sizes:
            partial_train_data = train_data.iloc[:int(len(train_data) * train_size)]
            model = model_func(partial_train_data)
            predictions = model.predict(len(test_data))
            train_errors.append(mean_squared_error(partial_train_data, model.predict(len(partial_train_data))))
            validation_errors.append(mean_squared_error(test_data, predictions))

        learning_curve_errors = {
            'train_sizes': train_sizes,
            'train_errors': train_errors,
            'validation_errors': validation_errors
        }

        logger.info(f"Calculated errors: {errors}")
        return errors, learning_curve_errors
    except Exception as e:
        log_error_with_file_path("Error calculating errors and learning curve", e)
        raise

def plot_learning_curve(errors, learning_curve_path):
    logger.info(f"Entering plot_learning_curve with learning_curve_path={learning_curve_path}")
    try:
        # Log the errors dictionary to ensure it contains the expected data
        logger.info(f"Errors data: {errors}")

        plt.figure(figsize=(14, 7))
        plt.plot(errors['train_sizes'], errors['train_errors'], label="Train Error")
        plt.plot(errors['train_sizes'], errors['validation_errors'], label="Validation Error")
        
        plt.xlabel("Training Size")
        plt.ylabel("Error")
        plt.title("Learning Curve")
        plt.legend()
        
        # Save the figure and log the status
        logger.info(f"Saving learning curve to {learning_curve_path}")
        plt.savefig(learning_curve_path, format='png')
        logger.info(f"Learning curve plot saved successfully to {learning_curve_path}")
        plt.show()
    except Exception as e:
        log_error_with_file_path("Error plotting learning curve", e)
        raise

def main(stock, start_date, end_date, output_graph_path, learning_curve_path, train_ratio, timeout_minutes, sarima_params):
    logger.info(f"Entering main with stock={stock}, start_date={start_date}, end_date={end_date}, output_graph_path={output_graph_path}, train_ratio={train_ratio}, timeout_minutes={timeout_minutes}")
    try:
        start_time = datetime.now()
        logger.info(f"Starting the main function at {start_time}.")
        
        data = load_data_from_yfinance(stock, start_date, end_date)
        logger.info("Data loaded successfully.")
        logger.info(f"Data Preview:\n{data.head()}")

        data['Brownian_Trend'] = generate_brownian_motion(len(data))
        scaled_data, scaler = min_max_scale(data)
        logger.info(f"Data after scaling:\n{scaled_data.head()}")

        # Save the scaled data and the scaler
        scaled_data.to_csv('output_data.csv')
        joblib.dump(scaler, 'scaler.pkl')

        # Set frequency again after scaling
        if scaled_data.index.freq is None:
            inferred_freq = pd.infer_freq(scaled_data.index)
            if inferred_freq:
                scaled_data = scaled_data.asfreq(inferred_freq)
            else:
                scaled_data = scaled_data.asfreq('B')

        train_size = int(len(scaled_data) * train_ratio)
        train_data = scaled_data['Brownian_Trend'][:train_size]
        test_data = scaled_data['Brownian_Trend'][train_size:]
        logger.info(f"Training Data Size: {len(train_data)}, Testing Data Size: {len(test_data)}")
        logger.info(f"Training Data Preview:\n{train_data.head()}")
        logger.info(f"Testing Data Preview:\n{test_data.head()}")

        logger.info("Starting hyperparameter optimization.")
        best_params = perform_hyperparameter_optimization(train_data, test_data, sarima_params, timeout_minutes)
        if not best_params:
            logger.error("Hyperparameter optimization failed.")
            return {
                'combined_model': None,
                'train_data': None,
                'test_data': None,
                'markov_states_series': None,
                'brownian_motion_model': None,
                'transformed_brownian_motion': None,
                'lnn_predictions': None,
                'transformed_lnn_predictions': None,
                'errors': None
            }
        logger.info(f"Best SARIMA Parameters: {best_params}")

        logger.info("Fitting the best SARIMA model.")
        best_model = fit_sarima(train_data, (best_params['p'], best_params['d'], best_params['q']), (best_params['P'], best_params['D'], best_params['Q'], best_params['S']))
        steps = len(test_data)  # Match the steps to the length of the test data
        predicted_seasons, conf_int = predict_seasons_with_intervals(best_model, steps)
        predicted_seasons = pd.Series(predicted_seasons, index=test_data.index)
        predicted_seasons.to_csv('sarima_predictions.csv')

        logger.info("Tracking state transitions and averages.")
        state_patterns, averages = track_state_transitions_and_averages(train_data)
        transition_matrix, patterns = calculate_transition_matrix(state_patterns)
        
        initial_probabilities = np.ones(len(patterns)) / len(patterns)

        logger.info("Simulating Markov chain.")
        markov_states_series = simulate_markov_chain(transition_matrix, initial_probabilities, steps)
        logger.info(f"Markov States: {markov_states_series}")

        brownian_motion_model = generate_brownian_motion(steps, start_price=test_data.iloc[0], drift=0.0001, volatility=0.01)
        brownian_motion_model = pd.Series(brownian_motion_model, index=test_data.index)

        # Ensure markov states series is of the same length as the test data
        if len(markov_states_series) > len(test_data):
            markov_states_series = markov_states_series[:len(test_data)]
        elif len(markov_states_series) < len(test_data):
            markov_states_series = np.pad(markov_states_series, (0, len(test_data) - len(markov_states_series)), 'constant')

        markov_states_series = gaussian_filter1d(markov_states_series, sigma=2)
        markov_states_series = MinMaxScaler().fit_transform(np.array(markov_states_series).reshape(-1, 1)).flatten()
        scaled_markov_states = pd.Series(markov_states_series, index=test_data.index)

        logger.info(f"Type of markov_states_series: {type(markov_states_series)}")
        logger.info(f"Type of brownian_motion_model: {type(brownian_motion_model)}")
        
        logger.info("Transforming Brownian motion model.")
        transformed_brownian_motion = brownian_motion_model.copy()
        brownian_mean = np.mean(transformed_brownian_motion)
        sarima_mean = np.mean(predicted_seasons)
        mean_adjustment = sarima_mean - brownian_mean
        transformed_brownian_motion += mean_adjustment

        # Truncate the transformed_brownian_motion to match test_data.index length
        if len(transformed_brownian_motion) > len(test_data):
            transformed_brownian_motion = transformed_brownian_motion[:len(test_data)]
        elif len(transformed_brownian_motion) < len(test_data):
            transformed_brownian_motion = np.pad(transformed_brownian_motion, (0, len(test_data) - len(transformed_brownian_motion)), 'constant')
        
        transformed_brownian_motion = pd.Series(transformed_brownian_motion, index=test_data.index)
        transformed_brownian_motion.to_csv('transformed_brownian_motion.csv')

        logger.info("Adjusting combined model.")
        alpha = 0.5
        combined_model = (alpha * scaled_markov_states) + ((1 - alpha) * brownian_motion_model)
        combined_model_slope = (combined_model.iloc[-1] - combined_model.iloc[0]) / len(combined_model)
        sarima_slope = (predicted_seasons.iloc[-1] - predicted_seasons.iloc[0]) / len(predicted_seasons)
        slope_ratio = sarima_slope / combined_model_slope
        combined_model = combined_model * slope_ratio
        combined_model += predicted_seasons.iloc[0] - combined_model.iloc[0]
        combined_model_mean = np.mean(combined_model)
        sarima_mean = np.mean(predicted_seasons)
        combined_model += sarima_mean - combined_model_mean

        # Logging combined_model type and content
        logger.info(f"Type of combined_model before access: {type(combined_model)}")
        logger.info(f"Content of combined_model before access: {combined_model}")

        # Add logging around the problematic line
        try:
            if isinstance(combined_model, (pd.DataFrame, pd.Series)):
                combined_model_values = combined_model.values
                logger.info("Successfully accessed combined_model.values")
            else:
                logger.error(f"Unexpected type for combined_model: {type(combined_model)}")
                raise AttributeError(f"Unexpected type for combined_model: {type(combined_model)}")
        except AttributeError as e:
            logger.error("AttributeError when accessing combined_model.values")
            raise

        logger.info("Fitting and predicting with LNN.")
        lnn_predictions = fit_predict_lnn(train_data, test_data, epochs=50)
        mean_difference = np.mean(predicted_seasons.values - lnn_predictions.values)
        transformed_lnn_predictions = lnn_predictions + mean_difference

        plot_results_with_intervals(train_data, test_data, predicted_seasons, conf_int, scaled_markov_states, brownian_motion_model, transformed_brownian_motion, combined_model, lnn_predictions, transformed_lnn_predictions, output_graph_path)

        mse_sarima = mean_squared_error(test_data, predicted_seasons)
        mse_brownian = mean_squared_error(test_data, brownian_motion_model)
        mse_transformed_brownian = mean_squared_error(test_data, transformed_brownian_motion)
        mse_combined = mean_squared_error(test_data, combined_model)
        mse_lnn = mean_squared_error(test_data, lnn_predictions)
        mse_transformed_lnn = mean_squared_error(test_data, transformed_lnn_predictions)

        mae_sarima = mean_absolute_error(test_data, predicted_seasons)
        r2_sarima = r2_score(test_data, predicted_seasons)
        mape_sarima = mean_absolute_percentage_error(test_data, predicted_seasons)

        logger.info(f"MSE SARIMA: {mse_sarima}")
        logger.info(f"MSE Brownian Motion: {mse_brownian}")
        logger.info(f"MSE Transformed Brownian Motion: {mse_transformed_brownian}")
        logger.info(f"MSE Combined Model: {mse_combined}")
        logger.info(f"MSE LNN: {mse_lnn}")
        logger.info(f"MSE Transformed LNN: {mse_transformed_lnn}")

        logger.info(f"MAE SARIMA: {mae_sarima}")
        logger.info(f"R2 SARIMA: {r2_sarima}")
        logger.info(f"MAPE SARIMA: {mape_sarima}")

        print(f"MSE SARIMA: {mse_sarima}")
        print(f"MSE Brownian Motion: {mse_brownian}")
        print(f"MSE Transformed Brownian Motion: {mse_transformed_brownian}")
        print(f"MSE Combined Model: {mse_combined}")
        print(f"MSE LNN: {mse_lnn}")
        print(f"MSE Transformed LNN: {mse_transformed_lnn}")

        print(f"MAE SARIMA: {mae_sarima}")
        print(f"R2 SARIMA: {r2_sarima}")
        print(f"MAPE SARIMA: {mape_sarima}")

        end_time = datetime.now()
        logger.info(f"Main function completed at {end_time}. Duration: {end_time - start_time}.")

        # Get errors for learning curve plotting
        errors, learning_curve_errors = get_errors_with_learning_curve(test_data, predicted_seasons, brownian_motion_model, transformed_brownian_motion, combined_model, lnn_predictions, transformed_lnn_predictions, train_data, lambda x: fit_sarima(x, (best_params['p'], best_params['d'], best_params['q']), (best_params['P'], best_params['D'], best_params['Q'], best_params['S'])), test_data)
        
        # Example of manually checking the data
        print("Train Sizes:", errors['train_sizes'])
        print("Train Errors:", errors['train_errors'])
        print("Validation Errors:", errors['validation_errors'])

        # Plot the learning curve
        plot_learning_curve(learning_curve_errors, learning_curve_path)

        print(f"Output graph saved to: {output_graph_path}")
        print(f"Learning curve graph saved to: {learning_curve_path}")

        return {
            'combined_model': combined_model,
            'train_data': train_data,
            'test_data': test_data,
            'markov_states_series': markov_states_series,
            'brownian_motion_model': brownian_motion_model,
            'transformed_brownian_motion': transformed_brownian_motion,
            'lnn_predictions': lnn_predictions,
            'transformed_lnn_predictions': transformed_lnn_predictions,
            'errors': errors
        }
    except Exception as e:
        log_error_with_file_path("Error in main function", e)
        raise

if __name__ == "__main__":
    try:
        stock = 'AAPL'
        start_date = '2010-01-01'
        end_date = '2020-12-31'
        output_graph_path = 'output_graph.png'
        learning_curve_path = 'learning_curve.png'
        train_ratio = 0.8
        timeout_minutes = 60
        sarima_params = {
            'p': [0, 1, 2],
            'd': [0, 1],
            'q': [0, 1, 2],
            'P': [0, 1],
            'D': [0, 1],
            'Q': [0, 1],
            'S': [12]
        }

        main(stock, start_date, end_date, output_graph_path, learning_curve_path, train_ratio, timeout_minutes, sarima_params)
    except Exception as e:
        log_error_with_file_path("Error executing the script", e)
        raise

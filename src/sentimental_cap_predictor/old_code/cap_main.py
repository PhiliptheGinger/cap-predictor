import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time
import logging
import importlib.util
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib
from colorama import Fore, Style, init
import yfinance as yf
from tqdm import tqdm
import datetime
import argparse
import traceback

# Initialize colorama
init(autoreset=True)

# Set up logging
LOG_FILE_PATH = '/mnt/data/cap_main_vs2.3.py'
LOG_OUTPUT_PATH = 'D:/Programming Projects/CAP/CAP_PREDICTOR/stable/cap_predictor_log.txt'

# Ensure the log directory exists
if not os.path.exists(os.path.dirname(LOG_OUTPUT_PATH)):
    os.makedirs(os.path.dirname(LOG_OUTPUT_PATH))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler(LOG_OUTPUT_PATH),
    logging.StreamHandler()
])
logger = logging.getLogger(__name__)

def log_error_with_file_path(msg, exc):
    tb_str = traceback.format_exception(type(exc), exc, exc.__traceback__)
    logger.error(f"{LOG_FILE_PATH} - {msg}: {''.join(tb_str)}")

# Define the paths to the scripts
CAP_PREDICTOR_PATH = r"D:/Programming Projects/CAP/CAP_PREDICTOR/stable/cap_predictor_combined_optimizer_vs2.3.py"
LNN_INTERACTION_PATH = r"D:/Programming Projects/CAP/CAP_PREDICTOR/stable/LNN_Interraction_vs2.3.py"
SENTIMENT_ANALYSIS_PATH = r"D:/Programming Projects/CAP/CAP_PREDICTOR/stable/sentiment_analysis_vs3.3.py"

def load_module(module_path):
    try:
        logger.info(f"Attempting to load module from path: {module_path}")
        
        if not os.path.exists(module_path):
            logger.error(f"Module file does not exist: {module_path}")
            raise FileNotFoundError(f"Module file does not exist: {module_path}")

        module_name = os.path.splitext(os.path.basename(module_path))[0]
        logger.info(f"Module name: {module_name}")
        
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        
        if spec is None:
            logger.error(f"Could not create a module spec for {module_path}")
            raise ImportError(f"Could not create a module spec for {module_path}")

        if spec.loader is None:
            logger.error(f"No loader found for module spec of {module_path}")
            raise ImportError(f"No loader found for module spec of {module_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        logger.info(f"Module {module_name} loaded successfully from {module_path}")
        return module
    except Exception as e:
        log_error_with_file_path("Error loading module", e)
        raise

def adjust_predictions_based_on_sentiment(predictions, sentiment_score, adjustment_factor=0.01):
    try:
        adjustment = sentiment_score * adjustment_factor
        if sentiment_score < 0:  # Negative sentiment
            adjusted_predictions = predictions * (1 + adjustment)  # Apply downward adjustment
        else:  # Positive sentiment
            adjusted_predictions = predictions * (1 + adjustment)  # Apply upward adjustment
        return adjusted_predictions
    except Exception as e:
        log_error_with_file_path("Error adjusting predictions based on sentiment", e)
        raise

def plot_results(train_data, test_data, sarima_predictions, lnn_predictions, transformed_brownian_motion, transformed_lnn_predictions, adjusted_combined_model, adjusted_lnn_pred, output_path):
    try:
        plt.figure(figsize=(15, 7))
        plt.plot(train_data.index, train_data, label='Training Data')
        plt.plot(test_data.index, test_data, label='Test Data')
        plt.plot(sarima_predictions.index, sarima_predictions, label='SARIMA Predictions')
        plt.plot(lnn_predictions.index, lnn_predictions, label='LNN Predictions')
        plt.plot(transformed_brownian_motion.index, transformed_brownian_motion, label='Transformed Brownian Motion')
        plt.plot(transformed_lnn_predictions.index, transformed_lnn_predictions, label='Transformed LNN Predictions')
        plt.plot(adjusted_combined_model.index, adjusted_combined_model, label='Adjusted Combined Model')
        plt.plot(adjusted_lnn_pred.index, adjusted_lnn_pred, label='Adjusted LNN Predictions')
        plt.legend()
        plt.savefig(output_path)
        plt.show()
        logger.info(f"Results plotted and saved to {output_path}")
    except Exception as e:
        log_error_with_file_path("Error plotting results", e)
        raise

def plot_learning_curves(train_errors, val_errors, output_path):
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(train_errors, label='Training Error')
        plt.plot(val_errors, label='Validation Error')
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.legend()
        plt.savefig(output_path)
        plt.close()  # Close the figure to release memory
        logger.info(f"Learning curves plotted and saved to {output_path}")
    except Exception as e:
        log_error_with_file_path("Error plotting learning curves", e)
        raise

def load_data_from_yfinance(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        data.index = pd.to_datetime(data.index)
        if len(data) < 3:
            raise ValueError("Need at least 3 dates to infer frequency")
        else:
            inferred_freq = pd.infer_freq(data.index)
            if inferred_freq:
                data = data.asfreq(inferred_freq)
            else:
                data = data.asfreq('B')  # Default to business day frequency
        return data
    except Exception as e:
        log_error_with_file_path("Error loading data from yfinance", e)
        raise

def calculate_metrics(y_true, y_pred, num_predictors):
    try:
        metrics = {}
        metrics['MAE'] = mean_absolute_error(y_true, y_pred)
        metrics['MSE'] = mean_squared_error(y_true, y_pred)
        metrics['RMSE'] = np.sqrt(metrics['MSE'])
        metrics['R2'] = r2_score(y_true, y_pred)
        if len(y_true) > num_predictors + 1:  # Ensure there's no division by zero
            metrics['Adjusted R2'] = 1 - ((1 - metrics['R2']) * (len(y_true) - 1) / (len(y_true) - num_predictors - 1))
        else:
            metrics['Adjusted R2'] = None
        metrics['Pearson'] = pearsonr(y_true, y_pred)[0]
        metrics['MAPE'] = mean_absolute_percentage_error(y_true, y_pred)
        return metrics
    except Exception as e:
        log_error_with_file_path("Error calculating metrics", e)
        raise

def analyze_stock(stock, sentiment_analysis, cap_predictor, lnn_interaction, start_date, end_date, output_graph_path, train_ratio, timeout_minutes, sarima_params):
    sentiment_score = sentiment_analysis.get_sentiment_score(stock)
    try:
        # Use the cap predictor to get initial predictions
        with tqdm(total=100, desc=f"Analyzing {stock}", colour='magenta') as pbar:
            logger.info(f"Calling cap_predictor.main for stock: {stock}")
            result = cap_predictor.main(
                stock, 
                start_date, 
                end_date, 
                output_graph_path, 
                train_ratio, 
                timeout_minutes, 
                sarima_params
            )
            logger.info(f"cap_predictor.main returned for stock: {stock} with result: {result}")
            
            # Check if the result is None or contains None elements
            if not result or any(r is None for r in result):
                logger.error("Hyperparameter optimization did not complete or returned invalid results.")
                continue_optimization = input(Fore.YELLOW + "Do you want to continue hyperparameter optimization? (yes/no): ").strip().lower()
                if continue_optimization == 'yes':
                    timeout_minutes = int(input(Fore.YELLOW + "Enter additional timeout for hyperparameter optimization in minutes: ").strip())
                    logger.info(f"Calling cap_predictor.main again for stock: {stock} with extended timeout")
                    result = cap_predictor.main(
                        stock, 
                        start_date, 
                        end_date, 
                        output_graph_path, 
                        train_ratio, 
                        timeout_minutes, 
                        sarima_params
                    )
                    logger.info(f"cap_predictor.main returned again for stock: {stock} with result: {result}")
                    if not result or any(r is None for r in result):
                        print(Fore.RED + "Hyperparameter optimization did not complete again. Exiting.")
                        return
                else:
                    print(Fore.RED + "Exiting optimization for stock: " + stock)
                    return

            if len(result) != 9:
                logger.error(f"Expected 9 elements from cap_predictor.main, got {len(result)}")
                raise ValueError(f"Expected 9 elements from cap_predictor.main, got {len(result)}")

            combined_model, train_data, test_data, markov_states, brownian_motion, transformed_brownian, lnn_predictions, transformed_lnn_predictions, errors = result

            # Ensure no None values in the results
            if any(val is None for val in [combined_model, train_data, test_data, markov_states, brownian_motion, transformed_brownian, lnn_predictions, transformed_lnn_predictions, errors]):
                logger.error("One or more essential elements returned as None. Exiting analysis.")
                return

            pbar.update(25)

            # Logging combined_model type
            logger.info(f"Type of combined_model: {type(combined_model)}")
            
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

            # Scale the data
            scaler = MinMaxScaler()
            combined_model_scaled = scaler.fit_transform(combined_model_values.reshape(-1, 1)).flatten()
            train_data_scaled = scaler.fit_transform(train_data.values.reshape(-1, 1)).flatten()
            test_data_scaled = scaler.transform(test_data.values.reshape(-1, 1)).flatten()
            
            # Save the scaler for future use
            joblib.dump(scaler, f'{stock}_scaler.pkl')
            
            # Save data for LNN processing
            combined_model_scaled = pd.Series(combined_model_scaled, index=test_data.index)
            combined_model_scaled.to_csv(f'{stock}_combined_model_scaled.csv')

            train_data_scaled = pd.Series(train_data_scaled, index=train_data.index)
            train_data_scaled.to_csv(f'{stock}_train_data_scaled.csv')

            test_data_scaled = pd.Series(test_data_scaled, index=test_data.index)
            test_data_scaled.to_csv(f'{stock}_test_data_scaled.csv')

            markov_states = pd.Series(markov_states)
            markov_states.to_csv(f'{stock}_markov_states.csv')

            brownian_motion = pd.Series(brownian_motion, index=test_data.index)
            brownian_motion.to_csv(f'{stock}_brownian_motion.csv')

            transformed_brownian = pd.Series(transformed_brownian, index=test_data.index)
            transformed_brownian.to_csv(f'{stock}_transformed_brownian.csv')
            
            pbar.update(25)

            # Call LNN script for further predictions
            logger.info(f"Calling LNN script for stock {stock}")
            lnn_predictions = lnn_interaction.fit_predict_lnn(train_data, test_data)  # Pass both train_data and test_data
            logger.info(f"LNN predictions for stock {stock}: {lnn_predictions}")
            
            adjusted_combined_model = adjust_predictions_based_on_sentiment(combined_model_scaled, sentiment_score)
            adjusted_lnn_pred = adjust_predictions_based_on_sentiment(pd.Series(lnn_predictions, index=test_data.index), sentiment_score)

            output_graph_path = f"{stock}_{output_graph_path}"
            plot_results(train_data, test_data, combined_model_scaled, lnn_predictions, transformed_brownian, transformed_lnn_predictions, adjusted_combined_model, adjusted_lnn_pred, output_graph_path)

            pbar.update(25)

            # Calculate metrics
            metrics_combined = calculate_metrics(test_data_scaled, combined_model_scaled, len(sarima_params))
            metrics_lnn = calculate_metrics(test_data_scaled, lnn_predictions, len(sarima_params))
            metrics_adjusted_combined = calculate_metrics(test_data_scaled, adjusted_combined_model, len(sarima_params))
            metrics_adjusted_lnn = calculate_metrics(test_data_scaled, adjusted_lnn_pred, len(sarima_params))
            
            logger.info(f"Metrics (Combined Model): {metrics_combined}")
            logger.info(f"Metrics (LNN): {metrics_lnn}")
            logger.info(f"Metrics (Adjusted Combined Model): {metrics_adjusted_combined}")
            logger.info(f"Metrics (Adjusted LNN): {metrics_adjusted_lnn}")

            pbar.update(25)
            
            # Print the results with colors
            print(Fore.YELLOW + f"\nStock: {stock}")
            for metric_name, metric_value in metrics_combined.items():
                print(Fore.CYAN + f"{metric_name} (Combined Model): {metric_value:.4f}")
            for metric_name, metric_value in metrics_lnn.items():
                print(Fore.CYAN + f"{metric_name} (LNN): {metric_value:.4f}")
            for metric_name, metric_value in metrics_adjusted_combined.items():
                print(Fore.CYAN + f"{metric_name} (Adjusted Combined Model): {metric_value:.4f}")
            for metric_name, metric_value in metrics_adjusted_lnn.items():
                print(Fore.CYAN + f"{metric_name} (Adjusted LNN): {metric_value:.4f}")
            
            # Plot learning curves
            learning_curve_path = f"{stock}_learning_curves.png"
            plot_learning_curves(errors['train'], errors['val'], learning_curve_path)
            
    except Exception as e:
        log_error_with_file_path(f"Error analyzing stock {stock}", e)


def continuous_analysis(stocks, start_date, end_date, output_graph_path, train_ratio, timeout_minutes, sarima_params, interval=60, test_mode=False):
    try:
        sentiment_analysis = load_module(SENTIMENT_ANALYSIS_PATH)
        if test_mode:
            sentiment_analysis.limit_articles = 10
        cap_predictor = load_module(CAP_PREDICTOR_PATH)
        lnn_interaction = load_module(LNN_INTERACTION_PATH)
        
        while True:
            print(Fore.YELLOW + "Continuous Analysis Menu")
            print(Fore.CYAN + "1. Select stocks to analyze continuously")
            print(Fore.CYAN + "2. Continue hyperparameter optimization from where it left off")
            menu_choice = input(Fore.YELLOW + "Enter your choice: ").strip()

            if menu_choice == '1':
                stocks = input(Fore.YELLOW + "Enter the stock tickers to analyze continuously, separated by commas: ").split(',')
                stocks = [stock.strip().upper() for stock in stocks]
            elif menu_choice == '2':
                print(Fore.YELLOW + "Continuing hyperparameter optimization from where it left off for all stocks.")
            else:
                print(Fore.RED + "Invalid choice. Please select a valid option.")
                continue

            for stock in stocks:
                logger.info(f"Running continuous analysis for {stock}")
                analyze_stock(stock, sentiment_analysis, cap_predictor, lnn_interaction, start_date, end_date, output_graph_path, train_ratio, timeout_minutes, sarima_params)
            logger.info(f"Sleeping for {interval} seconds before the next update.")
            time.sleep(interval)
    except KeyboardInterrupt:
        logger.info("Continuous analysis stopped by user.")
    except Exception as e:
        log_error_with_file_path("Error in continuous analysis", e)
        raise

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Stock Analysis Script")
        parser.add_argument('--test', action='store_true', help="Run in test mode with limited articles")
        args = parser.parse_args()

        start_time = time.time()

        stocks_to_analyze = input(Fore.YELLOW + "Enter the stock tickers to analyze, separated by commas: ").split(',')
        stocks_to_analyze = [stock.strip().upper() for stock in stocks_to_analyze]

        if not stocks_to_analyze:
            logger.info("No stocks provided. Exiting the program.")
            print(Fore.RED + "No stocks provided. Exiting the program.")
            exit()

        start_date = input(Fore.YELLOW + "Enter the start date (YYYY-MM-DD) or 'all' for full historical data: ").strip()
        if start_date.lower() == 'all':
            start_date = None
        end_date = input(Fore.YELLOW + "Enter the end date (YYYY-MM-DD) or 'p' for present: ").strip()
        if end_date.lower() == 'p':
            end_date = datetime.date.today().strftime('%Y-%m-%d')

        try:
            timeout_minutes = int(input(Fore.YELLOW + "Enter the timeout for hyperparameter optimization in minutes (default is 60): ").strip() or 60)
        except ValueError:
            timeout_minutes = 60

        OUTPUT_GRAPH_PATH = 'output_graph.png'
        TRAIN_RATIO = 0.8
        SARIMA_PARAMS = {
            'p': [0, 1, 2],
            'd': [0, 1],
            'q': [0, 1, 2],
            'P': [0, 1],
            'D': [0, 1],
            'Q': [0, 1],
            'S': [12]
        }

        sentiment_analysis = load_module(SENTIMENT_ANALYSIS_PATH)
        if args.test:
            sentiment_analysis.limit_articles = 10
        cap_predictor = load_module(CAP_PREDICTOR_PATH)
        lnn_interaction = load_module(LNN_INTERACTION_PATH)

        for stock in stocks_to_analyze:
            if len(stock) < 2:  # Add a simple validation for ticker
                logger.error(f"Invalid stock ticker: {stock}. Skipping.")
                continue
            analyze_stock(stock, sentiment_analysis, cap_predictor, lnn_interaction, start_date, end_date, OUTPUT_GRAPH_PATH, TRAIN_RATIO, timeout_minutes, SARIMA_PARAMS)

        continuous_stocks = input(Fore.YELLOW + "Enter the stock tickers to analyze " + Fore.GREEN + "continuously" + Fore.YELLOW + ", separated by commas: ").split(',')
        continuous_stocks = [stock.strip().upper() for stock in continuous_stocks]

        if continuous_stocks:
            continuous_analysis(continuous_stocks, start_date, end_date, OUTPUT_GRAPH_PATH, TRAIN_RATIO, timeout_minutes, SARIMA_PARAMS, test_mode=args.test)

        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Total execution time: {elapsed_time:.2f} seconds")
        print(Fore.GREEN + f"Total execution time: {elapsed_time:.2f} seconds")
        
    except Exception as e:
        log_error_with_file_path("Error executing the script", e)
        raise

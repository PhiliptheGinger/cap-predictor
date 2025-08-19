from pathlib import Path
import typer
from loguru import logger
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt

from sentimental_cap_predictor.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

def plot_results_with_intervals(train_data, test_data, predicted_seasons, conf_int, scaled_markov_states, brownian_motion_model, transformed_brownian_motion, combined_model, lnn_predictions, transformed_lnn_predictions, output_graph_path):
    """Plots results including confidence intervals and various model predictions."""
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
    plt.savefig(output_graph_path)
    plt.show()

def get_errors_with_learning_curve(true_values, sarima_pred, brownian_pred, transformed_brownian_pred, combined_pred, lnn_pred, transformed_lnn_pred, train_data, model_func, test_data):
    """Calculates various error metrics and generates learning curve data."""
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

    return errors, learning_curve_errors

def plot_learning_curve(errors, learning_curve_path):
    """Plots the learning curve based on training and validation errors."""
    plt.figure(figsize=(14, 7))
    plt.plot(errors['train_sizes'], errors['train_errors'], label="Train Error")
    plt.plot(errors['train_sizes'], errors['validation_errors'], label="Validation Error")
    
    plt.xlabel("Training Size")
    plt.ylabel("Error")
    plt.title("Learning Curve")
    plt.legend()
    
    plt.savefig(learning_curve_path)
    plt.show()

@app.command()
def main(
    output_path: Path = FIGURES_DIR / "plot.png",
):
    # Prompt the user for the ticker symbol
    ticker_symbol = input(f"{typer.style('Please enter the ticker symbol:', fg=typer.colors.YELLOW)} ").strip().upper()
    
    # Construct the file path based on the ticker symbol
    input_path = PROCESSED_DATA_DIR / f"{ticker_symbol}_features.csv"
    
    # Load the data
    logger.info("Loading data from file...")
    try:
        df = pd.read_csv(input_path)
        logger.success("Data loaded successfully.")
    except FileNotFoundError:
        logger.error(f"File not found: {input_path}. Please ensure the file exists in the PROCESSED_DATA_DIR.")
        return
    except Exception as e:
        logger.error(f"An error occurred while loading data: {e}")
        return

    # Assuming that the dataframe has columns like 'actual', 'SARIMA', etc.
    required_columns = [
        'actual', 'SARIMA', 'BrownianMotion',
        'TransformedBrownianMotion', 'CombinedModel', 'LNN', 'TransformedLNN'
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"The required columns are missing from the dataset: {missing_columns}")
        return
    
    logger.info("Calculating errors and generating learning curve...")
    errors, learning_curve = get_errors_with_learning_curve(
        df['actual'], df['SARIMA'], df['BrownianMotion'], df['TransformedBrownianMotion'],
        df['CombinedModel'], df['LNN'], df['TransformedLNN'], df['actual'], lambda x: x, df['actual']
    )

    logger.info("Plotting learning curve...")
    plot_learning_curve(learning_curve, output_path)
    logger.success(f"Learning curve plot saved to {output_path}.")

if __name__ == "__main__":
    app()

from pathlib import Path
import typer
from loguru import logger
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import ast

from sentimental_cap_predictor.sentimental_cap_predictor.sentimental_cap_predictor.config import PROCESSED_DATA_DIR

app = typer.Typer()

def parse_string_to_dict(df, columns):
    """Converts string representations of dictionaries to actual dictionaries in the DataFrame."""
    for column in columns:
        df[column] = df[column].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return df

def print_csv_parts(df):
    """Prints each part of the CSV file, including nested structures."""
    for index, row in df.iterrows():
        print(f"Row {index}:")
        for column, value in row.items():
            if isinstance(value, dict):
                print(f"  {column}:")
                for key, array in value.items():
                    print(f"    {key}: {array}")
            else:
                print(f"  {column}: {value}")
        print("\n")

def plot_results(df, output_graph_path):
    """Plots the actual values and predictions from various models."""
    plt.figure(figsize=(14, 7))
    
    # Ensure the index is set to the correct column, if needed
    if isinstance(df.index, pd.RangeIndex):
        df.index = pd.to_datetime(df.index)
    
    plt.plot(df.index, df['TrueValues'], label="True Values", color="blue")
    plt.plot(df.index, df['SARIMA'], label="SARIMA Predictions", color="orange")
    plt.plot(df.index, df['RandomWalk_L2'], label="Random Walk L2 Predictions", color="red")
    plt.plot(df.index, df['CIR'], label="CIR Predictions", color="green")
    plt.plot(df.index, df['GBM'], label="GBM Predictions", color="brown")
    plt.plot(df.index, df['JumpDiffusion'], label="Jump Diffusion Predictions", color="pink")
    plt.plot(df.index, df['LiquidEnsemble'], label="Liquid Ensemble", color="cyan")
    
    plt.xlabel("Time")
    plt.ylabel("Values")
    plt.title("True Values vs. Various Model Predictions")
    plt.legend()
    
    plt.savefig(output_graph_path)
    plt.show()

def plot_learning_curve(train_sizes, train_errors, val_errors, output_graph_path):
    """Plots the learning curve for the model."""
    plt.figure(figsize=(14, 7))
    
    plt.plot(train_sizes, train_errors, label="Training Error", color="red")
    plt.plot(train_sizes, val_errors, label="Validation Error", color="blue")
    
    plt.xlabel("Training Set Size")
    plt.ylabel("Error")
    plt.title("Learning Curve")
    plt.legend()
    
    plt.savefig(output_graph_path)
    plt.show()

def calculate_errors(true_values, predictions_dict):
    """Calculates various error metrics for all provided models."""
    errors = {}
    
    # Forward fill missing values in both true_values and predictions
    true_values_filled = true_values.fillna(method='ffill')
    
    for model_name, predictions in predictions_dict.items():
        predictions_filled = predictions.fillna(method='ffill')
        
        # Calculate errors with forward-filled data
        errors[f'MSE {model_name}'] = mean_squared_error(true_values_filled, predictions_filled)
        errors[f'MAE {model_name}'] = mean_absolute_error(true_values_filled, predictions_filled)
        errors[f'R2 {model_name}'] = r2_score(true_values_filled, predictions_filled)
        errors[f'MAPE {model_name}'] = mean_absolute_percentage_error(true_values_filled, predictions_filled)
    
    return errors

@app.command()
def main(
    output_path: Path = typer.Option(PROCESSED_DATA_DIR / "final_plot.png", help="Path to save the output graph."),
    learning_curve_path: Path = typer.Option(PROCESSED_DATA_DIR / "learning_curve.png", help="Path to save the learning curve plot."),
    num_lines: int = typer.Option(5, help="Number of lines to display from the CSV."),
):
    # Prompt the user for the ticker symbol
    ticker_symbol = input(f"{typer.style('Please enter the ticker symbol:', fg=typer.colors.YELLOW)} ").strip().upper()
    
    # Construct the file path based on the ticker symbol
    input_path = PROCESSED_DATA_DIR / f"{ticker_symbol}_final_predictions.csv"
    
    logger.info(f"Attempting to load data from: {input_path}")

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

    # Ensure required columns are present
    required_columns = ['TrueValues', 'SARIMA', 'RandomWalk_L2', 'CIR', 'GBM', 'JumpDiffusion', 'LiquidEnsemble']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"The required columns are missing from the dataset: {missing_columns}")
        return

    # If 'Date' column is present, set it as index
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

    # Display the first few lines of the CSV file
    logger.info(f"Displaying the first {num_lines} lines of the CSV file:")
    print_csv_parts(df.head(num_lines))

    # Prepare a dictionary of predictions for error calculations
    predictions_dict = {
        'SARIMA': df['SARIMA'],
        'RandomWalk_L2': df['RandomWalk_L2'],
        'CIR': df['CIR'],
        'GBM': df['GBM'],
        'JumpDiffusion': df['JumpDiffusion'],
        'LiquidEnsemble': df['LiquidEnsemble']
    }

    logger.info("Calculating errors...")
    errors = calculate_errors(df['TrueValues'], predictions_dict)

    logger.info("Errors calculated:")
    for error_name, error_value in errors.items():
        logger.info(f"{error_name}: {error_value}")

    # Plot the results
    logger.info("Plotting the results...")
    plot_results(df, output_path)

    # Plot the learning curve
    logger.info("Plotting the learning curve...")
    # Assuming you have precomputed training sizes and errors; replace these with your own logic
    train_sizes = np.linspace(0.1, 1.0, 10) * len(df)  # Example training sizes
    train_errors = np.random.rand(10)  # Replace with actual training errors
    val_errors = np.random.rand(10)  # Replace with actual validation errors
    plot_learning_curve(train_sizes, train_errors, val_errors, learning_curve_path)

    logger.success(f"Plot saved to {output_path} and learning curve saved to {learning_curve_path}.")

if __name__ == "__main__":
    app()

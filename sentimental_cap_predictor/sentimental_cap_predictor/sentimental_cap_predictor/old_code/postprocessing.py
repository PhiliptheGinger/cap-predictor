import sys
import pandas as pd
import typer
from pathlib import Path
from loguru import logger
import matplotlib.pyplot as plt

app = typer.Typer()

def adjust_y_intercept(df, prediction_columns, last_true_value):
    """Adjusts the y-intercept of prediction columns to match the last known true value."""
    for column in prediction_columns:
        # Check for NaN or None values and handle them
        if df[column].isnull().any():
            logger.warning(f"{column} contains NaN or None values. Filling missing values with forward fill.")
            df[column] = df[column].ffill().bfill()  # Forward and backward fill missing values
        
        initial_prediction = df[column].iloc[0]
        if pd.isnull(initial_prediction):
            logger.error(f"Initial prediction for {column} is None or NaN. Skipping adjustment for this column.")
            continue

        adjustment_amount = last_true_value - initial_prediction
        logger.info(f"Adjusting {column}: initial value = {initial_prediction}, "
                    f"adjustment = {adjustment_amount}, last true value = {last_true_value}")
        
        # Apply the adjustment
        df[column] += adjustment_amount
        
        # Log the first few values to verify
        logger.info(f"{column} after adjustment: {df[column].head(3).tolist()}")
        
    return df

def plot_intermediate(df, title, filename, processed_dir):
    """Helper function to plot intermediate results."""
    plt.figure(figsize=(10, 6))
    for column in df.columns:
        plt.plot(df.index, df[column], label=column)
    plt.title(title)
    plt.legend()
    plt.savefig(processed_dir / filename)
    plt.close()

@app.command()
def main(
    input_csv: Path = typer.Option(..., help="Path to the CSV file containing predictions."),
    output_csv: Path = typer.Option(..., help="Path to save the adjusted CSV file."),
    processed_dir: Path = typer.Option(..., help="Directory to save the plots."),
    true_values_column: str = typer.Option("TrueValues", help="Column name containing the true values.")
):
    # Ensure processed directory exists
    if not processed_dir.exists():
        logger.info(f"Processed directory does not exist. Creating: {processed_dir}")
        processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info(f"Loading data from {input_csv}")
    try:
        df = pd.read_csv(input_csv, parse_dates=True, index_col=0)
    except FileNotFoundError:
        logger.error(f"File not found: {input_csv}. Please ensure the file exists.")
        return
    except Exception as e:
        logger.error(f"An error occurred while loading data: {e}")
        return

    # Ensure the true values column exists
    if true_values_column not in df.columns:
        logger.error(f"Column '{true_values_column}' not found in the dataset.")
        return

    # Identify prediction columns (excluding true values and any other non-prediction columns)
    prediction_columns = [col for col in df.columns if col != true_values_column]

    # Plot predictions before any adjustments
    plot_intermediate(df, "Before Adjustments", "before_adjustments.png", processed_dir)

    # Retrieve the last true value
    last_true_value = df[true_values_column].iloc[-1]
    logger.info(f"Last known true value: {last_true_value}")

    # Adjust the y-intercept of all prediction columns
    logger.info("Adjusting y-intercepts of prediction columns to match the last known true value.")
    df = adjust_y_intercept(df, prediction_columns, last_true_value)

    # Plot after y-intercept adjustment
    plot_intermediate(df, "After Y-Intercept Adjustment", "after_y_intercept.png", processed_dir)

    # Save the adjusted DataFrame to CSV
    logger.info(f"Saving adjusted predictions to {output_csv}")
    df.to_csv(output_csv)

if __name__ == "__main__":
    app()

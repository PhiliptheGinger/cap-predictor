import pandas as pd
import typer
from pathlib import Path
from loguru import logger

app = typer.Typer()

def adjust_y_intercept(df, prediction_columns, last_true_value):
    """Adjusts the y-intercept of prediction columns to match the last known true value."""
    for column in prediction_columns:
        df[column] += last_true_value - df[column].iloc[0]
    return df

@app.command()
def main(
    input_csv: Path = typer.Option(..., help="Path to the CSV file containing predictions."),
    output_csv: Path = typer.Option(..., help="Path to save the adjusted CSV file."),
    true_values_column: str = typer.Option("TrueValues", help="Column name containing the true values."),
):
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

    # Retrieve the last true value
    last_true_value = df[true_values_column].iloc[-1]
    logger.info(f"Last known true value: {last_true_value}")

    # Adjust the y-intercept of all prediction columns
    logger.info("Adjusting y-intercepts of prediction columns to match the last known true value.")
    df = adjust_y_intercept(df, prediction_columns, last_true_value)

    # Save the adjusted DataFrame to CSV
    logger.info(f"Saving adjusted predictions to {output_csv}")
    df.to_csv(output_csv)

if __name__ == "__main__":
    app()

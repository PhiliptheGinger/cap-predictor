from pathlib import Path
import typer
from loguru import logger
import pandas as pd
import matplotlib.pyplot as plt

from sentimental_cap_predictor.config import PROCESSED_DATA_DIR

app = typer.Typer()

def print_csv_parts(df):
    """Prints each part of the CSV file, including nested structures."""
    for index, row in df.iterrows():
        print(f"Row {index}:")
        for column, value in row.items():
            print(f"  {column}: {value}")
        print("\n")

def plot_results(df, output_graph_path):
    """Plots the true values, LNN predictions, and biased predictions."""
    plt.figure(figsize=(14, 7))

    # Check and set date as index
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df.set_index('date', inplace=True)
    elif 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.set_index('Date', inplace=True)
    else:
        logger.error("Date column not found in the DataFrame.")
        return
    
    df = df[df.index.notna()]
    if df.empty:
        logger.error("DataFrame is empty after filtering invalid dates.")
        return
    
    # Ensure required columns exist
    required_columns = ['actual']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return
    
    date_min = df.index.min()
    date_max = df.index.max()
    logger.info(f"Date range in dataset: {date_min} to {date_max}")
    plt.xlim(date_min, date_max)

    # Plot true values
    plt.plot(df.index, df['actual'], label="True Values", color="blue")

    # Plot predictions if they exist
    if 'predicted' in df.columns:
        plt.plot(df.index, df['predicted'], label="Predictions", color="orange")
    elif 'LSTM_Predictions' in df.columns:
        plt.plot(df.index, df['LSTM_Predictions'], label="LSTM Predictions", color="orange")

    # Plot biased predictions if they exist
    if 'BiasedPrediction' in df.columns:
        plt.plot(df.index, df['BiasedPrediction'], label="Biased Prediction", color="purple", linestyle='-.')

    # Labels and title
    plt.xlabel("Time")
    plt.ylabel("Values")
    plt.title("True Values vs. LNN/LSTM Predictions and Biased Predictions")
    plt.legend()

    # Save plot and show
    plt.savefig(output_graph_path)
    plt.show()

    # Log message to confirm the graph was saved
    logger.info(f"Prediction results plot saved to {output_graph_path}")

def plot_learning_curve(df, ticker, output_dir):
    """Plots the learning curve data."""
    plt.figure(figsize=(10, 6))
    
    if 'Train Size' in df.columns and 'Train Loss' in df.columns and 'Validation Loss' in df.columns:
        plt.plot(df['Train Size'], df['Train Loss'], label='Training Loss')
        plt.plot(df['Train Size'], df['Validation Loss'], label='Validation Loss')
    else:
        logger.error("Required columns for learning curve not found in the DataFrame.")
        return
    
    plt.xlabel('Train Size')
    plt.ylabel('Loss')
    plt.title(f'Learning Curve for {ticker}')
    plt.legend()
    plt.grid(True)
    
    learning_curve_plot_path = output_dir / f"{ticker}_learning_curve.png"
    plt.savefig(learning_curve_plot_path)
    plt.show()
    
    logger.info(f"Learning curve plot saved to {learning_curve_plot_path}")

@app.command()
def main(
    ticker_symbol: str = typer.Argument(..., help="Ticker symbol to analyze."),
    mode: str = typer.Option("train_test", help="Mode for prediction: 'train_test' or 'production'"),
    output_path: Path = typer.Option(None, help="Path to save the output graph."),
    num_lines: int = typer.Option(5, help="Number of lines to display from the CSV files."),
):
    ticker_symbol = ticker_symbol.strip().upper()

    # Validate mode argument
    if mode not in ['train_test', 'production']:
        logger.error("Invalid mode. Please use 'train_test' or 'production'.")
        return

    # Set default output path if not provided, using the ticker symbol
    if output_path is None:
        output_path = PROCESSED_DATA_DIR / f"{ticker_symbol}_final_plot_{mode}.png"

    # Set file paths based on mode (match the actual file naming convention)
    prediction_path = PROCESSED_DATA_DIR / f"{ticker_symbol}_{mode}_predictions.csv"
    
    # Only load learning curve data in train_test mode
    if mode == "train_test":
        learning_curve_path = PROCESSED_DATA_DIR / f"{ticker_symbol}_learning_curve_{mode}.csv"
    else:
        learning_curve_path = None

    logger.info(f"Attempting to load prediction data from: {prediction_path}")
    if learning_curve_path:
        logger.info(f"Attempting to load learning curve data from: {learning_curve_path}")

    # Load the prediction data
    logger.info("Loading prediction data from file...")
    try:
        prediction_df = pd.read_csv(prediction_path)
        logger.success("Prediction data loaded successfully.")
    except FileNotFoundError:
        logger.error(f"Prediction file not found: {prediction_path}. Please ensure the file exists in the PROCESSED_DATA_DIR.")
        return
    except Exception as e:
        logger.error(f"An error occurred while loading prediction data: {e}")
        return

    # Load the learning curve data only in train_test mode
    if mode == "train_test":
        logger.info("Loading learning curve data from file...")
        try:
            learning_curve_df = pd.read_csv(learning_curve_path)
            logger.success("Learning curve data loaded successfully.")
        except FileNotFoundError:
            logger.error(f"Learning curve file not found: {learning_curve_path}. Please ensure the file exists in the PROCESSED_DATA_DIR.")
            return
        except Exception as e:
            logger.error(f"An error occurred while loading learning curve data: {e}")
            return

    # Ensure required columns are present in prediction data
    required_columns = ['actual', 'predicted']
    missing_columns = [col for col in required_columns if col not in prediction_df.columns]

    # Allow for either LNN or LSTM predictions
    if 'predicted' not in prediction_df.columns and 'LSTM_Predictions' not in prediction_df.columns:
        logger.error(f"Neither 'predicted' nor 'LSTM_Predictions' found in the prediction dataset.")
        return
    
    # Display the first few lines of the prediction CSV file
    logger.info(f"Displaying the first {num_lines} lines of the prediction CSV file:")
    print_csv_parts(prediction_df.head(num_lines))

    if mode == "train_test":
        # Display the first few lines of the learning curve CSV file in train_test mode
        logger.info(f"Displaying the first {num_lines} lines of the learning curve CSV file:")
        print(learning_curve_df.head(num_lines))

    # Plot the prediction results
    logger.info("Plotting the prediction results...")
    plot_results(prediction_df, output_path)

    # Plot the learning curve if in train_test mode
    if mode == "train_test":
        logger.info("Plotting the learning curve...")
        plot_learning_curve(learning_curve_df, ticker_symbol, PROCESSED_DATA_DIR)

    logger.success(f"Plots saved to {output_path.parent}.")

if __name__ == "__main__":
    app()

import os
import pandas as pd
import numpy as np
import typer
from pathlib import Path
from loguru import logger
from sentimental_cap_predictor.sentimental_cap_predictor.sentimental_cap_predictor.config import PROCESSED_DATA_DIR

app = typer.Typer()

# Function to calculate daily returns
def calculate_daily_returns(price_df):
    price_df['daily_returns'] = price_df['close'].pct_change()
    return price_df['daily_returns']

# Function to calculate covariance matrix
def calculate_covariance_matrix(all_returns):
    returns_df = pd.DataFrame(all_returns)
    return returns_df.cov()

@app.command()
def analyze_tickers(
    ticker_list: str = typer.Option(
        "NVDA,TSLA,PLTR,SQ,ROKU,ZG,CRWD,U,PINS,SHOP,SNOW,DDOG,FVRR,HIMS,AI,TWLO,ENPH,MRNA,BYND,CRSP,ILMN,RIVN,QS,LMND,TTD,SE,DOCU,LYFT,OTLY", 
        help="Comma-separated list of ticker symbols to analyze."
    ),
    mode: str = typer.Option("production", help="Mode for prediction: 'train_test' or 'production'"),
    output_path: Path = typer.Option(None, help="Path to save the output covariance matrix."),
):
    # Parse ticker list
    tickers = [ticker.strip().upper() for ticker in ticker_list.split(",")]

    # Set default output path if not provided
    if output_path is None:
        output_path = PROCESSED_DATA_DIR / "cov_matrix.csv"

    all_returns = {}

    # Loop through each ticker, process data, and calculate daily returns
    for ticker in tickers:
        data_path = PROCESSED_DATA_DIR / f"{ticker}_{mode}.csv"

        logger.info(f"Attempting to load data for {ticker} from: {data_path}")
        
        try:
            # Load stock data
            price_df = pd.read_csv(data_path)
            logger.success(f"Data for {ticker} loaded successfully.")
        except FileNotFoundError:
            logger.error(f"Data file for {ticker} not found: {data_path}. Skipping ticker.")
            continue
        except Exception as e:
            logger.error(f"An error occurred while loading data for {ticker}: {e}")
            continue

        # Ensure the 'close' column exists in the data
        if 'close' not in price_df.columns:
            logger.error(f"Required column 'close' not found in {ticker}'s data. Skipping.")
            continue

        # Calculate daily returns
        logger.info(f"Calculating daily returns for {ticker}...")
        daily_returns = calculate_daily_returns(price_df)
        all_returns[ticker] = daily_returns

    # Combine all daily returns into a DataFrame
    combined_returns_df = pd.DataFrame(all_returns)

    # Drop rows with NaN values (these appear after calculating percentage change)
    combined_returns_df.dropna(inplace=True)

    # Calculate the covariance matrix for all tickers
    logger.info("Calculating covariance matrix...")
    cov_matrix = calculate_covariance_matrix(combined_returns_df)
    logger.info("Covariance matrix calculated.")

    # Save covariance matrix to CSV
    logger.info(f"Saving covariance matrix to {output_path}")
    cov_matrix.to_csv(output_path)
    logger.success(f"Covariance matrix saved to {output_path}")

if __name__ == "__main__":
    app()

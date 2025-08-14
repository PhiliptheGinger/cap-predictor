from datetime import datetime as dt, timedelta
from dateutil.relativedelta import relativedelta
from pathlib import Path
from typing import Optional
import pandas as pd
import typer
import yfinance as yf
from loguru import logger
from typing_extensions import Annotated
from colorama import Fore, Style
import re

from sentimental_cap_predictor.sentimental_cap_predictor.sentimental_cap_predictor.config import RAW_DATA_DIR

app = typer.Typer()

def parse_flexible_date_range(date_range: str) -> tuple[str, str]:
    """Parse a flexible date range input like '1Y', '2M', '3D'."""
    match = re.match(r"(\d+)([YMD])", date_range.upper())
    if not match:
        raise ValueError("Invalid date range format. Please use formats like '1Y', '2M', '3D'.")

    amount, unit = match.groups()
    amount = int(amount)

    end_date = dt.today()

    if unit == 'Y':
        start_date = end_date - relativedelta(years=amount)
    elif unit == 'M':
        start_date = end_date - relativedelta(months=amount)
    elif unit == 'D':
        start_date = end_date - timedelta(days=amount)
    else:
        raise ValueError("Invalid date unit. Use 'Y' for years, 'M' for months, 'D' for days.")
    
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

def download_ticker_from_yfinance(
        ticker: str,
        start_date: str | dt | int,
        end_date: str | dt | int,
) -> pd.DataFrame:
    """Download data from yfinance and format appropriately."""
    # Convert start_date and end_date to strings if they are not already
    start_date = str(start_date)
    end_date = str(end_date)

    print(f"{Fore.YELLOW}Starting download for {ticker} from {start_date} to {end_date}.{Style.RESET_ALL}")
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        df.index = pd.to_datetime(df.index)
        print(f"{Fore.GREEN}Data for {ticker} downloaded successfully.{Style.RESET_ALL}")
    except Exception as err:
        logger.exception("Error getting data from yfinance:", err)
        raise err

    if len(df) < 3:
        msg = "Need at least 3 business days of data to infer frequency."
        logger.error("Not enough data to infer frequency", ValueError(msg))
        raise ValueError(msg)
    
    try:
        inferred_freq = pd.infer_freq(df.index)
        print(f"Inferred frequency: {inferred_freq}")
    except Exception as err:
        logger.error(
            "Error inferring frequency, defaulting to business days",
            err,
        )
        inferred_freq = "B"
        print("Defaulting to business day frequency.")
    
    df = df.asfreq(inferred_freq)
    print(f"Data adjusted to inferred frequency for {ticker}.")

    return df

def handle_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing data by forward filling and backfilling, then dropping any remaining NaNs."""
    print(f"{Fore.YELLOW}Handling missing data.{Style.RESET_ALL}")
    
    # Forward fill and backfill missing values
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Drop rows that still contain NaNs
    df = df.dropna()
    
    print(f"{Fore.GREEN}Missing data handled.{Style.RESET_ALL}")
    return df

@app.command()
def main(
    output_path: Annotated[Optional[Path], typer.Argument()] = None,
) -> None:
    print(f"{Fore.YELLOW}Please enter the ticker symbol:{Style.RESET_ALL}", end=' ')
    ticker = input(f"{Fore.GREEN}").strip().upper()

    print(f"{Fore.YELLOW}Please enter a date range (e.g., '1Y', '2M', '3D') or a start date (YYYY-MM-DD):{Style.RESET_ALL}", end=' ')
    date_input = input(f"{Fore.GREEN}").strip()

    if re.match(r"^\d+[YMD]$", date_input.upper()):
        start_date, end_date = parse_flexible_date_range(date_input)
    else:
        start_date = date_input
        print(f"{Fore.YELLOW}Please enter an end date (YYYY-MM-DD):{Style.RESET_ALL}", end=' ')
        end_date = input(f"{Fore.GREEN}").strip()

    print(f"{Style.RESET_ALL}Starting main function.")
    if not output_path:
        logger.trace(f"Inferring output file for {ticker}")
        output_path = RAW_DATA_DIR / f"{ticker}.feather"
        print(f"Inferred output path: {output_path}")

    logger.trace(f"Downloading {ticker} data from {start_date} thru {end_date}.")
    print(f"Downloading {ticker} data...")
    df = download_ticker_from_yfinance(ticker, start_date, end_date)
    print("Download complete.")
    
    # Handle missing data
    df = handle_missing_data(df)
    
    # Add the date index as a column before saving to feather
    df.reset_index(inplace=True)
    
    logger.trace(f"Writing feather to {output_path}.")
    print(f"Writing data to {output_path}.")
    df.to_feather(output_path)
    print(f"Data saved to {output_path}.")

if __name__ == "__main__":
    print(f"{Fore.YELLOW}Script execution started.{Style.RESET_ALL}")
    app()
    print(f"{Fore.GREEN}Script execution finished.{Style.RESET_ALL}")

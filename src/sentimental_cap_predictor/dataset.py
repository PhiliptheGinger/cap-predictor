from datetime import datetime as dt, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import typer
import yfinance as yf
from colorama import Fore, Style, init
from dotenv import load_dotenv
from loguru import logger
from typing_extensions import Annotated

from .config import RAW_DATA_DIR, ENABLE_TICKER_LOGS
from .data.news import extract_article_content, query_gdelt_for_news
from .data_bundle import DataBundle
from .preprocessing import merge_data

load_dotenv()

# Initialize colorama
init(autoreset=True)

app = typer.Typer()


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names by converting to lowercase.

    Spaces are replaced with underscores.
    """
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    return df


def check_for_nan(df: pd.DataFrame, context: str = "") -> None:
    """Check for NaN values and log them with context."""
    nan_values = df.isna().sum().sum()
    if nan_values > 0:
        logger.warning(
            f"{Fore.RED}Found {nan_values} NaNs after {context}."
            f"{Style.RESET_ALL}"
        )
        logger.warning(df[df.isna().any(axis=1)].to_string())
    else:
        if ENABLE_TICKER_LOGS:
            logger.info(
                f"{Fore.GREEN}No NaN values found after {context}."
                f"{Style.RESET_ALL}"
            )




def download_ticker_from_yfinance(ticker: str, period: str) -> pd.DataFrame:
    """Download price data for a ticker from yfinance for a given period."""
    if ENABLE_TICKER_LOGS:
        logger.info(
            f"{Fore.YELLOW}Starting price data download for {ticker} "
            f"for period: {period}."
            f"{Style.RESET_ALL}"
        )
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)

        if df.empty:
            logger.warning(
                f"{Fore.RED}No price data found for {ticker}.{Style.RESET_ALL}"
            )
            return pd.DataFrame()

        df.reset_index(inplace=True)
        check_for_nan(df, "downloading ticker data")
        return df
    except (requests.exceptions.RequestException, ValueError) as err:
        logger.error(
            f"{Fore.RED}Error getting price data from yfinance: {err}"
            f"{Style.RESET_ALL}"
        )
        return pd.DataFrame()
    except Exception as err:
        logger.exception(
            f"Unexpected error getting price data from yfinance: {err}"
        )
        raise


def handle_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing data without adding NaNs to complete columns."""
    if ENABLE_TICKER_LOGS:
        logger.info(f"{Fore.YELLOW}Handling missing data.{Style.RESET_ALL}")

    initial_missing = df.isnull().sum().sum()
    cols_with_missing = df.columns[df.isnull().any()]

    if "date" in df.columns:
        df["date"] = df["date"].fillna(method="ffill").fillna(method="bfill")

    df[cols_with_missing] = (
        df[cols_with_missing].fillna(method="ffill").fillna(method="bfill")
    )

    final_missing = df.isnull().sum().sum()
    if ENABLE_TICKER_LOGS:
        logger.info(
            f"{Fore.GREEN}Handled missing data. "
            f"Initial NaNs: {initial_missing}, "
            f"remaining after processing: {final_missing}."
            f"{Style.RESET_ALL}"
        )

    check_for_nan(df, "handling missing data")
    return df


def load_data_bundle(ticker: str) -> DataBundle:
    """Load price and news data for ``ticker`` into a :class:`DataBundle`.

    The function expects :mod:`dataset.main` to have stored point-in-time price
    and news data in ``RAW_DATA_DIR``. The ``date`` column becomes a
    ``DatetimeIndex`` and aligned before returning the validated bundle.
    Rows with timestamps in the future are dropped to avoid look-ahead bias.
    """

    price_path = RAW_DATA_DIR / f"{ticker}.feather"
    news_path = RAW_DATA_DIR / f"{ticker}_news.feather"

    price_df = pd.read_feather(price_path)
    if "date" in price_df.columns:
        price_df["date"] = pd.to_datetime(price_df["date"])
        price_df = price_df[price_df["date"] <= pd.Timestamp.utcnow()]
        price_df.set_index("date", inplace=True)

    if news_path.exists():
        news_df = pd.read_feather(news_path)
        if "date" in news_df.columns:
            news_df["date"] = pd.to_datetime(news_df["date"], errors="coerce")
            news_df = news_df[news_df["date"] <= pd.Timestamp.utcnow()]
            news_df.set_index("date", inplace=True)
        else:
            news_df.index = pd.to_datetime(news_df.index)
    else:
        news_df = pd.DataFrame(index=price_df.index)

    # Align news data to price index to avoid accidental look-ahead
    news_df = news_df.reindex(price_df.index).fillna(method="ffill")

    bundle = DataBundle(
        prices=price_df, sentiment=news_df, metadata={"ticker": ticker}
    )
    return bundle.validate()


@app.command()
def main(
    ticker: str,
    period: Annotated[
        str,
        typer.Option(
            "--period",
            "-p",
            help=(
                "Period for data collection (e.g., '1Y', '1M', '1W', "
                "or 'max'). Can also be provided positionally for backward "
                "compatibility."
            ),
        ),
    ] = "max",
    period_arg: Annotated[
        str | None,
        typer.Argument(
            metavar="PERIOD",
            help="Period for data collection (legacy positional form)",
            show_default=False,
        ),
    ] = None,
    output_path: Optional[Path] = None,
    news_output_path: Optional[Path] = None,
    use_headless: bool = False,
) -> None:
    if period_arg is not None:
        period = period_arg
    if ENABLE_TICKER_LOGS:
        logger.info(
            f"{Fore.YELLOW}Starting data collection for ticker: {ticker}."
            f"{Style.RESET_ALL}"
        )

    if not output_path:
        output_path = RAW_DATA_DIR / f"{ticker}.feather"

    if not news_output_path:
        news_output_path = RAW_DATA_DIR / f"{ticker}_news.feather"

    # Load existing price data if file exists
    if output_path.exists():
        existing_price_df = pd.read_feather(output_path)
        if ENABLE_TICKER_LOGS:
            logger.info(
                f"{Fore.GREEN}Loaded existing price data from {output_path}."
                f"{Style.RESET_ALL}"
            )
    else:
        if ENABLE_TICKER_LOGS:
            logger.info(
                f"{Fore.YELLOW}No existing price data found for {ticker}. "
                f"Creating a new file.{Style.RESET_ALL}"
            )
        existing_price_df = pd.DataFrame()

    # Download new price data
    new_price_df = download_ticker_from_yfinance(ticker, period)
    if new_price_df.empty:
        logger.error(
            f"{Fore.RED}No data available for {ticker}."
            f"{Style.RESET_ALL}"
        )
        return

    # Normalize and handle missing data
    new_price_df = normalize_column_names(new_price_df)
    new_price_df = handle_missing_data(new_price_df)

    # Merge new data with existing data
    merged_price_df = merge_data(
        existing_price_df, new_price_df, merge_on="date"
    )

    # Save merged data back to feather file
    merged_price_df.to_feather(output_path)
    if ENABLE_TICKER_LOGS:
        logger.info(f"Price data saved to {output_path}.")

    # Load existing news data if file exists
    if news_output_path.exists():
        existing_news_df = pd.read_feather(news_output_path)
        if ENABLE_TICKER_LOGS:
            logger.info(
                f"{Fore.GREEN}Loaded news data from {news_output_path}."
                f"{Style.RESET_ALL}"
            )
    else:
        if ENABLE_TICKER_LOGS:
            logger.info(
                f"{Fore.YELLOW}No existing news data found for {ticker}. "
                f"Creating a new file.{Style.RESET_ALL}"
            )
        existing_news_df = pd.DataFrame()

    # Query for new news data
    today = dt.today()
    two_weeks_ago = today - timedelta(days=14)
    start_date = two_weeks_ago.strftime("%Y%m%d000000")
    end_date = today.strftime("%Y%m%d000000")

    new_news_df = query_gdelt_for_news(ticker, start_date, end_date)

    if not new_news_df.empty:
        new_news_df = normalize_column_names(new_news_df)

        # Extract content for each article
        contents = []
        successful_extractions = 0
        for url in new_news_df["url"]:
            content = extract_article_content(url, use_headless=use_headless)
            if content:
                successful_extractions += 1
            contents.append(content)

        new_news_df["content"] = contents

        if ENABLE_TICKER_LOGS:
            logger.info(
                f"{Fore.GREEN}Successfully extracted content from "
                f"{successful_extractions} out of {len(new_news_df)} articles."
                f"{Style.RESET_ALL}"
            )

        # Merge new news data with existing data
        merged_news_df = merge_data(
            existing_news_df, new_news_df, merge_on="url"
        )

        # Save merged news data back to feather file
        merged_news_df.to_feather(news_output_path)
        if ENABLE_TICKER_LOGS:
            logger.info(f"News data saved to {news_output_path}.")
    else:
        logger.warning(
            f"{Fore.RED}No news data available for the given date range."
            f"{Style.RESET_ALL}"
        )


if __name__ == "__main__":
    app()

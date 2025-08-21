import os
from dotenv import load_dotenv
import requests
import pandas as pd
from newspaper import Article, Config
from datetime import datetime as dt, timedelta
from pathlib import Path
from typing import Optional
import typer
import yfinance as yf
from loguru import logger
from typing_extensions import Annotated
from colorama import Fore, Style, init
from .preprocessing import merge_data
from .data_bundle import DataBundle

load_dotenv()

from .config import RAW_DATA_DIR

# Initialize colorama
init(autoreset=True)

app = typer.Typer()

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names by converting to lowercase and removing spaces."""
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    return df

def check_for_nan(df: pd.DataFrame, context: str = '') -> None:
    """Check for NaN values in the dataframe and print them along with context."""
    nan_values = df.isna().sum().sum()
    if nan_values > 0:
        print(f"{Fore.RED}Warning: Found {nan_values} NaN values after {context}.{Style.RESET_ALL}")
        print(df[df.isna().any(axis=1)])
    else:
        print(f"{Fore.GREEN}No NaN values found after {context}.{Style.RESET_ALL}")

def query_gdelt_for_news(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Query GDELT API to get news articles based on a ticker and date range."""
    url = os.getenv('GDELT_API_URL', 'https://api.gdeltproject.org/api/v2/doc/doc')  # Default value provided
    
    params = {
        "query": ticker,
        "mode": "artlist",
        "startdatetime": start_date,
        "enddatetime": end_date,
        "maxrecords": 100,
        "format": "json"
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        articles = data.get('articles', [])
        return pd.DataFrame(articles)
    except requests.exceptions.RequestException as err:
        print(f"{Fore.RED}Error querying GDELT API: {err}{Style.RESET_ALL}")
        return pd.DataFrame()

def extract_article_content(url: str, use_headless: bool = False) -> Optional[str]:
    """Extract the main content from a news article URL using newspaper3k.

    The ``use_headless`` flag switches the user agent to mimic a headless
    browser, which can help when sites block default requests.  A small request
    timeout is also applied to avoid hanging when articles are unreachable.
    """
    try:
        config = Config()
        config.browser_user_agent = (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/120.0.0 Safari/537.36"
            if use_headless
            else "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        )
        config.request_timeout = 10
        article = Article(url, config=config, keep_article_html=True)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"{Fore.RED}Error extracting content from {url}: {e}{Style.RESET_ALL}")
        return None

def download_ticker_from_yfinance(ticker: str, period: str) -> pd.DataFrame:
    """Download the price data for a specific period for a ticker from yfinance."""
    print(f"{Fore.YELLOW}Starting price data download for {ticker} for period: {period}.{Style.RESET_ALL}")
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)

        if df.empty:
            print(f"{Fore.RED}No price data found for {ticker}.{Style.RESET_ALL}")
            return pd.DataFrame()

        df.reset_index(inplace=True)
        check_for_nan(df, "downloading ticker data")
        return df
    except Exception as err:
        logger.exception("Error getting price data from yfinance:", err)
        return pd.DataFrame()

def handle_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing data without introducing NaNs in already complete columns."""
    print(f"{Fore.YELLOW}Handling missing data.{Style.RESET_ALL}")
    
    initial_missing = df.isnull().sum().sum()
    cols_with_missing = df.columns[df.isnull().any()]
    
    if 'date' in df.columns:
        df['date'] = df['date'].fillna(method='ffill').fillna(method='bfill')

    df[cols_with_missing] = df[cols_with_missing].fillna(method='ffill').fillna(method='bfill')

    final_missing = df.isnull().sum().sum()
    print(f"{Fore.GREEN}Missing data handled. Initial NaNs: {initial_missing}, Remaining NaNs after processing: {final_missing}.{Style.RESET_ALL}")
    
    check_for_nan(df, "handling missing data")
    return df


def load_data_bundle(ticker: str) -> DataBundle:
    """Load price and news data for ``ticker`` into a :class:`DataBundle`.

    The function expects that :mod:`dataset.main` has previously stored point-in-
    time price and news data in ``RAW_DATA_DIR``.  It converts the ``date``
    column to a ``DatetimeIndex`` and ensures both frames are aligned before
    returning the validated bundle.  Any rows with timestamps in the future are
    dropped to guard against look-ahead bias.
    """

    price_path = RAW_DATA_DIR / f"{ticker}.feather"
    news_path = RAW_DATA_DIR / f"{ticker}_news.feather"

    price_df = pd.read_feather(price_path)
    if 'date' in price_df.columns:
        price_df['date'] = pd.to_datetime(price_df['date'])
        price_df = price_df[price_df['date'] <= pd.Timestamp.utcnow()]
        price_df.set_index('date', inplace=True)

    if news_path.exists():
        news_df = pd.read_feather(news_path)
        if 'date' in news_df.columns:
            news_df['date'] = pd.to_datetime(news_df['date'], errors='coerce')
            news_df = news_df[news_df['date'] <= pd.Timestamp.utcnow()]
            news_df.set_index('date', inplace=True)
        else:
            news_df.index = pd.to_datetime(news_df.index)
    else:
        news_df = pd.DataFrame(index=price_df.index)

    # Align news data to price index to avoid accidental look-ahead
    news_df = news_df.reindex(price_df.index).fillna(method='ffill')

    bundle = DataBundle(prices=price_df, sentiment=news_df, metadata={'ticker': ticker})
    return bundle.validate()

@app.command()
def main(
    ticker: str,
    period: Annotated[str, typer.Argument(help="Period for data collection (e.g., '1Y', '1M', '1W', or 'max')")] = "max",
    output_path: Optional[Path] = None,
    news_output_path: Optional[Path] = None,
    use_headless: bool = False
) -> None:
    print(f"{Fore.YELLOW}Starting data collection for ticker: {ticker}.{Style.RESET_ALL}")

    if not output_path:
        output_path = RAW_DATA_DIR / f"{ticker}.feather"

    if not news_output_path:
        news_output_path = RAW_DATA_DIR / f"{ticker}_news.feather"

    # Load existing price data if file exists
    if output_path.exists():
        existing_price_df = pd.read_feather(output_path)
        print(f"{Fore.GREEN}Loaded existing price data from {output_path}.{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}No existing price data found for {ticker}. Creating a new file.{Style.RESET_ALL}")
        existing_price_df = pd.DataFrame()

    # Download new price data
    new_price_df = download_ticker_from_yfinance(ticker, period)
    if new_price_df.empty:
        print(f"{Fore.RED}No data available for {ticker}.{Style.RESET_ALL}")
        return

    # Normalize and handle missing data
    new_price_df = normalize_column_names(new_price_df)
    new_price_df = handle_missing_data(new_price_df)

    # Merge new data with existing data
    merged_price_df = merge_data(existing_price_df, new_price_df, merge_on='date')
    
    # Save merged data back to feather file
    merged_price_df.to_feather(output_path)
    print(f"Price data saved to {output_path}.")

    # Load existing news data if file exists
    if news_output_path.exists():
        existing_news_df = pd.read_feather(news_output_path)
        print(f"{Fore.GREEN}Loaded existing news data from {news_output_path}.{Style.RESET_ALL}")
    else:
        print(f"{Fore.YELLOW}No existing news data found for {ticker}. Creating a new file.{Style.RESET_ALL}")
        existing_news_df = pd.DataFrame()

    # Query for new news data
    today = dt.today()
    two_weeks_ago = today - timedelta(days=14)
    start_date = two_weeks_ago.strftime('%Y%m%d000000')
    end_date = today.strftime('%Y%m%d000000')

    new_news_df = query_gdelt_for_news(ticker, start_date, end_date)

    if not new_news_df.empty:
        new_news_df = normalize_column_names(new_news_df)
        
        # Extract content for each article
        contents = []
        successful_extractions = 0
        for url in new_news_df['url']:
            content = extract_article_content(url, use_headless=use_headless)
            if content:
                successful_extractions += 1
            contents.append(content)
        
        new_news_df['content'] = contents
        
        print(f"{Fore.GREEN}Successfully extracted content from {successful_extractions} out of {len(new_news_df)} articles.{Style.RESET_ALL}")
        
        # Merge new news data with existing data
        merged_news_df = merge_data(existing_news_df, new_news_df, merge_on='url')

        # Save merged news data back to feather file
        merged_news_df.to_feather(news_output_path)
        print(f"News data saved to {news_output_path}.")
    else:
        print(f"{Fore.RED}No news data available for the given date range.{Style.RESET_ALL}")

if __name__ == "__main__":
    app()

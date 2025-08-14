import os
import pandas as pd
import numpy as np
import alpaca_trade_api as tradeapi
import typer
from pathlib import Path
from loguru import logger
import time
import sys
from dotenv import load_dotenv
from sentimental_cap_predictor.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from sentimental_cap_predictor.features import generate_predictions as feature_main
from sentimental_cap_predictor.dataset import main as dataset_main
from scipy.optimize import minimize

app = typer.Typer()

# Load environment variables from the .env file
load_dotenv()

# Load Ticker List from Environment
TICKER_LIST = (
    os.getenv("TICKER_LIST_TECH", "") + "," +
    os.getenv("TICKER_LIST_ENERGY", "") + "," +
    os.getenv("TICKER_LIST_HEALTH", "") + "," +
    os.getenv("TICKER_LIST_AUTO", "") + "," +
    os.getenv("TICKER_LIST_FINANCIAL", "") + "," +
    os.getenv("TICKER_LIST_MISC", "")
).split(",")

# Step 1: Configure Alpaca API for paper mode
API_KEY = os.getenv('APCA_API_KEY_ID', 'your_paper_api_key')
API_SECRET = os.getenv('APCA_API_SECRET_KEY', 'your_paper_api_secret')
BASE_URL = "https://paper-api.alpaca.markets"

# Initialize Alpaca API connection
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# Set up logging configuration
def setup_logging():
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_rotation = os.getenv("LOG_ROTATION", "10 MB")

    logger.remove()
    logger.add(sys.stdout, level=log_level)
    logger.add("sentimental_analysis.log", level=log_level, rotation=log_rotation)
    logger.info(f"Logging initialized with level: {log_level}")

# Calculate daily returns
def calculate_daily_returns(price_df):
    price_df['daily_returns'] = price_df['close'].pct_change()
    return price_df['daily_returns']

# Calculate expected return as the mean of daily returns
def calculate_expected_return(daily_returns):
    return daily_returns.mean()

# Calculate covariance matrix for all tickers
def calculate_covariance_matrix(all_returns):
    returns_df = pd.DataFrame(all_returns)
    return returns_df.cov()

# Calculate risk (standard deviation of daily returns)
def calculate_volatility(daily_returns):
    return daily_returns.std()

# Analyze tickers and return the results
def analyze_tickers(ticker: str = None, mode: str = 'production', period: str = '1Y', prediction_days: int = 14):
    setup_logging()
    start_time = time.time()
    logger.info(f"Starting stock analysis in {mode} mode.")

    if not PROCESSED_DATA_DIR.exists():
        logger.info(f"Processed directory does not exist. Creating: {PROCESSED_DATA_DIR}")
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    tickers_data = {}
    all_returns = {}

    def process_ticker(ticker):
        try:
            dataset_main(ticker=ticker, period=period)
            df_path = Path(RAW_DATA_DIR) / f"{ticker}.feather"
            
            # Check if the file exists
            if not df_path.exists():
                logger.error(f"Data file for ticker {ticker} not found at {df_path}. Skipping...")
                return
            
            price_df = pd.read_feather(df_path)
            
            # Calculate daily returns
            daily_returns = calculate_daily_returns(price_df)
            
            # Calculate expected return and risk (volatility)
            expected_return = calculate_expected_return(daily_returns)
            risk = calculate_volatility(daily_returns)
            
            # Store daily returns for covariance calculation later
            all_returns[ticker] = daily_returns
            
            # Check if news data exists
            news_path = Path(RAW_DATA_DIR) / f"{ticker}_news.feather"
            if not news_path.exists():
                logger.error(f"News data file for ticker {ticker} not found. Skipping...")
                return
            
            news_df = pd.read_feather(news_path)
            result = feature_main(price_df, news_df, ticker, mode, prediction_days)
            
            # Append expected return, risk, and store in tickers_data
            result['expected_returns'] = expected_return
            result['risks'] = risk
            tickers_data[ticker] = result
            return result

        except Exception as e:
            logger.error(f"Failed to process ticker {ticker}: {e}")


    if ticker:
        try:
            logger.info(f"Processing specific ticker: {ticker}")
            process_ticker(ticker)
        except Exception as e:
            logger.error(f"Failed to process ticker {ticker}: {e}")
    else:
        for ticker in TICKER_LIST:
            try:
                logger.info(f"Processing ticker: {ticker}")
                process_ticker(ticker)
            except Exception as e:
                logger.error(f"Failed to process ticker {ticker}: {e}")

    # Calculate covariance matrix after all tickers are processed
    cov_matrix = calculate_covariance_matrix(all_returns)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Stock analysis completed in {elapsed_time:.2f} seconds.")

    return tickers_data, cov_matrix

# Utility 1: Mean-Variance Optimization (MPT)
def mean_variance_optimization(expected_returns, cov_matrix, total_funds):
    num_assets = len(expected_returns)
    if cov_matrix.shape[0] != num_assets or cov_matrix.shape[1] != num_assets:
        raise ValueError(f"Covariance matrix shape {cov_matrix.shape} does not match number of assets {num_assets}")
    
    args = (expected_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    result = minimize(portfolio_volatility, num_assets * [1. / num_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    
    allocations = {asset: alloc * total_funds for asset, alloc in zip(expected_returns.keys(), result['x'])}
    return allocations


def portfolio_volatility(weights, expected_returns, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

# Utility 2: Proportional Allocation
def proportional_allocation(expected_returns, total_funds):
    total_expected_return = sum(expected_returns.values())
    allocations = {asset: (return_pct / total_expected_return) * total_funds for asset, return_pct in expected_returns.items()}
    return allocations

# Utility 3: Sharpe Ratio Allocation
def sharpe_ratio_allocation(expected_returns, risks, risk_free_rate, total_funds):
    sharpe_ratios = {asset: (expected_returns[asset] - risk_free_rate) / risks[asset] for asset in expected_returns}
    total_sharpe = sum(sharpe_ratios.values())
    allocations = {asset: (sharpe / total_sharpe) * total_funds for asset, sharpe in sharpe_ratios.items()}
    return allocations

# Calculate allocations based on the strategy
def calculate_allocations(strategy, expected_returns, total_funds, risks=None, risk_free_rate=None, cov_matrix=None):
    if strategy == 'mpt':
        return mean_variance_optimization(expected_returns, cov_matrix, total_funds)
    elif strategy == 'proportional':
        return proportional_allocation(expected_returns, total_funds)
    elif strategy == 'sharpe':
        return sharpe_ratio_allocation(expected_returns, risks, risk_free_rate, total_funds)
    else:
        raise ValueError("Invalid allocation strategy provided.")

# Function to find peaks and valleys in predictions
def find_peaks_and_valleys(predictions, window=3):
    peaks, valleys = [], []
    for i in range(window, len(predictions) - window):
        if predictions[i] > max(predictions[i - window:i]) and predictions[i] > max(predictions[i + 1:i + window + 1]):
            peaks.append(i)
        elif predictions[i] < min(predictions[i - window:i]) and predictions[i] < min(predictions[i + 1:i + window + 1]):
            valleys.append(i)
    return peaks, valleys

# Example function for calculating stop-loss threshold
def calculate_stop_loss(predictions):
    return 0.95

# Function to make trading decisions based on analysis results
def make_trading_decisions(ticker: str, data: pd.DataFrame, allocations: dict):
    peaks, valleys = find_peaks_and_valleys(data['LSTM_Predictions'])
    stop_loss_threshold = calculate_stop_loss(data['LSTM_Predictions'])

    for i in range(len(data)):
        current_price = data['LSTM_Predictions'][i]
        date = data['date'][i]
        position = None

        try:
            position = api.get_position(ticker)
        except tradeapi.rest.APIError:
            pass  # No open position

        quantity = allocations[ticker] / current_price

        if i in valleys and not position:
            print(f"Buying {quantity:.2f} shares of {ticker} at ${current_price:.2f} on {date}")
            api.submit_order(symbol=ticker, qty=int(quantity), side='buy', type='market', time_in_force='gtc')

        elif i in peaks and position:
            print(f"Selling {quantity:.2f} shares of {ticker} at ${current_price:.2f} on {date}")
            api.submit_order(symbol=ticker, qty=int(quantity), side='sell', type='market', time_in_force='gtc')

        elif position and current_price < float(position.avg_entry_price) * stop_loss_threshold:
            print(f"Stop-loss triggered: Selling {quantity:.2f} shares of {ticker} at ${current_price:.2f} on {date}")
            api.submit_order(symbol=ticker, qty=int(quantity), side='sell', type='market', time_in_force='gtc')

# Close final position
def close_final_position(ticker: str, data: pd.DataFrame, quantity: int):
    try:
        position = api.get_position(ticker)
        if position:
            current_price = data['LSTM_Predictions'].iloc[-1]
            print(f"Closing final position: Selling {quantity} shares of {ticker} at ${current_price:.2f} on {data['date'].iloc[-1]}")
            api.submit_order(symbol=ticker, qty=quantity, side='sell', type='market', time_in_force='gtc')
    except tradeapi.rest.APIError:
        pass

# Main execution of trading decisions
if __name__ == "__main__":
    tickers_data, cov_matrix = analyze_tickers(mode='production', period='1Y')

    for ticker, data in tickers_data.items():
        expected_returns = data['expected_returns']
        risks = data['risks']
        risk_free_rate = data.get('risk_free_rate', 0.02)
        total_funds = 10000
        strategy = 'mpt'

        allocations = calculate_allocations(strategy, expected_returns, total_funds, risks=risks, risk_free_rate=risk_free_rate, cov_matrix=cov_matrix)
        make_trading_decisions(ticker, data, allocations)
        close_final_position(ticker, data, int(allocations[ticker] / data['LSTM_Predictions'].iloc[-1]))

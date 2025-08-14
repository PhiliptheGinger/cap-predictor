import os
import pandas as pd
import numpy as np
import alpaca_trade_api as tradeapi
from analyze_batch import analyze_tickers  # Ensure to import this from the batch analysis script
from scipy.optimize import minimize

# Step 1: Configure Alpaca API for paper mode
API_KEY = os.getenv('APCA_API_KEY_ID', 'your_paper_api_key')
API_SECRET = os.getenv('APCA_API_SECRET_KEY', 'your_paper_api_secret')
BASE_URL = "https://paper-api.alpaca.markets"

# Initialize Alpaca API connection
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# Step 2: Analyze tickers using the analyze batch script (tickers provided internally by the batch script)
def run_analysis(mode: str, period: str = '1Y', prediction_days: int = 14):
    # This function should analyze all tickers handled by analyze_tickers and return data for each ticker
    return analyze_tickers(mode=mode, period=period, prediction_days=prediction_days)

# Utility 1: Mean-Variance Optimization (MPT)
def mean_variance_optimization(expected_returns, cov_matrix, total_funds):
    num_assets = len(expected_returns)
    args = (expected_returns, cov_matrix)
    
    # Constraints: sum of weights should equal 1 (fully allocated funds)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    # Bounds for the weights (between 0 and 1 for each asset)
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    # Perform the optimization to minimize portfolio volatility
    result = minimize(portfolio_volatility, num_assets * [1. / num_assets], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    
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

# Step 3: Function to calculate allocations based on the selected strategy
def calculate_allocations(strategy, expected_returns, total_funds, risks=None, risk_free_rate=None, cov_matrix=None):
    if strategy == 'mpt':
        return mean_variance_optimization(expected_returns, cov_matrix, total_funds)
    elif strategy == 'proportional':
        return proportional_allocation(expected_returns, total_funds)
    elif strategy == 'sharpe':
        return sharpe_ratio_allocation(expected_returns, risks, risk_free_rate, total_funds)
    else:
        raise ValueError("Invalid allocation strategy provided.")

# Step 4: Function to find peaks and valleys
def find_peaks_and_valleys(predictions, window=3):
    peaks = []
    valleys = []
    for i in range(window, len(predictions) - window):
        if predictions[i] > max(predictions[i - window:i]) and predictions[i] > max(predictions[i + 1:i + window + 1]):
            peaks.append(i)
        elif predictions[i] < min(predictions[i - window:i]) and predictions[i] < min(predictions[i + 1:i + window + 1]):
            valleys.append(i)
    return peaks, valleys

# Example function for calculating stop-loss threshold
def calculate_stop_loss(predictions):
    return 0.95

# Example function to make trading decisions based on predictions and allocations
def make_trading_decisions(ticker: str, data: pd.DataFrame, allocations: dict):
    peaks, valleys = find_peaks_and_valleys(data['LSTM_Predictions'])

    stop_loss_threshold = calculate_stop_loss(data['LSTM_Predictions'])

    for i in range(len(data)):
        current_price = data['LSTM_Predictions'][i]
        date = data['date'][i]

        # Check if there is an open position
        position = None
        try:
            position = api.get_position(ticker)
        except tradeapi.rest.APIError:
            pass  # No open position

        # Allocate based on the strategy
        quantity = allocations[ticker] / current_price  # Calculate quantity based on the allocated funds

        if i in valleys and not position:
            # Buy at valley
            print(f"Buying {quantity:.2f} shares of {ticker} at ${current_price:.2f} on {date}")
            api.submit_order(
                symbol=ticker,
                qty=int(quantity),
                side='buy',
                type='market',
                time_in_force='gtc'
            )

        elif i in peaks and position:
            # Sell at peak
            print(f"Selling {quantity:.2f} shares of {ticker} at ${current_price:.2f} on {date}")
            api.submit_order(
                symbol=ticker,
                qty=int(quantity),
                side='sell',
                type='market',
                time_in_force='gtc'
            )

        elif position and current_price < float(position.avg_entry_price) * stop_loss_threshold:
            # Execute stop-loss
            print(f"Stop-loss triggered: Selling {quantity:.2f} shares of {ticker} at ${current_price:.2f} on {date}")
            api.submit_order(
                symbol=ticker,
                qty=int(quantity),
                side='sell',
                type='market',
                time_in_force='gtc'
            )

# Final check for open positions at the end of the script
def close_final_position(ticker: str, data: pd.DataFrame, quantity: int):
    try:
        position = api.get_position(ticker)
        if position:
            current_price = data['LSTM_Predictions'].iloc[-1]
            print(f"Closing final position: Selling {quantity} shares of {ticker} at ${current_price:.2f} on {data['date'].iloc[-1]}")
            api.submit_order(
                symbol=ticker,
                qty=quantity,
                side='sell',
                type='market',
                time_in_force='gtc'
            )
    except tradeapi.rest.APIError:
        pass  # No open position

# Main execution of trading decisions
if __name__ == "__main__":
    # Step 4: Run analysis for the tickers dynamically, no need to pass the ticker explicitly
    tickers_data = run_analysis(mode='production', period='1Y')
    
    # Process multiple tickers from the batch analysis (assuming the analyze_tickers function returns this)
    for ticker, data in tickers_data.items():
        # Step 5: Calculate allocations for each ticker
        expected_returns = data['expected_returns']  # From analysis
        risks = data['risks']  # From analysis
        cov_matrix = data['cov_matrix']  # From analysis
        risk_free_rate = data.get('risk_free_rate', 0.02)  # Set a default risk-free rate
        
        total_funds = 10000  # You could customize this per ticker or use one value for all
        strategy = 'mpt'  # Choose strategy ('mpt', 'proportional', or 'sharpe')

        allocations = calculate_allocations(strategy, expected_returns, total_funds, risks=risks, risk_free_rate=risk_free_rate, cov_matrix=cov_matrix)

        # Step 6: Make trading decisions for each ticker based on predictions and allocations
        make_trading_decisions(ticker, data, allocations)

        # Step 7: Ensure all positions are closed at the end
        close_final_position(ticker, data, int(allocations[ticker] / data['LSTM_Predictions'].iloc[-1]))

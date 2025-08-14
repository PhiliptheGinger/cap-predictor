import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def simulate_gbm(S0, mu, sigma, T, steps):
    dt = T / steps
    S = np.zeros(steps)
    S[0] = S0

    for t in range(1, steps):
        dW = np.random.normal(0, np.sqrt(dt))
        S[t] = S[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)
    
    # Check if the output is all zeros
    if np.all(S == 0):
        logging.warning("All values in the GBM simulation are zeros. Check parameters or initial conditions.")
    
    return {"GBM": S}

if __name__ == "__main__":
    # Updated initial conditions
    S0 = 100  # Initial stock price
    mu = 0.05  # Expected return
    sigma = 0.2  # Volatility
    T = 1  # Time horizon
    steps = 74  # Number of steps (matching your test data)

    logging.info(f"Starting GBM simulation with inputs: S0={S0}, mu={mu}, sigma={sigma}, T={T}, steps={steps}")

    gbm_results = simulate_gbm(S0, mu, sigma, T, steps)
    logging.info(f"GBM simulation results: {gbm_results['GBM'][:5]}... (first 5 values)")

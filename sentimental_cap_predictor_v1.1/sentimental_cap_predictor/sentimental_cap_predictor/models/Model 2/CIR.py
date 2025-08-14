import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def simulate_cir(X0, kappa, theta, sigma, T, steps):
    dt = T / steps
    X = np.zeros(steps)
    X[0] = X0

    for t in range(1, steps):
        dW = np.random.normal(0, np.sqrt(dt))
        X[t] = X[t-1] + kappa * (theta - X[t-1]) * dt + sigma * np.sqrt(max(X[t-1], 0)) * dW

        if X[t] < 0:
            X[t] = 0  # CIR process should never become negative
    
    # Check if the output is all zeros
    if np.all(X == 0):
        logging.warning("All values in the CIR simulation are zeros. Check parameters or initial conditions.")
    
    return {"CIR": X}

if __name__ == "__main__":
    # Updated initial conditions
    X0 = 0.05  # Initial short rate
    kappa = 0.5  # Speed of mean reversion
    theta = 0.02  # Long-term mean rate
    sigma = 0.1  # Volatility
    T = 1  # Time horizon
    steps = 74  # Number of steps (matching your test data)

    logging.info(f"Starting CIR simulation with inputs: X0={X0}, kappa={kappa}, theta={theta}, sigma={sigma}, T={T}, steps={steps}")

    cir_results = simulate_cir(X0, kappa, theta, sigma, T, steps)
    logging.info(f"CIR simulation results: {cir_results['CIR'][:5]}... (first 5 values)")

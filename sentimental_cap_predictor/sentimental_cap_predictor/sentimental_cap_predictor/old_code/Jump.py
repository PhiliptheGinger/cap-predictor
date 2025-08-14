import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def simulate_jump_diffusion(S0, mu, sigma, lambda_, muJ, sigmaJ, T, steps):
    dt = T / steps
    S = np.zeros(steps)
    S[0] = S0

    for t in range(1, steps):
        dW = np.random.normal(0, np.sqrt(dt))
        dN = np.random.poisson(lambda_ * dt)
        S[t] = S[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW + dN * (muJ + sigmaJ * np.random.normal(0, 1)))
    
    # Check if the output is all zeros
    if np.all(S == 0):
        logging.warning("All values in the Jump Diffusion simulation are zeros. Check parameters or initial conditions.")
    
    return {"JumpDiffusion": S}

if __name__ == "__main__":
    # Updated initial conditions
    S0 = 100  # Initial stock price
    mu = 0.05  # Expected return
    sigma = 0.2  # Volatility
    lambda_ = 0.1  # Jump intensity
    muJ = -0.1  # Jump mean
    sigmaJ = 0.3  # Jump volatility
    T = 1  # Time horizon
    steps = 74  # Number of steps (matching your test data)

    logging.info(f"Starting Jump Diffusion simulation with inputs: S0={S0}, mu={mu}, sigma={sigma}, lambda_={lambda_}, muJ={muJ}, sigmaJ={sigmaJ}, T={T}, steps={steps}")

    jump_results = simulate_jump_diffusion(S0, mu, sigma, lambda_, muJ, sigmaJ, T, steps)
    logging.info(f"Jump Diffusion simulation results: {jump_results['JumpDiffusion'][:5]}... (first 5 values)")

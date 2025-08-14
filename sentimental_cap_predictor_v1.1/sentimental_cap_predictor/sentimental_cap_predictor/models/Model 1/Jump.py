import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def simulate_jump_diffusion(S0, mu, sigma, T, lambda_j, mu_j, sigma_j, steps):
    dt = T / steps
    S = np.zeros(steps)
    S[0] = S0

    for t in range(1, steps):
        dW = np.random.normal(0, np.sqrt(dt))
        J = np.random.normal(mu_j, sigma_j) if np.random.uniform(0, 1) < lambda_j * dt else 0
        S[t] = S[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW + J)
    
    # Check if the output is all zeros
    if np.all(S == 0):
        logging.warning("All values in the Jump Diffusion simulation are zeros. Check parameters or initial conditions.")

    return {"JumpDiffusion": S}

if __name__ == "__main__":
    # Updated initial conditions
    S0 = 100  # Initial stock price
    mu = 0.05  # Expected return
    sigma = 0.2  # Volatility
    lambda_j = 0.1  # Jump intensity
    mu_j = 0.01  # Mean of jump size
    sigma_j = 0.05  # Volatility of jump size
    T = 1  # Time horizon
    steps = 74  # Number of steps (matching your test data)

    logging.info(f"Starting Jump Diffusion simulation with inputs: S0={S0}, mu={mu}, sigma={sigma}, T={T}, lambda_j={lambda_j}, mu_j={mu_j}, sigma_j={sigma_j}, steps={steps}")

    jump_diffusion_results = simulate_jump_diffusion(S0, mu, sigma, T, lambda_j, mu_j, sigma_j, steps)
    logging.info(f"Jump Diffusion simulation results: {jump_diffusion_results['JumpDiffusion'][:5]}... (first 5 values)")

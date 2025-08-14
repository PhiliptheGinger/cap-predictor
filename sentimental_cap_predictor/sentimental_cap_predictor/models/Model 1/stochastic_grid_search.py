import numpy as np
from itertools import product

# Example functions for the CIR, GBM, and Jump Diffusion models (placeholders)

def simulate_CIR(kappa, theta, sigma, X0, T=1, N=100):
    dt = T / N
    X = np.zeros(N)
    X[0] = X0
    for i in range(1, N):
        dX = kappa * (theta - X[i-1]) * dt + sigma * np.sqrt(X[i-1]) * np.random.normal()
        X[i] = X[i-1] + dX
    return X

def simulate_GBM(mu, sigma, S0, T=1, N=100):
    dt = T / N
    S = np.zeros(N)
    S[0] = S0
    for i in range(1, N):
        dS = mu * S[i-1] * dt + sigma * S[i-1] * np.random.normal()
        S[i] = S[i-1] + dS
    return S

def simulate_JumpDiffusion(lambda_, muJ, sigmaJ, S0, T=1, N=100):
    dt = T / N
    S = np.zeros(N)
    S[0] = S0
    for i in range(1, N):
        jump = np.random.poisson(lambda_) * (muJ + sigmaJ * np.random.normal())
        dS = S[i-1] * dt + jump
        S[i] = S[i-1] + dS
    return S

# Define parameter ranges for grid search
kappa_range = [0.1, 0.5, 1.0]
theta_range = [0.01, 0.05, 0.1]
sigma_range = [0.01, 0.05, 0.1]
X0_range = [0.01, 0.1, 0.5]

mu_range = [0.01, 0.05, 0.1]
S0_range = [1, 10, 50]

lambda_range = [0.1, 0.5, 1.0]
muJ_range = [0.01, 0.05, 0.1]
sigmaJ_range = [0.01, 0.05, 0.1]

# Create all combinations of parameters
cir_params = list(product(kappa_range, theta_range, sigma_range, X0_range))
gbm_params = list(product(mu_range, sigma_range, S0_range))
jump_params = list(product(lambda_range, muJ_range, sigmaJ_range, S0_range))

# Grid search loop
best_cir_params = None
best_gbm_params = None
best_jump_params = None

min_error_cir = float('inf')
min_error_gbm = float('inf')
min_error_jump = float('inf')

# Evaluate CIR model
for kappa, theta, sigma, X0 in cir_params:
    X = simulate_CIR(kappa, theta, sigma, X0)
    error = np.mean(np.square(X))  # Example error metric
    if error < min_error_cir:
        min_error_cir = error
        best_cir_params = (kappa, theta, sigma, X0)

# Evaluate GBM model
for mu, sigma, S0 in gbm_params:
    S = simulate_GBM(mu, sigma, S0)
    error = np.mean(np.square(S))  # Example error metric
    if error < min_error_gbm:
        min_error_gbm = error
        best_gbm_params = (mu, sigma, S0)

# Evaluate Jump Diffusion model
for lambda_, muJ, sigmaJ, S0 in jump_params:
    S = simulate_JumpDiffusion(lambda_, muJ, sigmaJ, S0)
    error = np.mean(np.square(S))  # Example error metric
    if error < min_error_jump:
        min_error_jump = error
        best_jump_params = (lambda_, muJ, sigmaJ, S0)

# Output the best parameters
print("Best CIR parameters:", best_cir_params)
print("Best GBM parameters:", best_gbm_params)
print("Best Jump Diffusion parameters:", best_jump_params)

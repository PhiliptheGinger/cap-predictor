import pytest
import numpy as np

# Add the directory to sys.path if it's not already there
import sys
sys.path.append(r"D:\Programming Projects\CAP\sentimental_cap_predictor\sentimental_cap_predictor\sentimental_cap_predictor\modeling")

# Now import the functions from the Python file
from stochastic_grid_search import simulate_CIR, simulate_GBM, simulate_JumpDiffusion, calculate_diff


# Test parameters
kappa = 0.5
theta = 0.05
sigma = 0.1
X0 = 0.1
mu = 0.05
sigma_gbm = 0.1
S0 = 10
lambda_ = 0.5
muJ = 0.05
sigmaJ = 0.1
N = 100  # Number of steps
T = 1  # Time horizon

# CIR model tests
def test_simulate_CIR_output_length():
    X = simulate_CIR(kappa, theta, sigma, X0, T, N)
    assert len(X) == N, "CIR output length does not match N"

def test_simulate_CIR_non_negative():
    X = simulate_CIR(kappa, theta, sigma, X0, T, N)
    assert np.all(X >= 0), "CIR model has negative values"

def test_simulate_CIR_diff_length():
    X = simulate_CIR(kappa, theta, sigma, X0, T, N)
    X_diff = calculate_diff(X)
    assert len(X_diff) == N - 1, "Differencing output length for CIR does not match N-1"

# GBM model tests
def test_simulate_GBM_output_length():
    S = simulate_GBM(mu, sigma_gbm, S0, T, N)
    assert len(S) == N, "GBM output length does not match N"

def test_simulate_GBM_diff_length():
    S = simulate_GBM(mu, sigma_gbm, S0, T, N)
    S_diff = calculate_diff(S)
    assert len(S_diff) == N - 1, "Differencing output length for GBM does not match N-1"

# Jump Diffusion model tests
def test_simulate_JumpDiffusion_output_length():
    S = simulate_JumpDiffusion(lambda_, muJ, sigmaJ, S0, T, N)
    assert len(S) == N, "Jump Diffusion output length does not match N"

def test_simulate_JumpDiffusion_diff_length():
    S = simulate_JumpDiffusion(lambda_, muJ, sigmaJ, S0, T, N)
    S_diff = calculate_diff(S)
    assert len(S_diff) == N - 1, "Differencing output length for Jump Diffusion does not match N-1"

# Test edge cases and exceptions
def test_simulate_CIR_zero_sigma():
    X = simulate_CIR(kappa, theta, 0, X0, T, N)
    assert len(X) == N, "CIR output length with sigma=0 does not match N"

def test_simulate_GBM_zero_sigma():
    S = simulate_GBM(mu, 0, S0, T, N)
    assert len(S) == N, "GBM output length with sigma=0 does not match N"

def test_simulate_JumpDiffusion_zero_lambda():
    S = simulate_JumpDiffusion(0, muJ, sigmaJ, S0, T, N)
    assert len(S) == N, "Jump Diffusion output length with lambda=0 does not match N"

# Test if the models handle large N without issues
@pytest.mark.parametrize("large_N", [1000, 10000, 100000])
def test_large_N_simulations(large_N):
    X = simulate_CIR(kappa, theta, sigma, X0, T, large_N)
    S_gbm = simulate_GBM(mu, sigma_gbm, S0, T, large_N)
    S_jump = simulate_JumpDiffusion(lambda_, muJ, sigmaJ, S0, T, large_N)
    
    assert len(X) == large_N, "CIR output length does not match large N"
    assert len(S_gbm) == large_N, "GBM output length does not match large N"
    assert len(S_jump) == large_N, "Jump Diffusion output length does not match large N"


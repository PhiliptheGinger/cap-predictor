import pytest
import numpy as np
from unittest.mock import patch

# Import the functions from your module
from sentimental_cap_predictor.modeling.markov_chain_generator import transition_matrix, simulate_markov_chain, smooth_markov_chain

def test_transition_matrix():
    # Example data with 3 states
    data = np.array([0, 1, 2, 0, 1, 2, 1, 0])
    n_states = 3
    
    # Call the function
    T = transition_matrix(data, n_states)
    
    # Assert that T is a 3x3 matrix
    assert T.shape == (n_states, n_states)
    
    # Assert that the rows of T sum to 1 (since it's a probability matrix)
    np.testing.assert_almost_equal(T.sum(axis=1), np.ones(n_states))
    
    # Test a specific expected value, for example:
    assert T[0, 1] > 0  # Transition from state 0 to state 1

def test_simulate_markov_chain():
    P = np.array([[0.1, 0.6, 0.3],
                  [0.4, 0.4, 0.2],
                  [0.3, 0.3, 0.4]])
    π = np.array([0.2, 0.5, 0.3])
    
    with patch('numpy.random.choice') as mock_choice:
        mock_choice.side_effect = [1, 2, 0, 1]  # Predefined choices
        
        # Simulate the Markov chain
        states = simulate_markov_chain(P, π, steps=4)
        
        # Assert the correct sequence of states
        assert states == [1, 2, 0, 1]

def test_smooth_markov_chain():
    states = [0, 2, 1, 3]
    smoothing_factor = 0.1
    
    # Call the smoothing function
    smoothed_states = smooth_markov_chain(states, smoothing_factor)
    
    # Calculate expected smoothed values
    expected_smoothed_states = [
        (1 - smoothing_factor) * 2 + smoothing_factor * 0,
        (1 - smoothing_factor) * 1 + smoothing_factor * 2,
        (1 - smoothing_factor) * 3 + smoothing_factor * 1
    ]
    
    # Assert the smoothed states match the expected values
    np.testing.assert_almost_equal(smoothed_states, expected_smoothed_states)

# Additional test cases can be added for edge cases, such as empty data, invalid input, etc.

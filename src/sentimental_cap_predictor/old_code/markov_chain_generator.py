import numpy as np

def transition_matrix(data, n_states):
    """Creates the transition matrix for a Markov chain."""
    T = np.zeros((n_states, n_states))
    
    for (i, j) in zip(data, data[1:]):
        T[i, j] += 1
    
    # Ensure no row has a zero sum; handle cases where there are no transitions from a state
    row_sums = T.sum(axis=1, keepdims=True)
    T = np.where(row_sums == 0, 1/n_states, T)  # Replace rows with no transitions with equal probabilities
    
    # Normalize to create probability matrix
    T = T / T.sum(axis=1, keepdims=True)
    
    return T

def simulate_markov_chain(P, π, steps=100):
    """
    Simulates a Markov chain using the transition matrix (P) and initial distribution (π).
    
    Args:
    - P (numpy array): Transition matrix.
    - π (numpy array): Initial distribution vector.
    - steps (int): Number of steps to simulate.
    
    Returns:
    - states (list): List of simulated states.
    """
    states = [np.random.choice(len(π), p=π)]
    
    for _ in range(steps - 1):
        current_state = states[-1]
        
        if current_state >= len(P):
            print(f"Warning: Current state {current_state} is out of bounds for the transition matrix.")
            next_state = np.random.choice(len(π), p=π)  # Fallback to initial distribution
        else:
            next_state = np.random.choice(len(π), p=P[current_state])
        
        states.append(next_state)
        # Debug log to track transitions
        print(f"Transitioned from {states[-2]} to {next_state}")
    
    return states

def smooth_markov_chain(states, smoothing_factor=0.1):
    """
    Smooths a Markov chain using a smoothing factor.
    
    Args:
    - states (list): List of states from the Markov chain.
    - smoothing_factor (float): Smoothing factor for averaging.
    
    Returns:
    - smoothed_states (list): Smoothed states.
    """
    if len(states) < 2:
        return states  # No smoothing needed if there are fewer than 2 states
    
    smoothed_states = []
    
    for i in range(1, len(states)):
        smoothed_value = (1 - smoothing_factor) * states[i] + smoothing_factor * states[i - 1]
        smoothed_states.append(smoothed_value)
        # Debug log to track smoothing process
        print(f"Smoothed value from {states[i-1]} and {states[i]} to {smoothed_value}")
    
    return smoothed_states

# Example usage:
# Generate some random data for demonstration
data = np.random.randint(0, 3, 100)
n_states = len(np.unique(data))

# Generate transition matrix
T = transition_matrix(data, n_states)

# Initial distribution (π) based on the empirical distribution
unique, counts = np.unique(data, return_counts=True)
π = counts / counts.sum()

# Simulate Markov chain
simulated_states = simulate_markov_chain(T, π, steps=100)

# Smooth the simulated Markov chain
smoothed_states = smooth_markov_chain(simulated_states, smoothing_factor=0.1)

print("Simulated States:", simulated_states)
print("Smoothed States:", smoothed_states)

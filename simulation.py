import numpy as np

def run_market_simulation(weights, returns, cov_matrix, steps=10):
    """Step 1: Simulate the raw market path without user influence."""
    p_return = np.dot(weights, returns)
    p_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    time_steps = np.linspace(0, 1, steps)
    drift = (p_return - 0.5 * p_volatility**2) * time_steps
    diffusion = p_volatility * np.random.normal(0, np.sqrt(time_steps))
    
    return np.exp(drift + diffusion).tolist()

def get_advisory_adjustment(current_weights, panic_level):
    """Step 2: Generate the advisory recommendation based on the panic reaction."""
    # Logic: Higher panic = higher shift to safe assets (index 0)
    adjustment = panic_level * 0.2 
    new_weights = np.array(current_weights).copy()
    
    # Shift from risky (index 3) to safe (index 0)
    new_weights[3] = max(0, new_weights[3] - adjustment)
    new_weights[0] = new_weights[0] + adjustment
    
    return (new_weights / np.sum(new_weights)).tolist()

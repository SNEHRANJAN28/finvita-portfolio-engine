import numpy as np

def simulate_portfolio_volatility(current_weights, returns, cov_matrix, duration_minutes=10, steps=10):
    p_return = np.dot(current_weights, returns)
    p_volatility = np.sqrt(np.dot(current_weights.T, np.dot(cov_matrix, current_weights)))
    
    time_steps = np.linspace(0, duration_minutes/60, steps)
    drift = (p_return - 0.5 * p_volatility**2) * time_steps
    diffusion = p_volatility * np.random.normal(0, np.sqrt(time_steps))
    
    simulation_path = np.exp(drift + diffusion)
    return simulation_path.tolist()

def adjust_portfolio_on_user_input(current_weights, user_panic_level):
    adjustment = user_panic_level * 0.1 
    new_weights = current_weights.copy()
    
    # Simple logic: shift weight away from high-risk to safe assets
    new_weights[-1] = max(0, new_weights[-1] - adjustment)
    new_weights[0] = new_weights[0] + adjustment
    
    return new_weights / np.sum(new_weights)
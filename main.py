import os
import numpy as np
from fastapi import FastAPI
from models import PortfolioRequest, SimulationRequest
from simulation import simulate_portfolio_volatility, adjust_portfolio_on_user_input

app = FastAPI(title="FinVita Indian Portfolio Engine API")

# Ensure API Key is loaded via environment variables
client_api_key = os.getenv("GEMINI_API_KEY")

base_covariance = np.array([
    [0.025, 0.010, 0.001, 0.004], [0.010, 0.040, 0.000, 0.006],
    [0.001, 0.000, 0.002, 0.001], [0.004, 0.006, 0.001, 0.015]
])

@app.get("/")
def home():
    return {"status": "healthy", "engine": "FinVita Portfolio Engine"}

@app.post("/optimize")
def optimize_portfolio(request: PortfolioRequest):
    # Your optimization logic here
    return {"status": "success"}

@app.post("/simulate-volatility")
def simulate(request: SimulationRequest):
    # Now this will show all 3 fields in the Swagger UI
    simulation = simulate_portfolio_volatility(
        np.array(request.weights), 
        np.array(request.returns), 
        base_covariance
    )
    new_weights = adjust_portfolio_on_user_input(
        np.array(request.weights), 
        request.panic_level
    )
    
    return {
        "simulation_path": simulation,
        "suggested_adjustment": new_weights.tolist(),
        "status": "simulation_complete"
    }

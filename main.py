from fastapi import FastAPI
from models import SimulationRequest
from simulation import run_market_simulation, get_advisory_adjustment
import numpy as np

app = FastAPI(title="FinVita Sequential Engine")

base_covariance = np.array([
    [0.025, 0.010, 0.001, 0.004], [0.010, 0.040, 0.000, 0.006],
    [0.001, 0.000, 0.002, 0.001], [0.004, 0.006, 0.001, 0.015]
])

@app.post("/simulate-and-advise")
def simulate_and_advise(request: SimulationRequest):
    # 1. Run raw simulation
    market_path = run_market_simulation(
        np.array(request.weights), 
        np.array(request.returns), 
        base_covariance
    )
    
    # 2. Generate advisory adjustment based on the panic level provided
    advisory_recommendation = get_advisory_adjustment(
        request.weights, 
        request.panic_level
    )
    
    return {
        "market_simulation": market_path,
        "user_panic_input": request.panic_level,
        "advisory_adjustment": advisory_recommendation,
        "message": "Market simulated. Advisory adjustment generated based on reaction."
    }

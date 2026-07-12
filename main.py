import os
import json
import numpy as np
import scipy.optimize as sco
from fastapi import FastAPI
from google import genai
from google.genai import types

# Import from modular files
from models import PortfolioRequest
from simulation import simulate_portfolio_volatility, adjust_portfolio_on_user_input

app = FastAPI(title="FinVita Indian Portfolio Engine API")

if "GEMINI_API_KEY" not in os.environ:
    raise ValueError("Please set your GEMINI_API_KEY environment variable.")

client = genai.Client()
MODEL_ID = 'gemini-2.5-flash'

base_covariance = np.array([
    [0.025, 0.010, 0.001, 0.004], [0.010, 0.040, 0.000, 0.006],
    [0.001, 0.000, 0.002, 0.001], [0.004, 0.006, 0.001, 0.015]
])

@app.get("/")
def home():
    return {"status": "healthy", "engine": "FinVita Portfolio Engine"}

@app.post("/optimize")
def optimize_portfolio(request: PortfolioRequest):
    # (Insert your existing optimization logic here)
    return {"status": "success", "message": "Optimization complete"}

@app.post("/simulate-volatility")
def simulate(weights: list[float], returns: list[float], panic_level: float = 0.0):
    simulation = simulate_portfolio_volatility(np.array(weights), np.array(returns), base_covariance)
    new_weights = adjust_portfolio_on_user_input(np.array(weights), panic_level)
    
    return {
        "simulation_path": simulation,
        "suggested_adjustment": new_weights.tolist(),
        "status": "simulation_complete"
    }

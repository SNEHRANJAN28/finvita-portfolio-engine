import os
import json
import numpy as np
import scipy.optimize as sco
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# Initialize the Gemini Client safely from Environment Variables locally
if "GEMINI_API_KEY" not in os.environ:
    raise ValueError("Please set your GEMINI_API_KEY environment variable locally (e.g., export GEMINI_API_KEY='your_key').")

client = genai.Client()
MODEL_ID = 'gemini-2.5-flash'

# STEP 1: USER PROFILE INGESTION & VALDIATION SCHEMA
# Defines the multi-parameter criteria structure including investment bounds, 
# categorical risk rules, savings-to-income weights, and asset management constraints.
class PortfolioRequest(BaseModel):
    amount: float = Field(50000.0, description="Total investment capital in INR")
    horizon: int = Field(5, description="Investment horizon timeline in years")
    risk_tolerance: str = Field("medium", description="Categorical risk tolerance level (low, medium, high)")
    savings_to_income_ratio: float = Field(0.30, description="Proportion of monthly income allocated to savings/investments")
    primary_investment_goal: str = Field("wealth_accumulation", description="Primary objective")
    preferred_rebalancing_frequency: str = Field("annually", description="Desired portfolio adjustment intervals")

def run_local_portfolio_engine(request: PortfolioRequest):
    print("\n" + "="*60)
    print("🚀 INITIALIZING LOCAL FINVITA ENGINE (WITH LIVE GEMINI SEARCH)")
    print("="*60)
    
    risk_tolerance = request.risk_tolerance.strip().lower()
    amount = request.amount
    horizon = request.horizon
    
    # Extract structural constraints mapped to the user schema
    savings_ratio = request.savings_to_income_ratio
    investment_goal = request.primary_investment_goal.strip().lower()
    rebalance_freq = request.preferred_rebalancing_frequency.strip().lower()
    
    risk_map = {"low": 0.25, "medium": 0.55, "high": 0.85}
    user_profile = {
        "amount": amount,
        "investment_horizon": horizon,
        "risk_tolerance": risk_tolerance,
        "risk_score": risk_map.get(risk_tolerance, 0.55)
    }

    using_fallback = False
    discovered_assets = []
    adjusted_returns = []
    market_data_json = {"items": []}

    # STEP 2: DUAL-PHASE DATA DISCOVERY (LIVE AI GROUNDING & SCHEMA PARSING)
    # Phase A: Google Search Grounding to scrape active NSE/BSE asset metrics, live macroeconomic 
    # factors, and sector trends, strictly isolating domestic instruments from global markets.
    try:
        print("🔍 Step 2A: Querying Google Search via Gemini Grounding for Indian Market Data...")
        market_search_prompt = f"""
        You are an expert quantitative market researcher specializing exclusively in the Indian financial markets (NSE, BSE, and SEBI-regulated instruments).
        The user profile is: Risk Level: {user_profile['risk_tolerance'].upper()}, Investment Horizon: {user_profile['investment_horizon']} years.

        Look up current real-time market data across prominent Indian investment paths via Google Search. 
        Identify exactly 4 distinct best asset opportunities in India right now suited for this profile.

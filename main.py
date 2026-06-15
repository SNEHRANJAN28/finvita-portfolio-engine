import os
import json
import numpy as np
import scipy.optimize as sco
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI(title="FinVita Indian Portfolio Engine API")

# Initialize the Gemini Client safely from Environment Variables
if "GEMINI_API_KEY" not in os.environ:
    raise ValueError("Please set your GEMINI_API_KEY environment variable.")

client = genai.Client()
MODEL_ID = 'gemini-2.5-flash'

class PortfolioRequest(BaseModel):
    amount: float = 50000.0
    horizon: int = 5
    risk_tolerance: str = "medium"

@app.get("/")
def home():
    return {"status": "healthy", "engine": "FinVita Indian Portfolio Optimization Engine"}

@app.post("/optimize")
def optimize_portfolio(request: PortfolioRequest):
    risk_tolerance = request.risk_tolerance.strip().lower()
    amount = request.amount
    horizon = request.horizon
    
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

    try:
        market_search_prompt = f"""
        You are an expert quantitative market researcher specializing exclusively in the Indian financial markets (NSE, BSE, and SEBI-regulated instruments).
        The user profile is: Risk Level: {user_profile['risk_tolerance'].upper()}, Investment Horizon: {user_profile['investment_horizon']} years.

        Look up current real-time market data across prominent Indian investment paths via Google Search. 
        Identify exactly 4 distinct best asset opportunities in India right now suited for this profile.
        
        CRITICAL: DO NOT return US equities, global ETFs, or international stock tickers. No S&P 500, no NVIDIA, no Vanguard, no US Bonds.
        ONLY select valid Indian market choices such as:
        - Nifty 50 Index ETF / Mutual Funds
        - Nifty Next 50 ETF
        - Indian Corporate Bond Funds (e.g., HDFC, SBI, ICICI Debt)
        - Domestic Gold ETFs listed on the National Stock Exchange of India (NSE)

        For each asset, provide the name, an estimated annual return decimal, and a news sentiment score from -0.5 to 0.5.
        """
        
        search_response = client.models.generate_content(
            model=MODEL_ID,
            contents=market_search_prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.1
            ),
        )

        native_json_schema = types.Schema(
            type=types.Type.OBJECT,
            properties={
                "items": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "asset_name": types.Schema(type=types.Type.STRING),
                            "forecasted_return": types.Schema(type=types.Type.NUMBER),
                            "sentiment_score": types.Schema(type=types.Type.NUMBER),
                        },
                        required=["asset_name", "forecasted_return", "sentiment_score"],
                    ),
                )
            },
            required=["items"],
        )

        parsing_prompt = f"""
        Extract the 4 Indian assets from this text and format into JSON. 
        Verify that all asset names are local Indian financial instruments. Convert percentages to decimals.
        {search_response.text}
        """
        
        structured_response = client.models.generate_content(
            model=MODEL_ID,
            contents=parsing_prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=native_json_schema, temperature=0.0),
        )

        market_data_json = json.loads(structured_response.text)
        for item in market_data_json["items"]:
            discovered_assets.append(item["asset_name"])
            adjusted_returns.append(item["forecasted_return"] + (0.5 * item["sentiment_score"]))
            
    except Exception:
        using_fallback = True
        if user_profile["risk_tolerance"] == "low":
            fallback_assets = [
                {"asset_name": "SBI Sovereign G-Sec Fund", "forecasted_return": 0.072, "sentiment_score": 0.1},
                {"asset_name": "HDFC Corporate Bond Fund", "forecasted_return": 0.078, "sentiment_score": 0.0},
                {"asset_name": "Nippon India Gold ETF", "forecasted_return": 0.085, "sentiment_score": 0.2},
                {"asset_name": "UTI Nifty 50 Index Fund", "forecasted_return": 0.120, "sentiment_score": 0.1}
            ]
        elif user_profile["risk_tolerance"] == "high":
            fallback_assets = [
                {"asset_name": "Nippon India Nifty Next 50 ETF", "forecasted_return": 0.145, "sentiment_score": 0.2},
                {"asset_name": "Motilal Oswal Midcap Fund", "forecasted_return": 0.170, "sentiment_score": 0.3},
                {"asset_name": "Tata Digital India Fund", "forecasted_return": 0.185, "sentiment_score": 0.4},
                {"asset_name": "ICICI Prudential Multi-Asset Fund", "forecasted_return": 0.130, "sentiment_score": 0.1}
            ]
        else:
            fallback_assets = [
                {"asset_name": "UTI Nifty 50 Index ETF", "forecasted_return": 0.130, "sentiment_score": 0.2},
                {"asset_name": "ICICI Prudential Balanced Advantage Fund", "forecasted_return": 0.115, "sentiment_score": 0.1},
                {"asset_name": "Aditya Birla Sun Life Corporate Bond Fund", "forecasted_return": 0.082, "sentiment_score": 0.0},
                {"asset_name": "HDFC Gold ETF", "forecasted_return": 0.090, "sentiment_score": 0.2}
            ]
        market_data_json = {"items": fallback_assets}
        for item in fallback_assets:
            discovered_assets.append(item["asset_name"])
            adjusted_returns.append(item["forecasted_return"] + (0.5 * item["sentiment_score"]))

    base_covariance = np.array([
        [0.025, 0.010, 0.001, 0.004], [0.010, 0.040, 0.000, 0.006],
        [0.001, 0.000, 0.002, 0.001], [0.004, 0.006, 0.001, 0.015]
    ])
    num_assets = len(discovered_assets)
    covariance_matrix = base_covariance[:num_assets, :num_assets]
    R_f = 0.065
    R = np.array(adjusted_returns)

    def portfolio_performance(weights, returns, cov_matrix):
        p_return = np.dot(weights, returns)
        p_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return p_return, p_volatility

    def neg_sharpe_ratio(weights, returns, cov_matrix, risk_free_rate):
        p_return, p_volatility = portfolio_performance(weights, returns, cov_matrix)
        if p_volatility == 0: return 0
        return -(p_return - risk_free_rate) / p_volatility

    optimized_result = sco.minimize(
        fun=neg_sharpe_ratio, 
        x0=num_assets * [1.0 / num_assets], 
        args=(R, covariance_matrix, R_f),
        method='SLSQP', 
        bounds=[(0.0, 1.0) for _ in range(num_assets)], 
        constraints={'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    )

    optimal_weights = optimized_result.x
    expected_p_return, expected_p_volatility = portfolio_performance(optimal_weights, R, covariance_matrix)
    final_sharpe_ratio = -optimized_result.fun

    frontend_allocations_list = []
    allocation_string_mapping = {}
    
    for idx, asset in enumerate(discovered_assets):
        allocation_percentage = max(0.0, round(optimal_weights[idx] * 100, 2))
        allocated_amount = round((allocation_percentage / 100) * user_profile["amount"], 2)
        
        allocation_string_mapping[asset] = {"percentage": allocation_percentage, "amount_str": f"₹{allocated_amount:,}"}
        
        frontend_allocations_list.append({
            "asset_name": asset,
            "percentage": allocation_percentage,
            "amount": allocated_amount,
            "amount_str": f"₹{allocated_amount:,}"
        })

    llm_explainability_payload = {
        "user_profile": {"investment_capital_in_inr": f"₹{user_profile['amount']:,}", "investment_horizon_years": user_profile["investment_horizon"], "risk_tolerance_profile": user_profile["risk_tolerance"].upper()},
        "mathematical_outputs": {"portfolio_expected_annualized_return": f"{round(expected_p_return * 100, 2)}%", "portfolio_volatility_risk": f"{round(expected_p_volatility * 100, 2)}%", "sharpe_ratio_score": round(final_sharpe_ratio, 2)}
    }

    output_report = f"Advisory Brief for Your Investment Portfolio\n\nClient Profile: {user_profile['risk_tolerance'].upper()} Risk Tolerance, Long-Term Horizon ({user_profile['investment_horizon']} years), Investment Amount: ₹{user_profile['amount']:,}\n" + "-"*80 + "\n"

    if not using_fallback:
        try:
            explainability_prompt = f"""
            You are an elite quantitative asset manager specialized exclusively in Indian financial markets. 
            Review this complete system telemetry payload consisting of Indian user criteria and optimized allocation configurations:
            {json.dumps(llm_explainability_payload, indent=4)}
            And the exact allocations:
            {json.dumps(allocation_string_mapping, indent=4)}

            Write an elegant, deeply personalized financial advisory brief for the user based ONLY on the data inside the payload. 
            Do not mention US stocks or international funds. Double check that all metrics match the calculation fields exactly.
            Use these exact headers:
            ### 🏛️ The Core Strategy Why
            ### 📑 Stock & Instrument Drilldown
            ### 🚀 Strategic Suggestions
            """
            final_response = client.models.generate_content(model=MODEL_ID, contents=explainability_prompt, config=types.GenerateContentConfig(temperature=0.3))
            output_report += final_response.text
            return {
                "status": "success", 
                "fallback_active": False, 
                "report": output_report,
                "allocations": frontend_allocations_list
            }
        except Exception:
            using_fallback = True

    if using_fallback:
        p_ret = llm_explainability_payload["mathematical_outputs"]["portfolio_expected_annualized_return"]
        p_vol = llm_explainability_payload["mathematical_outputs"]["portfolio_volatility_risk"]
        p_sr = llm_explainability_payload["mathematical_outputs"]["sharpe_ratio_score"]
        
        output_report += "### 🏛️ The Core Strategy Why\n\nYour investment strategy has been structured using local baseline asset parameters. This approach has delivered an **expected annualized return of " + p_ret + "** combined with a managed **portfolio volatility of " + p_vol + "**, yielding a stable **Sharpe Ratio of " + str(p_sr) + "**.\n\n### 📑 Stock & Instrument Drilldown\n\n"
        for asset, data in allocation_string_mapping.items():
            if data["percentage"] > 0:
                output_report += f"* **{asset} - Allocation: {data['percentage']}% ({data['amount_str']})**\n    * **Why:** Component functions to generate optimal compounding returns within your risk parameters.\n"
            else:
                output_report += f"* **{asset} - Allocation: 0.0% (₹0.0)**\n    * **Why:** Excluded to maintain marginal efficiency bounds.\n"
        output_report += "\n### 🚀 Strategic Suggestions\n\n1. Commit to Long-Term Domestic Compounding.\n2. Enforce Routine Annual Rebalancing.\n3. Averaging Capital Extensions."
        
        return {
            "status": "success", 
            "fallback_active": True, 
            "report": output_report,
            "allocations": frontend_allocations_list
        }

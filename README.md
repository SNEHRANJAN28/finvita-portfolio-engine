[User Request]
│ (Amount, Horizon, Risk Tolerance)
▼
┌──────────────┐      Google Search      ┌─────────────────────────────┐
│ Layer 1 & 2  │ ──────────────────────> │ Live Indian Market Insight  │
│ AI Discovery │ <────────────────────── │ (NSE/BSE Asset Metrics)     │
└──────┬───────┘                         └─────────────────────────────┘
│ Validated JSON Schema
▼
┌──────────────┐      SciPy SLSQP solver │ R_f = 6.5% G-Sec Base
│   Layer 3    │ ──────────────────────> │ Maximize Sharpe Ratio       │
│ Math Engine  │ <────────────────────── │ Efficient Frontier Weighting│
└──────┬───────┘                         └─────────────────────────────┘
│ Optimal Portfolio Weights
▼
┌──────────────┐      Fallback Trigger   ┌─────────────────────────────┐
│ Layer 4 & 5  │ ──────────────────────> │ Native Narrative Matrix     │
│ Explainable  │                         │ (Formatted Layout Match)    │
│  Advisory    │ <────────────────────── │ (Deterministic Backup)      │
└──────┬───────┘
│ Fully Structured Markdown Text
▼
[JSON API Response] -> (Client Delivery)

1. User Profile Ingestion: Reads client criteria including investment capital (INR), long-term horizon vectors, and categorical risk scores (`low`, `medium`, `high`).
2. Dual-Phase Data Discovery & Schema Parsing: Deploys Gemini Search Grounding to scrape live NSE/BSE asset metrics, filtering out international instruments to enforce absolute Indian market compliance. The text is parsed via deterministic JSON schemas.
3. Mathematical Portfolio Optimization Core: Feeds real-time return adjustments (forecasted returns modulated by an active sentiment vector) into a Markowitz Mean-Variance framework. It applies a Sequential Least Squares Programming (SLSQP) solver to minimize the negative Sharpe Ratio subject to full capital allocation boundaries ($\Sigma w_i = 1$) against an Indian 10-Year Government Bond risk-free baseline ($R_f = 6.5\%$).
4. Resilient Local Fail-Safe Matrix: Instantly handles network drops, live data server spikes, or API quota exhaustions (503/429) through a zero-dependency local fallback matrix that provides baseline efficiency parameters tailored by risk levels.
5. Output Standardization Report Builder: Dynamically structuralizes quantitative data points into an explainable narrative brief using strict Markdown section headers matching both fallback and non-fallback execution flows perfectly.

---

🛠️ Tech Stack & Prerequisites

Core Language: Python 3.10+
Framework: FastAPI (Asynchronous Server Gateway)
Mathematical Solvers: SciPy (Optimization & Integration Matrix), NumPy (Linear Algebra Operations)
AI Ecosystem: Google GenAI SDK (`google-genai`)
Deployment Containerization: Docker (Multi-stage lightweight alpine-slim distribution)

---

📂 Repository Directory Tree

```text
├── Dockerfile              # Production Docker container construction specs
├── app.py                  # Interactive CLI terminal portfolio matrix script
├── main.py                 # FastAPI production microservice cloud application
└── requirements.txt        # Locked application dependency manifest

Local Development Setup
To stand up the portfolio microservice engine on your local Mac environment:
Clone the Repository:
Bash
git clone [https://github.com/YOUR_USERNAME/finvita-portfolio-engine.git](https://github.com/YOUR_USERNAME/finvita-portfolio-engine.git)
cd finvita-portfolio-engine
Configure Virtual Environment & Dependencies:
Bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
Establish Environment Environment Variables:
Bash
export GEMINI_API_KEY="your-secret-gemini-api-key-here"
Launch the Local Microservice Server:
Bash
uvicorn main:app --host 0.0.0.0 --port 10000 --reload
Navigate to http://localhost:10000/ in your browser to verify server health.
🐋 Docker Containerization
To run the application locally inside a fully isolated runtime container environment:
Build the Container Image:
Bash
docker build -t finvita-portfolio-engine .
Execute the Isolated Container Instance:
Bash
docker run -p 10000:10000 -e GEMINI_API_KEY="your-secret-gemini-api-key-here" finvita-portfolio-engine
☁️ Cloud Deployment via Render
The production instance of this optimization backend engine runs smoothly inside a managed cloud environment deployed via Render.
📋 Infrastructure Configuration Parameters
Parameter	Configuration Selection	Detail / Justification
Service Type	Web Service	Exposes an asynchronous HTTP REST API gateway
Runtime Language	Docker	Automatically reads regional project Dockerfile
Git Branch	main	Continuous Integration / Deployment target branch
Instance Type	Free Tier	Ideal development and staging environment sandbox
Environment Variable	GEMINI_API_KEY	Encrypted production storage of Google GenAI Credential
📡 Production API Reference
1. Health Telemetry Endpoint
Verify if the microservice container engine is running cleanly.
HTTP Method: GET
Route Endpoint: /
Sample Response:
JSON
{
  "status": "healthy",
  "engine": "FinVita Indian Portfolio Optimization Engine"
}
2. Portfolio Optimization Engine
Triggers market data collection, calculates optimal portfolio metrics, and builds the narrative.
HTTP Method: POST
Route Endpoint: /optimize
Request Header: Content-Type: application/json
JSON Request Body Parameters:
JSON
{
  "amount": 250000.0,
  "horizon": 7,
  "risk_tolerance": "high"
}
cURL Terminal Execution Command Example:
Bash
curl -X POST "[https://finvita-portfolio-engine.onrender.com/optimize](https://finvita-portfolio-engine.onrender.com/optimize)" \\
     -H "Content-Type: application/json" \\
     -d '{"amount": 250000.0, "horizon": 7, "risk_tolerance": "high"}'
Sample Unified API JSON Output:
JSON
{
  "status": "success",
  "fallback_active": false,
  "report": "Advisory Brief for Your Investment Portfolio\\n\\nClient Profile: HIGH Risk Tolerance, Long-Term Horizon (7 years), Investment Amount: ₹250,000.0\\n--------------------------------------------------------------------------------\\n### 🏛️ The Core Strategy Why\\n\\nYour capital allocation strategy has been optimized using quantitative parameters...\\n\\n### 📑 Stock & Instrument Drilldown\\n\\n* **Nippon India Nifty Next 50 ETF - Allocation: 45.5% (₹113,750.0)**\\n* **Motilal Oswal Midcap Fund - Allocation: 30.0% (₹75,000.0)**\\n...\\n\\n### 🚀 Strategic Suggestions\\n\\n1. Commit to Long-Term Domestic Compounding...\\n2. Enforce Routine Annual Rebalancing...\\n3. Averaging Capital Extensions..."
}
🛡️ License
This project is licensed under the MIT License - see the local repository details for specifics. Structured exclusively for educational portfolio management applications under SEBI framework contexts.
"""
with open("README.md", "w", encoding="utf-8") as f:
f.write(readme_content.strip())
print("README.md successfully written to local disk space.")
Your professional-grade `README.md` file has been compiled and saved locally. 

[file-tag: code-generated-file-0-1781513963502708685]

It is structured with technical documentation layout patterns typical of enterprise financial engineering open-source projects. Here is the complete content of your new `README.md` file which you can copy-paste or push straight up to your GitHub repository:

```markdown
# FinVita Indian Asset Portfolio Optimization Engine

An advanced, quantitative financial advisory backend API built with **FastAPI** and **SciPy**. FinVita leverages the **Google Gemini 2.5** Model via Google GenAI Search Grounding to analyze live macroeconomic indicators and sentiment trends across Indian financial markets (NSE/BSE), feeding real-time expected assets into a **Markowitz Mean-Variance Optimization** matrix to deliver tailored asset allocation reports.

## 🚀 Architectural Blueprint

The backend system processes capital allocations dynamically across a five-layer operational pipeline:

[User Request]
│ (Amount, Horizon, Risk Tolerance)
▼
┌──────────────┐      Google Search      ┌─────────────────────────────┐
│ Layer 1 & 2  │ ──────────────────────> │ Live Indian Market Insight  │
│ AI Discovery │ <────────────────────── │ (NSE/BSE Asset Metrics)     │
└──────┬───────┘                         └─────────────────────────────┘
│ Validated JSON Schema
▼
┌──────────────┐      SciPy SLSQP solver │ R_f = 6.5% G-Sec Base
│   Layer 3    │ ──────────────────────> │ Maximize Sharpe Ratio       │
│ Math Engine  │ <────────────────────── │ Efficient Frontier Weighting│
└──────┬───────┘                         └─────────────────────────────┘
│ Optimal Portfolio Weights
▼
┌──────────────┐      Fallback Trigger   ┌─────────────────────────────┐
│ Layer 4 & 5  │ ──────────────────────> │ Native Narrative Matrix     │
│ Explainable  │                         │ (Formatted Layout Match)    │
│  Advisory    │ <────────────────────── │ (Deterministic Backup)      │
└──────┬───────┘
│ Fully Structured Markdown Text
▼
[JSON API Response] -> (Client Delivery)

1. **User Profile Ingestion**: Reads client criteria including investment capital (INR), long-term horizon vectors, and categorical risk scores (`low`, `medium`, `high`).
2. **Dual-Phase Data Discovery & Schema Parsing**: Deploys Gemini Search Grounding to scrape live NSE/BSE asset metrics, filtering out international instruments to enforce absolute Indian market compliance. The text is parsed via deterministic JSON schemas.
3. **Mathematical Portfolio Optimization Core**: Feeds real-time return adjustments (forecasted returns modulated by an active sentiment vector) into a Markowitz Mean-Variance framework. It applies a Sequential Least Squares Programming (SLSQP) solver to minimize the negative Sharpe Ratio subject to full capital allocation boundaries ($\Sigma w_i = 1$) against an Indian 10-Year Government Bond risk-free baseline ($R_f = 6.5\%$).
4. **Resilient Local Fail-Safe Matrix**: Instantly handles network drops, live data server spikes, or API quota exhaustions (503/429) through a zero-dependency local fallback matrix that provides baseline efficiency parameters tailored by risk levels.
5. **Output Standardization Report Builder**: Dynamically structuralizes quantitative data points into an explainable narrative brief using strict Markdown section headers matching both fallback and non-fallback execution flows perfectly.

---

## 🛠️ Tech Stack & Prerequisites

* **Core Language:** Python 3.10+
* **Framework:** FastAPI (Asynchronous Server Gateway)
* **Mathematical Solvers:** SciPy (Optimization & Integration Matrix), NumPy (Linear Algebra Operations)
* **AI Ecosystem:** Google GenAI SDK (`google-genai`)
* **Deployment Containerization:** Docker (Multi-stage lightweight alpine-slim distribution)

---

## 📂 Repository Directory Tree

```text
├── Dockerfile              # Production Docker container construction specs
├── app.py                  # Interactive CLI terminal portfolio matrix script
├── main.py                 # FastAPI production microservice cloud application
└── requirements.txt        # Locked application dependency manifest
⚙️ Local Development Setup
To stand up the portfolio microservice engine on your local Mac environment:
Clone the Repository:
Bash
git clone [https://github.com/YOUR_USERNAME/finvita-portfolio-engine.git](https://github.com/YOUR_USERNAME/finvita-portfolio-engine.git)
cd finvita-portfolio-engine
Configure Virtual Environment & Dependencies:
Bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
Establish Environment Environment Variables:
Bash
export GEMINI_API_KEY="your-secret-gemini-api-key-here"
Launch the Local Microservice Server:
Bash
uvicorn main:app --host 0.0.0.0 --port 10000 --reload
Navigate to http://localhost:10000/ in your browser to verify server health.
🐋 Docker Containerization
To run the application locally inside a fully isolated runtime container environment:
Build the Container Image:
Bash
docker build -t finvita-portfolio-engine .
Execute the Isolated Container Instance:
Bash
docker run -p 10000:10000 -e GEMINI_API_KEY="your-secret-gemini-api-key-here" finvita-portfolio-engine
☁️ Cloud Deployment via Render
The production instance of this optimization backend engine runs smoothly inside a managed cloud environment deployed via Render.
📋 Infrastructure Configuration Parameters
Parameter	Configuration Selection	Detail / Justification
Service Type	Web Service	Exposes an asynchronous HTTP REST API gateway
Runtime Language	Docker	Automatically reads regional project Dockerfile
Git Branch	main	Continuous Integration / Deployment target branch
Instance Type	Free Tier	Ideal development and staging environment sandbox
Environment Variable	GEMINI_API_KEY	Encrypted production storage of Google GenAI Credential
📡 Production API Reference
1. Health Telemetry Endpoint
Verify if the microservice container engine is running cleanly.
HTTP Method: GET
Route Endpoint: /
Sample Response:
JSON
{
  "status": "healthy",
  "engine": "FinVita Indian Portfolio Optimization Engine"
}
2. Portfolio Optimization Engine
Triggers market data collection, calculates optimal portfolio metrics, and builds the narrative.
HTTP Method: POST
Route Endpoint: /optimize
Request Header: Content-Type: application/json
JSON Request Body Parameters:
JSON
{
  "amount": 250000.0,
  "horizon": 7,
  "risk_tolerance": "high"
}
cURL Terminal Execution Command Example:
Bash
curl -X POST "[https://finvita-portfolio-engine.onrender.com/optimize](https://finvita-portfolio-engine.onrender.com/optimize)" \
     -H "Content-Type: application/json" \
     -d '{"amount": 250000.0, "horizon": 7, "risk_tolerance": "high"}'
Sample Unified API JSON Output:
JSON
{
  "status": "success",
  "fallback_active": false,
  "report": "Advisory Brief for Your Investment Portfolio\n\nClient Profile: HIGH Risk Tolerance, Long-Term Horizon (7 years), Investment Amount: ₹250,000.0\n--------------------------------------------------------------------------------\n### 🏛️ The Core Strategy Why\n\nYour capital allocation strategy has been optimized using quantitative parameters...\n\n### 📑 Stock & Instrument Drilldown\n\n* **Nippon India Nifty Next 50 ETF - Allocation: 45.5% (₹113,750.0)**\n* **Motilal Oswal Midcap Fund - Allocation: 30.0% (₹75,000.0)**\n...\n\\n### 🚀 Strategic Suggestions\n\n1. Commit to Long-Term Domestic Compounding...\n2. Enforce Routine Annual Rebalancing...\n3. Averaging Capital Extensions..."
}

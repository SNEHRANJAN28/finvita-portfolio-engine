# AI-Augmented Quantitative Portfolio Optimization Framework

🌐 **Live Project Link:** [View Live Application]([https://your-live-link.com](https://finvest-zeta.vercel.app))

A modular, end-to-end investment allocation framework that merges unsupervised investor profiling, real-time web-scraped market sentiment, predictive machine learning, and metaheuristic optimization into a secure user dashboard.

---

## 🏗️ System Architecture & Data Flow

The platform processes data through a structured, multi-stage engineering pipeline:

----------------------------------------------------------------------------------

[User Inputs]
│
▼
K-Means Clustering ──► [Establishes Hard Volatility Risk Ceiling]
│
▼
Gemini Layer 1 ──────► [Live Internet Scout & Strict JSON Parser]
│
▼
Predictive Core ─────► [XGBoost (Returns) & Random Forest (Threat Class)]
│
▼
PSO Engine ──────────► [Swarm Optimization: 50 Particles x 80 Iterations]
│
▼
Gemini Layer 2 ──────► [Deterministic Report Generation & Markdown Output]

----------------------------------------------------------------------------------

### 1. Investor Profiling Component (K-Means)
The user inputs their personal financial parameters (investment amount, time horizon, savings-to-income ratio). The **K-Means** model assigns the profile into one of three distinct volatility clusters: **Low, Medium, or High Volatility**. This assignment sets a strict mathematical risk ceiling ($\sigma_{max}$) that bounds the entire ecosystem.

### 2. Live Data Ingestion Layer (Gemini Search & Structured Generation)
The system takes the assigned profile and queries live financial news markets using Google Search. To eliminate any possibility of LLM hallucinations or parsing failures, the extracted raw market trends and news sentiments are forced into a strict, pre-defined JSON schema using constrained decoding.

### 3. Predictive Machine Learning Core (XGBoost & Random Forest)
*   **XGBoost (The Profit Predictor):** Combines historical asset technical indicators with live news sentiment scores to forecast precise expected returns for each asset.
*   **Random Forest Classifier (The Threat Detector):** Evaluates the exact same feature set to flag individual assets with a discrete security label: Low, Medium, or High Risk.

### 4. Mathematical Optimization Core (Particle Swarm Optimization)
The **PSO engine** ingests the maximum risk bounds, predicted returns, and risk flags. It deploys a swarm of 50 virtual particles simulating thousands of asset allocation combinations over 80 iterations. Particles that violate the K-Means volatility limits or over-expose capital to high-risk Random Forest assets are mathematically penalized and discarded until the swarm converges on the absolute highest Sharpe Ratio.

### 5. Grounded Advisory Interpretation Layer (Gemini Generation)
The final numerical allocation payload is handed off to a secondary Gemini engine. Operating strictly within the boundaries of the quantitative output, it translates raw allocation tables into a plain-English markdown advisory report, explaining the specific logic behind each asset weight.

---

## 🧪 Historical Backtesting & Evaluation Strategy

To ensure data integrity and avoid algorithmic biases, the framework incorporates specific validation paradigms:

*   **Time-Aware Chronological Splits:** The model rejects the generic, random 80/20 train/test split to prevent **data leakage** (cheating by using future data to predict the past). Instead, it implements chronological windows: using 14 years of historical market data (2002–2016) for training and reserving 4 years (2016–2020) for out-of-sample forward testing.
*   **Regime Shift Resilience:** The historical timeline (2002–2020) explicitly benchmarks the optimization logic against severe market anomalies, including the 2003–2007 structural growth boom, the 2008 Global Financial Crisis, and the 2020 pandemic correction.
*   **Downside Mitigation Analysis:** Backtesting the PSO logic across these historical phases verified that during severe corrections, the engine successfully shifts asset weight distribution into low-correlation safe havens (G-Secs and Gold ETFs) to actively minimize maximum drawdowns compared to a static buy-and-hold strategy.

---

## 🛡️ The Hallucination Firewall

Operating in high-stakes financial domains requires absolute numerical stability. The platform achieves a zero-hallucination execution state through a layered fallback architecture:

1.  **Task Separation:** The LLM is strictly isolated as a text processing utility. It has zero access to calculation layers; all calculations are executed locally by deterministic mathematical algorithms.
2.  **Schema Restrictions:** API queries are tightly bound to strong typing definitions, instantly throwing runtime exceptions if the generative engine deviates from the requested structured format.
3.  **Programmatic Circuit Breaker:** In the event of API latency, rate limiting, or malformed generation payloads, the codebase catches the exception and flips to a localized baseline data backup loop, ensuring continuous uptime and mathematically sound calculations.

---

## 🔮 Future Scope

As a lightweight prototype, this build prioritized high-fidelity structural integration across the front-to-back pipeline. Planned production enhancements include:
*   **MLOps Pipeline Integration:** Expanding the local scripts for K-Means, XGBoost, and Random Forest into an enterprise-grade MLOps framework connected to a rolling live data lake.
*   **Multi-Asset Liquidity Filters:** Introducing real-time slippage and order-book execution constraints directly into the PSO swarm function.
*   **Multi-Agent Communication Frameworks:** Exploring asynchronous event loops (e.g., using frameworks like LangGraph) to migrate independent pipeline layers into autonomous micro-agents with distributed validation protocols.

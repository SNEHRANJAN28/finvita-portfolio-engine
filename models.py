from pydantic import BaseModel, Field

class PortfolioRequest(BaseModel):
    amount: float = Field(50000.0, description="Total investment capital in INR")
    horizon: int = Field(5, description="Investment horizon timeline in years")
    risk_tolerance: str = Field("medium", description="Categorical risk tolerance level (low, medium, high)")
    savings_to_income_ratio: float = Field(0.30, description="Proportion of monthly income allocated to savings")
    primary_investment_goal: str = Field("wealth_accumulation", description="Primary objective")
    preferred_rebalancing_frequency: str = Field("annually", description="Desired adjustment intervals")

class SimulationRequest(BaseModel):
    weights: list[float]
    returns: list[float]
    panic_level: float = Field(0.0, ge=0.0, le=1.0)

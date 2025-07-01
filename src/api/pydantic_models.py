from pydantic import BaseModel, Field, confloat
from typing import Optional

# 🧾 Input schema for /predict request
class CustomerData(BaseModel):
    Recency: float = Field(..., ge=0, description="Days since last transaction")
    Frequency: float = Field(..., ge=0, description="Number of purchases")
    Monetary: float = Field(..., ge=0, description="Total amount spent")
    AvgTransactionAmount: float = Field(..., ge=0, description="Average spend per transaction")
    Age: Optional[float] = Field(None, ge=0, description="Customer age (optional)")

# ✅ Output schema for /predict response
class PredictionResult(BaseModel):
    prediction: int = Field(..., description="0 = low risk, 1 = high risk")
    risk_probability: confloat(ge=0.0, le=1.0) = Field(..., description="Probability of being high risk")

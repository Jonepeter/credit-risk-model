"""
Pydantic models for FastAPI endpoints.
"""

# ... define your Pydantic models here ... 
from pydantic import BaseModel, Field
from typing import List, Optional, Any

class FeaturesModel(BaseModel):
    mode_ProviderId_ProviderId_6: Optional[float] = None
    mode_PricingStrategy_2: Optional[float] = None
    avg_transaction_dayofweek: Optional[float] = None
    mode_ProviderId_ProviderId_5: Optional[float] = None
    unique_PricingStrategy_count: Optional[int] = None
    avg_transaction_amount: Optional[float] = None
    avg_transaction_day: Optional[float] = None
    unique_ProviderId_count: Optional[int] = None
    mode_ChannelId_ChannelId_2: Optional[float] = None
    mode_ChannelId_ChannelId_3: Optional[float] = None
    transaction_count: Optional[int] = None
    total_transaction_amount: Optional[float] = None
    avg_value: Optional[float] = None
    min_transaction_year: Optional[int] = None
    unique_ChannelId_count: Optional[int] = None
    std_transaction_amount: Optional[float] = None
    max_transaction_year: Optional[int] = None
    avg_transaction_hour: Optional[float] = None
    avg_transaction_month: Optional[float] = None
    mode_ProviderId_ProviderId_1: Optional[float] = None
    mode_PricingStrategy_4: Optional[float] = None
    unique_ProductCategory_count: Optional[int] = None
    total_value: Optional[float] = None


class TransactionsRequest(BaseModel):
    transactions: List[FeaturesModel]

class PredictionResult(BaseModel):
    CustomerId: int
    risk_probability: float = Field(..., ge=0.0, le=1.0)
    predicted_high_risk: int

class PredictionsResponse(BaseModel):
    results: List[PredictionResult]

class ErrorResponse(BaseModel):
    detail: str


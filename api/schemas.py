from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    """Transaction payload accepted by the inference API."""

    transaction_id: str = Field(..., min_length=1, description="Unique transaction identifier.")
    data: dict[str, Any] = Field(
        ..., description="Transaction fields matching the trained FraudX feature schema."
    )


class PredictionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    prediction: int
    threshold: float
    model_version: str
    persisted: bool


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    mongodb_connected: bool


class ModelInfoResponse(BaseModel):
    model_version: str
    threshold: float
    ensemble_weights: dict[str, float]
    feature_count: int

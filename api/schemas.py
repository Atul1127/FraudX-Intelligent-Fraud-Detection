from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    """Raw transaction payload accepted by the inference API."""

    data: dict[str, Any] = Field(
        ..., description="Raw IEEE-CIS transaction fields for a single row."
    )


class PredictionResponse(BaseModel):
    fraud_probability: float
    prediction: int
    threshold: float
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_version: str
    threshold: float
    ensemble_weights: dict[str, float]
    feature_count: int

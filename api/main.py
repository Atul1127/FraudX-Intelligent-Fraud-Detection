from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException

from api.dependencies import load_model
from api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    TransactionRequest,
)


model = None
cfg: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, cfg
    try:
        model, cfg = load_model()
    except FileNotFoundError:
        # The API can still start before a trained checkpoint exists.
        model, cfg = None, {}
    yield
    model = None


app = FastAPI(
    title="FraudX API",
    description="Production-oriented fraud detection inference API.",
    version="1.0.0",
    lifespan=lifespan,
)


def _require_model():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model checkpoint is not available. Train FraudX before serving predictions.",
        )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=model is not None)


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    _require_model()
    threshold = float(cfg.get("ensemble", {}).get("default_threshold", 0.5))
    return ModelInfoResponse(
        model_version="local-checkpoint",
        threshold=threshold,
        ensemble_weights=dict(model.WEIGHTS),
        feature_count=len(model.feature_names or []),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: TransactionRequest) -> PredictionResponse:
    """Score a preprocessed transaction row.

    The payload is intentionally accepted as a dictionary so the API can evolve
    with the IEEE-CIS feature schema. Full raw-row feature construction will be
    added in the next serving phase once stateful historical features are backed
    by MongoDB.
    """
    _require_model()

    row = pd.DataFrame([request.data])
    missing = [name for name in model.feature_names or [] if name not in row.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Payload is missing model features.",
                "missing_features": missing[:50],
                "missing_count": len(missing),
            },
        )

    row = row[model.feature_names]
    try:
        probability = float(model.predict_proba(row)[0])
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {exc}") from exc

    threshold = float(cfg.get("ensemble", {}).get("default_threshold", 0.5))
    return PredictionResponse(
        fraud_probability=probability,
        prediction=int(probability >= threshold),
        threshold=threshold,
        model_version="local-checkpoint",
    )

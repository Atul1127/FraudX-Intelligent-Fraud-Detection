from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException

from api.dependencies import load_model
from api.mongodb import MongoStore
from api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    TransactionRequest,
)


model = None
cfg: dict[str, Any] = {}
mongo = MongoStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, cfg
    try:
        model, cfg = load_model()
    except FileNotFoundError:
        model, cfg = None, {}

    # MongoDB is optional during local development; the API remains available
    # when the database is not running.
    mongo.connect()
    yield
    mongo.close()
    model = None


app = FastAPI(
    title="FraudX API",
    description="Production-oriented fraud detection inference API.",
    version="1.1.0",
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
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        mongodb_connected=mongo.connected,
    )


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
    """Score a transaction and persist the transaction/prediction when MongoDB is available.

    Historical feature construction is intentionally not performed here yet.
    The next serving step will use MongoDB-backed transaction history to build
    causal velocity/frequency features before inference.
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
    prediction = int(probability >= threshold)
    model_version = "local-checkpoint"
    persisted = False

    if mongo.connected:
        try:
            mongo.save_transaction(request.transaction_id, request.data)
            mongo.save_prediction(
                request.transaction_id,
                probability,
                prediction,
                threshold,
                model_version,
            )
            mongo.save_audit(
                "prediction",
                {
                    "transaction_id": request.transaction_id,
                    "prediction": prediction,
                    "model_version": model_version,
                },
            )
            persisted = True
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"MongoDB persistence failed: {exc}") from exc

    return PredictionResponse(
        transaction_id=request.transaction_id,
        fraud_probability=probability,
        prediction=prediction,
        threshold=threshold,
        model_version=model_version,
        persisted=persisted,
    )

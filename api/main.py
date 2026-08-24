from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query

from api.dependencies import load_model
from api.feature_store import OnlineFeatureStore
from api.mongodb import MongoStore
from api.schemas import HealthResponse, ModelInfoResponse, PredictionResponse, TransactionRequest
from src.monitoring.drift import prediction_shift


model = None
cfg: dict[str, Any] = {}
mongo = MongoStore()
feature_store: OnlineFeatureStore | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, cfg, feature_store
    try:
        model, cfg = load_model()
    except FileNotFoundError:
        model, cfg = None, {}
    mongo.connect()
    feature_store = OnlineFeatureStore(mongo, cfg) if model is not None else None
    yield
    mongo.close()
    feature_store = None
    model = None


app = FastAPI(
    title="FraudX API",
    description="Production-oriented fraud detection inference API.",
    version="1.4.0",
    lifespan=lifespan,
)


def _require_model():
    if model is None:
        raise HTTPException(status_code=503, detail="Model checkpoint is not available. Train FraudX before serving predictions.")


def _require_online_serving():
    _require_model()
    if not mongo.connected or feature_store is None:
        raise HTTPException(status_code=503, detail="MongoDB is required for online historical feature construction.")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=model is not None, mongodb_connected=mongo.connected)


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


@app.get("/monitoring/predictions")
def prediction_monitoring(window_hours: int = Query(24, ge=1, le=168)) -> dict[str, Any]:
    """Monitor prediction volume, fraud rate, score shift, and PSI drift."""
    if not mongo.connected:
        raise HTTPException(status_code=503, detail="MongoDB is required for prediction monitoring.")
    data = mongo.get_prediction_monitoring(window_hours)
    current, previous = data["current"], data["previous"]
    result: dict[str, Any] = {
        "window_hours": window_hours,
        "current": {k: v for k, v in current.items() if k != "probabilities"},
        "previous": {k: v for k, v in previous.items() if k != "probabilities"},
        "status": "insufficient_data",
    }
    if current["probabilities"] and previous["probabilities"]:
        shift = prediction_shift(previous["probabilities"], current["probabilities"])
        result["prediction_drift"] = shift
        result["status"] = shift["level"]
    else:
        result["prediction_drift"] = {"psi": None, "level": "insufficient_data"}
    return result


@app.get("/monitoring/features")
def feature_monitoring(window_hours: int = Query(24, ge=1, le=168)) -> dict[str, Any]:
    """Compare online numeric feature distributions across adjacent windows."""
    if not mongo.connected:
        raise HTTPException(status_code=503, detail="MongoDB is required for feature monitoring.")
    data = mongo.get_feature_monitoring(window_hours)
    fields: dict[str, Any] = {}
    drift_levels: list[str] = []
    for field, current in data["current"].items():
        previous = data["previous"].get(field, [])
        if current and previous:
            shift = prediction_shift(previous, current)
            fields[field] = {"psi": shift["psi"], "level": shift["level"], "current_count": len(current), "previous_count": len(previous)}
            drift_levels.append(shift["level"])
        else:
            fields[field] = {"psi": None, "level": "insufficient_data", "current_count": len(current), "previous_count": len(previous)}
    status = "drift" if "drift" in drift_levels else "warning" if "warning" in drift_levels else "stable" if drift_levels else "insufficient_data"
    return {"window_hours": window_hours, "status": status, "features": fields}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: TransactionRequest) -> PredictionResponse:
    """Score a raw transaction using MongoDB-backed historical features."""
    _require_online_serving()
    try:
        row = feature_store.build(request.data, model)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Feature construction failed: {exc}") from exc

    try:
        probability = float(model.predict_proba(row)[0])
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {exc}") from exc

    threshold = float(cfg.get("ensemble", {}).get("default_threshold", 0.5))
    prediction = int(probability >= threshold)
    model_version = "local-checkpoint"
    try:
        mongo.save_transaction(request.transaction_id, request.data)
        mongo.save_prediction(request.transaction_id, probability, prediction, threshold, model_version)
        mongo.save_audit("prediction", {"transaction_id": request.transaction_id, "prediction": prediction, "model_version": model_version})
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"MongoDB persistence failed: {exc}") from exc

    return PredictionResponse(
        transaction_id=request.transaction_id,
        fraud_probability=probability,
        prediction=prediction,
        threshold=threshold,
        model_version=model_version,
        persisted=True,
    )

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow


EXPERIMENT_NAME = "FraudX-Fraud-Detection"


def _flatten_params(value: Any, prefix: str = "") -> dict[str, str]:
    """Flatten nested config into MLflow-safe string parameters."""
    params: dict[str, str] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            params.update(_flatten_params(child, name))
    elif isinstance(value, (list, tuple)):
        params[prefix] = json.dumps(value, default=str)
    else:
        params[prefix] = str(value)
    return params


def start_run(cfg: dict) -> Any:
    mlflow_cfg = cfg.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri", "file:./mlruns")
    experiment_name = mlflow_cfg.get("experiment_name", EXPERIMENT_NAME)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return mlflow.start_run(run_name=mlflow_cfg.get("run_name"))


def log_training_run(cfg: dict, report: dict, checkpoint_dir: str | Path) -> None:
    """Log FraudX training configuration, metrics and model artifacts."""
    checkpoint_dir = Path(checkpoint_dir)
    params = _flatten_params(cfg)

    # MLflow parameter values have size limits; keep the run useful by logging
    # the complete config as an artifact as well.
    for key, value in params.items():
        if len(value) <= 500:
            mlflow.log_param(key[:250], value)

    metric_keys = {
        "auc_roc",
        "auc_pr",
        "precision",
        "recall",
        "f1",
        "best_threshold",
    }
    for key in metric_keys:
        value = report.get(key)
        if value is not None:
            mlflow.log_metric(key, float(value))

    for model_name, metrics in report.get("model_comparison", {}).items():
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"{model_name}_{metric_name}", float(value))

    config_path = checkpoint_dir / "mlflow_config.json"
    config_path.write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")
    mlflow.log_artifact(str(config_path), artifact_path="metadata")
    mlflow.log_artifact(str(checkpoint_dir / "training_report.json"), artifact_path="metadata")
    mlflow.log_artifacts(str(checkpoint_dir), artifact_path="model")
    config_path.unlink(missing_ok=True)

    mlflow.set_tags(
        {
            "project": "FraudX",
            "model_type": "weighted_ensemble",
            "models": "XGBoost,LightGBM,CatBoost",
            "serving": "FastAPI+MongoDB",
        }
    )

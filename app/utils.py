from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st
import yaml

# Allow imports from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_config(path: str = "config.yaml") -> dict:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open() as f:
        return yaml.safe_load(f)


@st.cache_resource
def load_model(checkpoint_dir: str, _cfg_key: str):
    from src.models.ensemble import FraudEnsemble

    cfg = load_config()
    return FraudEnsemble.load(checkpoint_dir, cfg)


@st.cache_data
def load_val_data(proc_dir: str):
    import pickle

    proc_path = Path(proc_dir)
    if not proc_path.is_absolute():
        proc_path = PROJECT_ROOT / proc_path
    with (proc_path / "features_val.pkl").open("rb") as f:
        X_val, y_val = pickle.load(f)
    return X_val, y_val


@st.cache_data
def compute_val_probas(checkpoint_dir: str, proc_dir: str):
    model = load_model(checkpoint_dir, checkpoint_dir)
    X_val, y_val = load_val_data(proc_dir)
    probas = model.predict_proba(X_val)
    return y_val.values, probas


@st.cache_resource
def load_explainers(checkpoint_dir: str, proc_dir: str, _cfg_key: str):
    from src.explain import build_explainers

    model = load_model(checkpoint_dir, checkpoint_dir)
    X_val, _ = load_val_data(proc_dir)
    background = X_val.sample(min(200, len(X_val)), random_state=42)
    return build_explainers(model, background)


def load_training_report(checkpoint_dir: str) -> dict | None:
    path = Path(checkpoint_dir)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    report_path = path / "training_report.json"
    if not report_path.exists():
        return None
    with report_path.open() as f:
        return json.load(f)


def checkpoint_ready(checkpoint_dir: str = "models/checkpoints") -> bool:
    ckpt = Path(checkpoint_dir)
    if not ckpt.is_absolute():
        ckpt = PROJECT_ROOT / ckpt
    required = [
        "xgb_model.joblib",
        "lgb_model.joblib",
        "cat_model.joblib",
        "feature_names.joblib",
    ]
    return all((ckpt / filename).exists() for filename in required)


def processed_data_ready(proc_dir: str = "data/processed") -> bool:
    proc = Path(proc_dir)
    if not proc.is_absolute():
        proc = PROJECT_ROOT / proc
    return (proc / "features_val.pkl").exists()

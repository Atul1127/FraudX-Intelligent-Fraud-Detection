from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st
import yaml

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@st.cache_resource
def load_model(checkpoint_dir: str, _cfg_key: str):
    from src.models.ensemble import FraudEnsemble
    cfg = load_config()
    return FraudEnsemble.load(checkpoint_dir, cfg)


@st.cache_data
def load_val_data(proc_dir: str):
    import pickle
    with open(Path(proc_dir) / "features_val.pkl", "rb") as f:
        X_val, y_val = pickle.load(f)
    return X_val, y_val


@st.cache_data
def compute_val_probas(checkpoint_dir: str, proc_dir: str):
    import numpy as np
    model = load_model(checkpoint_dir, checkpoint_dir)
    X_val, y_val = load_val_data(proc_dir)
    probas = model.predict_proba(X_val)
    return y_val.values, probas


@st.cache_resource
def load_explainers(checkpoint_dir: str, proc_dir: str, _cfg_key: str):
    from src.explain import build_explainers
    import pickle
    model = load_model(checkpoint_dir, checkpoint_dir)
    with open(Path(proc_dir) / "features_val.pkl", "rb") as f:
        X_val, _ = pickle.load(f)
    background = X_val.sample(min(200, len(X_val)), random_state=42)
    return build_explainers(model, background)


def load_training_report(checkpoint_dir: str) -> dict | None:
    path = Path(checkpoint_dir) / "training_report.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def checkpoint_ready(checkpoint_dir: str = "models/checkpoints") -> bool:
    ckpt = Path(checkpoint_dir)
    return (ckpt / "xgb.pkl").exists() and (ckpt / "lgbm.pkl").exists()


def processed_data_ready(proc_dir: str = "data/processed") -> bool:
    proc = Path(proc_dir)
    return (proc / "features_val.pkl").exists()

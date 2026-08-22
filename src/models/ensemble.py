from __future__ import annotations

import json
from pathlib import Path

import joblib
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


class FraudEnsemble:
    """Train and serve the weighted XGBoost/LightGBM/CatBoost ensemble."""

    WEIGHTS = {"XGBoost": 0.35, "LightGBM": 0.35, "CatBoost": 0.30}

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.feature_names: list[str] | None = None

        tuned_path = Path(cfg.get("optuna", {}).get("best_params_path", "models/optuna/best_params.json"))
        tuned = {}
        if tuned_path.exists():
            with tuned_path.open() as f:
                tuned = json.load(f)

        seed = cfg["data"]["random_seed"]
        xgb_cfg = cfg["xgboost"]
        lgb_cfg = cfg["lightgbm"]

        self.xgb_model = XGBClassifier(
            **tuned.get("XGBoost", {}).get("params", {}),
            eval_metric=xgb_cfg["eval_metric"],
            tree_method=xgb_cfg["tree_method"],
            random_state=seed,
        )
        self.lgb_model = LGBMClassifier(
            **tuned.get("LightGBM", {}).get("params", {}),
            is_unbalance=True,
            verbose=-1,
            random_state=seed,
        )
        self.cat_model = CatBoostClassifier(
            **tuned.get("CatBoost", {}).get("params", {}),
            auto_class_weights="Balanced",
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=seed,
            verbose=False,
        )

    def fit(self, X_train, y_train, X_val, y_val):
        self.feature_names = list(X_train.columns)
        self.xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        self.lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        self.cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        return self

    def predict_individual(self, X):
        probabilities = {
            "XGBoost": self.xgb_model.predict_proba(X)[:, 1],
            "LightGBM": self.lgb_model.predict_proba(X)[:, 1],
            "CatBoost": self.cat_model.predict_proba(X)[:, 1],
        }
        probabilities["Ensemble"] = sum(
            self.WEIGHTS[name] * probabilities[name] for name in self.WEIGHTS
        )
        return probabilities

    def predict_proba(self, X):
        return self.predict_individual(X)["Ensemble"]

    def save(self, checkpoint_dir: str | Path) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.xgb_model, checkpoint_dir / "xgb_model.joblib")
        joblib.dump(self.lgb_model, checkpoint_dir / "lgb_model.joblib")
        joblib.dump(self.cat_model, checkpoint_dir / "cat_model.joblib")
        joblib.dump(self.feature_names, checkpoint_dir / "feature_names.joblib")

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

        tuned_path = Path(
            cfg.get("optuna", {}).get(
                "best_params_path", "models/optuna/best_params.json"
            )
        )
        tuned = {}
        if tuned_path.exists():
            with tuned_path.open() as f:
                tuned = json.load(f)

        seed = cfg["data"].get("random_seed", cfg["data"].get("random_state", 42))
        models_cfg = cfg.get("models", {})
        xgb_cfg = models_cfg.get("xgboost", {})
        lgb_cfg = models_cfg.get("lightgbm", {})
        cat_cfg = models_cfg.get("catboost", {})

        xgb_params = {
            key: value
            for key, value in xgb_cfg.items()
            if key not in {"enabled", "random_state", "eval_metric"}
        }
        xgb_params.update(tuned.get("XGBoost", {}).get("params", {}))
        self.xgb_model = XGBClassifier(
            **xgb_params,
            eval_metric=xgb_cfg.get("eval_metric", "aucpr"),
            tree_method=xgb_cfg.get("tree_method", "hist"),
            random_state=seed,
        )

        lgb_params = {
            key: value
            for key, value in lgb_cfg.items()
            if key not in {"enabled", "random_state"}
        }
        lgb_params.update(tuned.get("LightGBM", {}).get("params", {}))
        self.lgb_model = LGBMClassifier(
            **lgb_params,
            is_unbalance=lgb_cfg.get("is_unbalance", True),
            verbose=-1,
            random_state=seed,
        )

        cat_params = {
            key: value
            for key, value in cat_cfg.items()
            if key not in {"enabled", "random_seed", "eval_metric"}
        }
        cat_params.update(tuned.get("CatBoost", {}).get("params", {}))
        self.cat_model = CatBoostClassifier(
            **cat_params,
            auto_class_weights=cat_cfg.get("auto_class_weights", "Balanced"),
            loss_function="Logloss",
            eval_metric=cat_cfg.get("eval_metric", "AUC"),
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

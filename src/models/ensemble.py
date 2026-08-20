from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


class FraudEnsemble:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.feature_names = None

        xgb_cfg = cfg["xgboost"]
        lgb_cfg = cfg["lightgbm"]

        self.xgb_model = XGBClassifier(
            n_estimators=xgb_cfg["n_estimators"],
            max_depth=xgb_cfg["max_depth"],
            learning_rate=xgb_cfg["learning_rate"],
            subsample=xgb_cfg["subsample"],
            colsample_bytree=xgb_cfg["colsample_bytree"],
            scale_pos_weight=xgb_cfg["scale_pos_weight"],
            eval_metric=xgb_cfg["eval_metric"],
            tree_method=xgb_cfg["tree_method"],
            random_state=cfg["data"]["random_seed"],
        )

        self.lgb_model = LGBMClassifier(
            n_estimators=lgb_cfg["n_estimators"],
            max_depth=lgb_cfg["max_depth"],
            num_leaves=lgb_cfg["num_leaves"],
            learning_rate=lgb_cfg["learning_rate"],
            subsample=lgb_cfg["subsample"],
            colsample_bytree=lgb_cfg["colsample_bytree"],
            is_unbalance=lgb_cfg["is_unbalance"],
            verbose=-1,
            random_state=cfg["data"]["random_seed"],
        )

        self.cat_model = CatBoostClassifier(
            iterations=500,
            depth=7,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            auto_class_weights="Balanced",
            random_seed=cfg["data"]["random_seed"],
            verbose=False,
        )

    def fit(self, X_train, y_train, X_val, y_val):
        self.feature_names = list(X_train.columns)

        self.xgb_model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        self.lgb_model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
        )

        self.cat_model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            verbose=False,
        )

        return self

    def predict_individual(self, X):
        return {
            "XGBoost": self.xgb_model.predict_proba(X)[:, 1],
            "LightGBM": self.lgb_model.predict_proba(X)[:, 1],
            "CatBoost": self.cat_model.predict_proba(X)[:, 1],
        }

    def predict_proba(self, X):
        predictions = self.predict_individual(X)

        return (
            0.35 * predictions["XGBoost"]
            + 0.35 * predictions["LightGBM"]
            + 0.30 * predictions["CatBoost"]
        )

    def save(self, checkpoint_dir: str | Path):
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            self.xgb_model,
            checkpoint_dir / "xgb_model.joblib",
        )

        joblib.dump(
            self.lgb_model,
            checkpoint_dir / "lgb_model.joblib",
        )

        joblib.dump(
            self.cat_model,
            checkpoint_dir / "cat_model.joblib",
        )

        joblib.dump(
            self.feature_names,
            checkpoint_dir / "feature_names.joblib",
        )

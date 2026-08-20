from pathlib import Path
import json
import joblib

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


class FraudEnsemble:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.feature_names = None

        tuned_path = Path("models/optuna/best_params.json")

        tuned = {}

        if tuned_path.exists():
            with open(tuned_path, "r") as f:
                tuned = json.load(f)

        xgb_cfg = cfg["xgboost"]
        lgb_cfg = cfg["lightgbm"]

        xgb_params = tuned.get("XGBoost", {}).get("params", {})
        lgb_params = tuned.get("LightGBM", {}).get("params", {})
        cat_params = tuned.get("CatBoost", {}).get("params", {})

        self.xgb_model = XGBClassifier(
            **xgb_params,
            eval_metric=xgb_cfg["eval_metric"],
            tree_method=xgb_cfg["tree_method"],
            random_state=cfg["data"]["random_seed"],
        )

        self.lgb_model = LGBMClassifier(
            **lgb_params,
            is_unbalance=True,
            verbose=-1,
            random_state=cfg["data"]["random_seed"],
        )

        self.cat_model = CatBoostClassifier(
            **cat_params,
            auto_class_weights="Balanced",
            loss_function="Logloss",
            eval_metric="AUC",
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
            verbose=False,
        )

        return self

    def predict_individual(self, X):
        xgb_prob = self.xgb_model.predict_proba(X)[:, 1]
        lgb_prob = self.lgb_model.predict_proba(X)[:, 1]
        cat_prob = self.cat_model.predict_proba(X)[:, 1]

        ensemble_prob = (
            0.35 * xgb_prob
            + 0.35 * lgb_prob
            + 0.30 * cat_prob
        )

        return {
            "XGBoost": xgb_prob,
            "LightGBM": lgb_prob,
            "CatBoost": cat_prob,
            "Ensemble": ensemble_prob,
        }

    def predict_proba(self, X):
        return self.predict_individual(X)["Ensemble"]

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

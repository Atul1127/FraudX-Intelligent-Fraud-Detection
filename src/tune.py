from __future__ import annotations

import json
import pickle
from pathlib import Path

import optuna
import yaml
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


TRIALS = 5


def load_data(cfg):
    proc = Path(cfg["data"]["processed_dir"])

    with open(proc / "features_train.pkl", "rb") as f:
        X_train, y_train = pickle.load(f)

    with open(proc / "features_val.pkl", "rb") as f:
        X_val, y_val = pickle.load(f)

    return X_train, y_train, X_val, y_val


def tune_xgb(X_train, y_train, X_val, y_val):
    def objective(trial):
        model = XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 250, 450),
            max_depth=trial.suggest_int("max_depth", 5, 8),
            learning_rate=trial.suggest_float(
                "learning_rate", 0.03, 0.10
            ),
            subsample=trial.suggest_float(
                "subsample", 0.8, 1.0
            ),
            colsample_bytree=trial.suggest_float(
                "colsample_bytree", 0.8, 1.0
            ),
            scale_pos_weight=trial.suggest_float(
                "scale_pos_weight", 3.0, 8.0
            ),
            eval_metric="auc",
            tree_method="hist",
            random_state=42,
        )

        model.fit(X_train, y_train, verbose=False)

        return average_precision_score(
            y_val,
            model.predict_proba(X_val)[:, 1],
        )

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=TRIALS)

    return study


def tune_lgb(X_train, y_train, X_val, y_val):
    def objective(trial):
        model = LGBMClassifier(
            n_estimators=trial.suggest_int("n_estimators", 250, 450),
            max_depth=trial.suggest_int("max_depth", 5, 10),
            num_leaves=trial.suggest_int("num_leaves", 30, 80),
            learning_rate=trial.suggest_float(
                "learning_rate", 0.03, 0.10
            ),
            subsample=trial.suggest_float(
                "subsample", 0.8, 1.0
            ),
            colsample_bytree=trial.suggest_float(
                "colsample_bytree", 0.8, 1.0
            ),
            is_unbalance=True,
            verbose=-1,
            random_state=42,
        )

        model.fit(X_train, y_train)

        return average_precision_score(
            y_val,
            model.predict_proba(X_val)[:, 1],
        )

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=TRIALS)

    return study


def tune_cat(X_train, y_train, X_val, y_val):
    def objective(trial):
        model = CatBoostClassifier(
            iterations=trial.suggest_int(
                "iterations", 250, 450
            ),
            depth=trial.suggest_int("depth", 6, 9),
            learning_rate=trial.suggest_float(
                "learning_rate", 0.03, 0.10
            ),
            l2_leaf_reg=trial.suggest_float(
                "l2_leaf_reg", 2.0, 8.0
            ),
            auto_class_weights="Balanced",
            eval_metric="AUC",
            random_seed=42,
            verbose=False,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            verbose=False,
        )

        return average_precision_score(
            y_val,
            model.predict_proba(X_val)[:, 1],
        )

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=TRIALS)

    return study


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    X_train, y_train, X_val, y_val = load_data(cfg)

    print(f"Train: {len(X_train):,}")
    print(f"Validation: {len(X_val):,}")
    print(f"Trials per model: {TRIALS}")

    results = {}

    print("\nTuning XGBoost...")
    xgb = tune_xgb(
        X_train, y_train, X_val, y_val
    )
    results["XGBoost"] = {
        "pr_auc": xgb.best_value,
        "params": xgb.best_params,
    }

    print("\nTuning LightGBM...")
    lgb = tune_lgb(
        X_train, y_train, X_val, y_val
    )
    results["LightGBM"] = {
        "pr_auc": lgb.best_value,
        "params": lgb.best_params,
    }

    print("\nTuning CatBoost...")
    cat = tune_cat(
        X_train, y_train, X_val, y_val
    )
    results["CatBoost"] = {
        "pr_auc": cat.best_value,
        "params": cat.best_params,
    }

    output = Path("models/optuna")
    output.mkdir(parents=True, exist_ok=True)

    with open(output / "best_params.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nBest Results:")
    for name, result in results.items():
        print(
            f"{name}: PR-AUC = "
            f"{result['pr_auc']:.4f}"
        )

    print("\nSaved:")
    print("models/optuna/best_params.json")


if __name__ == "__main__":
    main()

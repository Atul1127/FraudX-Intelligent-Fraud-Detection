from __future__ import annotations

import argparse
import json
from pathlib import Path

import optuna
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


def load_data(cfg):
    proc = Path(cfg["data"]["processed_dir"])

    with open(proc / "features_train.pkl", "rb") as f:
        import pickle
        X_train, y_train = pickle.load(f)

    with open(proc / "features_val.pkl", "rb") as f:
        import pickle
        X_val, y_val = pickle.load(f)

    return X_train, y_train, X_val, y_val


def get_metrics(y, prob):
    thresholds = [i / 100 for i in range(20, 81)]

    best_f1 = 0.0
    best_threshold = 0.5

    for threshold in thresholds:
        pred = (prob >= threshold).astype(int)
        score = f1_score(y, pred)

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    return {
        "auc_roc": roc_auc_score(y, prob),
        "auc_pr": average_precision_score(y, prob),
        "f1": best_f1,
        "threshold": best_threshold,
    }


def tune_xgb(X_train, y_train, X_val, y_val, trials):
    def objective(trial):
        model = XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 300, 800),
            max_depth=trial.suggest_int("max_depth", 4, 9),
            learning_rate=trial.suggest_float(
                "learning_rate", 0.02, 0.12, log=True
            ),
            subsample=trial.suggest_float("subsample", 0.7, 1.0),
            colsample_bytree=trial.suggest_float(
                "colsample_bytree", 0.6, 1.0
            ),
            min_child_weight=trial.suggest_int(
                "min_child_weight", 1, 10
            ),
            gamma=trial.suggest_float("gamma", 0.0, 2.0),
            scale_pos_weight=trial.suggest_float(
                "scale_pos_weight", 3.0, 15.0
            ),
            eval_metric="auc",
            tree_method="hist",
            random_state=42,
        )

        model.fit(X_train, y_train, verbose=False)

        prob = model.predict_proba(X_val)[:, 1]

        return average_precision_score(y_val, prob)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials)

    return study


def tune_lgb(X_train, y_train, X_val, y_val, trials):
    def objective(trial):
        model = LGBMClassifier(
            n_estimators=trial.suggest_int("n_estimators", 300, 800),
            max_depth=trial.suggest_int("max_depth", 4, 12),
            num_leaves=trial.suggest_int("num_leaves", 20, 100),
            learning_rate=trial.suggest_float(
                "learning_rate", 0.02, 0.12, log=True
            ),
            subsample=trial.suggest_float("subsample", 0.7, 1.0),
            colsample_bytree=trial.suggest_float(
                "colsample_bytree", 0.6, 1.0
            ),
            min_child_samples=trial.suggest_int(
                "min_child_samples", 10, 100
            ),
            reg_lambda=trial.suggest_float(
                "reg_lambda", 0.0, 5.0
            ),
            is_unbalance=True,
            verbose=-1,
            random_state=42,
        )

        model.fit(X_train, y_train)

        prob = model.predict_proba(X_val)[:, 1]

        return average_precision_score(y_val, prob)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials)

    return study


def tune_cat(X_train, y_train, X_val, y_val, trials):
    def objective(trial):
        model = CatBoostClassifier(
            iterations=trial.suggest_int("iterations", 300, 800),
            depth=trial.suggest_int("depth", 5, 10),
            learning_rate=trial.suggest_float(
                "learning_rate", 0.02, 0.12, log=True
            ),
            l2_leaf_reg=trial.suggest_float(
                "l2_leaf_reg", 1.0, 10.0
            ),
            random_strength=trial.suggest_float(
                "random_strength", 0.0, 2.0
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

        prob = model.predict_proba(X_val)[:, 1]

        return average_precision_score(y_val, prob)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials)

    return study


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=15)
    args = parser.parse_args()

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    X_train, y_train, X_val, y_val = load_data(cfg)

    print(f"Training rows: {len(X_train):,}")
    print(f"Validation rows: {len(X_val):,}")
    print(f"Optuna trials per model: {args.trials}")

    results = {}

    print("\nTuning XGBoost...")
    xgb_study = tune_xgb(
        X_train, y_train, X_val, y_val, args.trials
    )
    results["XGBoost"] = xgb_study.best_params
    print("Best XGBoost PR-AUC:", xgb_study.best_value)

    print("\nTuning LightGBM...")
    lgb_study = tune_lgb(
        X_train, y_train, X_val, y_val, args.trials
    )
    results["LightGBM"] = lgb_study.best_params
    print("Best LightGBM PR-AUC:", lgb_study.best_value)

    print("\nTuning CatBoost...")
    cat_study = tune_cat(
        X_train, y_train, X_val, y_val, args.trials
    )
    results["CatBoost"] = cat_study.best_params
    print("Best CatBoost PR-AUC:", cat_study.best_value)

    output = Path("models/optuna")
    output.mkdir(parents=True, exist_ok=True)

    with open(output / "best_params.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nBest parameters saved to:")
    print(output / "best_params.json")


if __name__ == "__main__":
    main()

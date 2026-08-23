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


def temporal_inner_split(X_train, y_train, validation_size: float):
    if not 0 < validation_size < 1:
        raise ValueError("inner validation size must be between 0 and 1")
    split = int(len(X_train) * (1 - validation_size))
    if split <= 0 or split >= len(X_train):
        raise ValueError("inner validation split must leave train and validation rows")
    return (
        X_train.iloc[:split],
        X_train.iloc[split:],
        y_train.iloc[:split],
        y_train.iloc[split:],
    )


def tune_xgb(X_train, y_train, X_val, y_val, trials: int):
    def objective(trial):
        model = XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 250, 450),
            max_depth=trial.suggest_int("max_depth", 5, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.03, 0.10),
            subsample=trial.suggest_float("subsample", 0.8, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.8, 1.0),
            scale_pos_weight=trial.suggest_float("scale_pos_weight", 3.0, 8.0),
            eval_metric="aucpr",
            tree_method="hist",
            random_state=42,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return average_precision_score(y_val, model.predict_proba(X_val)[:, 1])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials)
    return study


def tune_lgb(X_train, y_train, X_val, y_val, trials: int):
    def objective(trial):
        model = LGBMClassifier(
            n_estimators=trial.suggest_int("n_estimators", 250, 450),
            max_depth=trial.suggest_int("max_depth", 5, 10),
            num_leaves=trial.suggest_int("num_leaves", 30, 80),
            learning_rate=trial.suggest_float("learning_rate", 0.03, 0.10),
            subsample=trial.suggest_float("subsample", 0.8, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.8, 1.0),
            is_unbalance=True,
            verbose=-1,
            random_state=42,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        return average_precision_score(y_val, model.predict_proba(X_val)[:, 1])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials)
    return study


def tune_cat(X_train, y_train, X_val, y_val, trials: int):
    def objective(trial):
        model = CatBoostClassifier(
            iterations=trial.suggest_int("iterations", 250, 450),
            depth=trial.suggest_int("depth", 6, 9),
            learning_rate=trial.suggest_float("learning_rate", 0.03, 0.10),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 2.0, 8.0),
            auto_class_weights="Balanced",
            eval_metric="AUC",
            random_seed=42,
            verbose=False,
        )
        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        return average_precision_score(y_val, model.predict_proba(X_val)[:, 1])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials)
    return study


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    X_train_full, y_train_full, X_final_val, _ = load_data(cfg)
    optuna_cfg = cfg.get("optuna", {})
    trials = int(optuna_cfg.get("n_trials", TRIALS))
    inner_validation_size = float(optuna_cfg.get("inner_validation_size", 0.2))

    # Keep the repository's final chronological validation set untouched.
    X_train, X_inner_val, y_train, y_inner_val = temporal_inner_split(
        X_train_full, y_train_full, inner_validation_size
    )

    print(f"Development train: {len(X_train):,}")
    print(f"Inner validation: {len(X_inner_val):,}")
    print(f"Final untouched validation: {len(X_final_val):,}")
    print(f"Trials per model: {trials}")

    results = {}

    print("\nTuning XGBoost...")
    xgb = tune_xgb(X_train, y_train, X_inner_val, y_inner_val, trials)
    results["XGBoost"] = {"pr_auc": xgb.best_value, "params": xgb.best_params}

    print("\nTuning LightGBM...")
    lgb = tune_lgb(X_train, y_train, X_inner_val, y_inner_val, trials)
    results["LightGBM"] = {"pr_auc": lgb.best_value, "params": lgb.best_params}

    print("\nTuning CatBoost...")
    cat = tune_cat(X_train, y_train, X_inner_val, y_inner_val, trials)
    results["CatBoost"] = {"pr_auc": cat.best_value, "params": cat.best_params}

    output = Path("models/optuna")
    output.mkdir(parents=True, exist_ok=True)

    with open(output / "best_params.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nBest Results:")
    for name, result in results.items():
        print(f"{name}: inner PR-AUC = {result['pr_auc']:.4f}")

    print("\nSaved:")
    print("models/optuna/best_params.json")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import yaml
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from xgboost import XGBClassifier


def load_data(cfg):
    proc = Path(cfg["data"]["processed_dir"])

    with open(proc / "features_train.pkl", "rb") as f:
        X_train, y_train = pickle.load(f)

    with open(proc / "features_val.pkl", "rb") as f:
        X_val, y_val = pickle.load(f)

    return X_train, y_train, X_val, y_val


def build_models(params):
    return (
        XGBClassifier(
            **params["XGBoost"]["params"],
            eval_metric="auc",
            tree_method="hist",
            random_state=42,
        ),
        LGBMClassifier(
            **params["LightGBM"]["params"],
            is_unbalance=True,
            verbose=-1,
            random_state=42,
        ),
        CatBoostClassifier(
            **params["CatBoost"]["params"],
            auto_class_weights="Balanced",
            loss_function="Logloss",
            random_seed=42,
            verbose=False,
        ),
    )


def best_f1(y_true, probability):
    best = (0.0, 0.5)

    for threshold in np.arange(0.20, 0.81, 0.01):
        pred = (probability >= threshold).astype(int)
        score = f1_score(y_true, pred)

        if score > best[0]:
            best = (score, threshold)

    return best


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    X_train, y_train, X_val, y_val = load_data(cfg)

    with open("models/optuna/best_params.json") as f:
        params = json.load(f)

    # Chronological split for meta-training
    split = int(len(X_train) * 0.80)

    X_base = X_train.iloc[:split]
    y_base = y_train.iloc[:split]

    X_meta = X_train.iloc[split:]
    y_meta = y_train.iloc[split:]

    print(f"Base training: {len(X_base):,}")
    print(f"Meta training: {len(X_meta):,}")
    print(f"Final validation: {len(X_val):,}")

    # Train base models on earlier transactions
    models = build_models(params)

    print("\nTraining base models...")

    base_predictions = []
    meta_predictions = []
    final_predictions = []

    for name, model in zip(
        ["XGBoost", "LightGBM", "CatBoost"],
        models,
    ):
        print(f"  {name}...")

        model.fit(X_base, y_base)

        meta_predictions.append(
            model.predict_proba(X_meta)[:, 1]
        )

        # Refit on all available training data
        model.fit(X_train, y_train)

        final_predictions.append(
            model.predict_proba(X_val)[:, 1]
        )

    meta_X = np.column_stack(meta_predictions)
    final_X = np.column_stack(final_predictions)

    # Meta learner
    meta_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    )

    meta_model.fit(meta_X, y_meta)

    stacked_probability = meta_model.predict_proba(final_X)[:, 1]

    auc_roc = roc_auc_score(
        y_val,
        stacked_probability,
    )

    auc_pr = average_precision_score(
        y_val,
        stacked_probability,
    )

    f1, threshold = best_f1(
        y_val,
        stacked_probability,
    )

    print("\nStacking Results:")
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"AUC-PR:  {auc_pr:.4f}")
    print(f"F1:      {f1:.4f}")
    print(f"Threshold: {threshold:.3f}")

    output = Path("models/stacking")
    output.mkdir(parents=True, exist_ok=True)

    with open(output / "stacking_metrics.json", "w") as f:
        json.dump(
            {
                "auc_roc": auc_roc,
                "auc_pr": auc_pr,
                "f1": f1,
                "threshold": threshold,
                "meta_weights": meta_model.coef_[0].tolist(),
            },
            f,
            indent=2,
        )

    print("\nSaved:")
    print("models/stacking/stacking_metrics.json")


if __name__ == "__main__":
    main()

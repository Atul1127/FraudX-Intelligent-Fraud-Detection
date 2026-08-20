from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

from src.models.ensemble import FraudEnsemble
from src.evaluate import find_best_threshold


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    proc = Path(cfg["data"]["processed_dir"])

    with open(proc / "features_train.pkl", "rb") as f:
        X_train_old, y_train_old = pickle.load(f)

    with open(proc / "features_val.pkl", "rb") as f:
        X_val_old, y_val_old = pickle.load(f)

    X = pd.concat(
        [X_train_old, X_val_old],
        axis=0,
        ignore_index=True,
    )

    y = pd.concat(
        [y_train_old, y_val_old],
        axis=0,
        ignore_index=True,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("Random validation benchmark")
    print(f"Train: {len(X_train):,}")
    print(f"Validation: {len(X_val):,}")

    model = FraudEnsemble(cfg)

    model.fit(
        X_train,
        y_train,
        X_val,
        y_val,
    )

    probabilities = model.predict_individual(X_val)

    print("\nResults:")

    for name, probability in probabilities.items():
        auc_roc = roc_auc_score(
            y_val,
            probability,
        )

        auc_pr = average_precision_score(
            y_val,
            probability,
        )

        threshold, best_f1 = find_best_threshold(
            y_val.values,
            probability,
        )

        print(
            f"{name:10s} | "
            f"ROC-AUC: {auc_roc:.4f} | "
            f"PR-AUC: {auc_pr:.4f} | "
            f"F1: {best_f1:.4f} | "
            f"Threshold: {threshold:.3f}"
        )


if __name__ == "__main__":
    main()

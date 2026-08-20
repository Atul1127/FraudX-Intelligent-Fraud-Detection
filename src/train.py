from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.models.ensemble import FraudEnsemble
from src.evaluate import compute_metrics, find_best_threshold


class Trainer:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.model = FraudEnsemble(cfg)

    def run(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> dict:

        self.model.fit(
            X_train,
            y_train,
            X_val,
            y_val,
        )

        print("\nEvaluating models on validation set...")

        predictions = self.model.predict_individual(X_val)

        comparison = {}

        for name, model_proba in predictions.items():
            threshold, best_f1 = find_best_threshold(
                y_val.values,
                model_proba,
            )

            metrics = compute_metrics(
                y_val.values,
                model_proba,
                threshold=threshold,
            )

            comparison[name] = {
                **metrics,
                "best_threshold": float(threshold),
                "best_f1": float(best_f1),
            }

            print(
                f"{name:10s} | "
                f"AUC-ROC: {metrics['auc_roc']:.4f} | "
                f"AUC-PR: {metrics['auc_pr']:.4f} | "
                f"F1: {metrics['f1']:.4f}"
            )

        ensemble_proba = self.model.predict_proba(X_val)

        best_threshold, best_f1 = find_best_threshold(
            y_val.values,
            ensemble_proba,
        )

        metrics = compute_metrics(
            y_val.values,
            ensemble_proba,
            threshold=best_threshold,
        )

        print("\nFinal Ensemble:")
        print(f"  AUC-ROC:       {metrics['auc_roc']:.4f}")
        print(f"  AUC-PR:        {metrics['auc_pr']:.4f}")
        print(f"  Best threshold: {best_threshold:.3f}")
        print(f"  Precision:     {metrics['precision']:.4f}")
        print(f"  Recall:        {metrics['recall']:.4f}")
        print(f"  F1:            {metrics['f1']:.4f}")

        self.cfg["ensemble"]["default_threshold"] = float(
            best_threshold
        )

        feat_imp = dict(
            zip(
                self.model.feature_names,
                self.model.xgb_model.feature_importances_.tolist(),
            )
        )

        top_features = sorted(
            feat_imp.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:25]

        report = {
            **metrics,
            "best_threshold": float(best_threshold),
            "top_features": top_features,
            "model_comparison": comparison,
        }

        return report

    def save(
        self,
        checkpoint_dir: str | Path,
        report: dict,
    ) -> None:

        checkpoint_dir = Path(checkpoint_dir)

        self.model.save(checkpoint_dir)

        with open(
            checkpoint_dir / "training_report.json",
            "w",
        ) as f:
            json.dump(
                report,
                f,
                indent=2,
            )

        print(
            f"Training report saved to "
            f"{checkpoint_dir / 'training_report.json'}"
        )

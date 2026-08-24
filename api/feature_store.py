from __future__ import annotations

from typing import Any

import pandas as pd

from src.data.features import build_features


class OnlineFeatureStore:
    """Build the same feature representation used by FraudX from MongoDB history."""

    def __init__(self, mongo, cfg: dict[str, Any]) -> None:
        self.mongo = mongo
        self.cfg = cfg

    def build(self, transaction: dict[str, Any], model) -> pd.DataFrame:
        if not model.category_mappings:
            raise RuntimeError(
                "Model preprocessing metadata is missing. Retrain FraudX so "
                "models/checkpoints/feature_metadata.joblib is created."
            )

        required = ["TransactionDT", "TransactionAmt"]
        missing = [field for field in required if transaction.get(field) is None]
        if missing:
            raise ValueError(f"Missing required transaction fields: {missing}")

        velocity_cfg = self.cfg.get("features", {}).get("velocity", {})
        frequency_cfg = self.cfg.get("features", {}).get("frequency", {})
        windows = velocity_cfg.get("windows", [3600, 86400, 604800])
        max_window = max(windows) if windows else 604800
        frequency_columns = frequency_cfg.get("columns", [])

        history = self.mongo.get_history(
            transaction,
            frequency_columns=frequency_columns,
            max_window=max_window,
        )

        current = dict(transaction)
        current["__fraudx_current"] = True
        rows = []
        for item in history:
            item = dict(item)
            item["__fraudx_current"] = False
            rows.append(item)
        rows.append(current)

        frame = pd.DataFrame(rows)
        features = build_features(
            frame,
            self.cfg,
            category_mappings=model.category_mappings,
        )

        if "__fraudx_current" not in features.columns:
            raise RuntimeError("Online feature builder lost current-row marker")

        current_mask = features["__fraudx_current"].astype(bool)
        current_features = features.loc[current_mask].copy()
        if len(current_features) != 1:
            raise RuntimeError("Online feature builder produced an invalid current-row count")

        current_features = current_features.drop(columns=["__fraudx_current"])
        return current_features.reindex(columns=model.feature_names)

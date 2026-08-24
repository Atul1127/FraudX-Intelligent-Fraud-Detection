from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from pymongo import MongoClient
from pymongo.collection import Collection


class MongoStore:
    """Small persistence layer for transactions, predictions and audit records."""

    def __init__(self) -> None:
        self.uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        self.database_name = os.getenv("MONGODB_DATABASE", "fraudx")
        self.client: MongoClient | None = None
        self.db = None

    def connect(self) -> bool:
        try:
            self.client = MongoClient(self.uri, serverSelectionTimeoutMS=2000)
            self.client.admin.command("ping")
            self.db = self.client[self.database_name]
            self._ensure_indexes()
            return True
        except Exception:
            self.close()
            return False

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
        self.client = None
        self.db = None

    @property
    def connected(self) -> bool:
        return self.db is not None

    def _collection(self, name: str) -> Collection:
        if self.db is None:
            raise RuntimeError("MongoDB is not connected")
        return self.db[name]

    def _ensure_indexes(self) -> None:
        self._collection("transactions").create_index("transaction_id", unique=True)
        self._collection("predictions").create_index("transaction_id")
        self._collection("predictions").create_index("created_at")
        self._collection("audit_logs").create_index("created_at")

    def save_transaction(self, transaction_id: str, data: dict[str, Any]) -> None:
        self._collection("transactions").update_one(
            {"transaction_id": transaction_id},
            {
                "$set": {
                    "transaction_id": transaction_id,
                    "data": data,
                    "updated_at": datetime.now(timezone.utc),
                },
                "$setOnInsert": {"created_at": datetime.now(timezone.utc)},
            },
            upsert=True,
        )

    def save_prediction(
        self,
        transaction_id: str,
        probability: float,
        prediction: int,
        threshold: float,
        model_version: str,
    ) -> None:
        self._collection("predictions").insert_one(
            {
                "transaction_id": transaction_id,
                "fraud_probability": probability,
                "prediction": prediction,
                "threshold": threshold,
                "model_version": model_version,
                "created_at": datetime.now(timezone.utc),
            }
        )

    def save_audit(self, action: str, details: dict[str, Any]) -> None:
        self._collection("audit_logs").insert_one(
            {
                "action": action,
                "details": details,
                "created_at": datetime.now(timezone.utc),
            }
        )

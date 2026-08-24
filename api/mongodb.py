from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any

from pymongo import MongoClient
from pymongo.collection import Collection


class MongoStore:
    """Persistence and historical transaction access for online FraudX serving."""

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
        transactions = self._collection("transactions")
        transactions.create_index("transaction_id", unique=True)
        transactions.create_index("data.TransactionDT")
        transactions.create_index("created_at")
        for field in [
            "card1", "card2", "card3", "card5", "addr1", "addr2",
            "P_emaildomain", "R_emaildomain",
        ]:
            transactions.create_index(f"data.{field}")
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

    def get_history(
        self,
        transaction: dict[str, Any],
        frequency_columns: list[str],
        max_window: int,
    ) -> list[dict[str, Any]]:
        collection = self._collection("transactions")
        transaction_dt = transaction.get("TransactionDT")
        if transaction_dt is None:
            raise ValueError("TransactionDT is required for online feature construction")
        clauses: list[dict[str, Any]] = []
        for field in frequency_columns:
            value = transaction.get(field)
            if value is not None:
                clauses.append({f"data.{field}": value})
        time_filter = {
            "data.TransactionDT": {
                "$lt": transaction_dt,
                "$gte": max(0, transaction_dt - max_window),
            }
        }
        query: dict[str, Any] = {"$and": [time_filter, {"$or": clauses}]} if clauses else time_filter
        documents = collection.find(query, {"_id": 0, "data": 1}).sort("data.TransactionDT", 1)
        return [doc["data"] for doc in documents]

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

    @staticmethod
    def _window_bounds(window_hours: int) -> tuple[datetime, datetime, datetime]:
        if window_hours < 1:
            raise ValueError("window_hours must be at least 1")
        now = datetime.now(timezone.utc)
        current_start = now - timedelta(hours=window_hours)
        previous_start = current_start - timedelta(hours=window_hours)
        return previous_start, current_start, now

    def get_prediction_monitoring(self, window_hours: int = 24) -> dict[str, Any]:
        """Return current/previous prediction-window statistics for monitoring."""
        previous_start, current_start, now = self._window_bounds(window_hours)
        collection = self._collection("predictions")

        def read(start: datetime, end: datetime) -> list[dict[str, Any]]:
            return list(collection.find(
                {"created_at": {"$gte": start, "$lt": end}},
                {"_id": 0, "fraud_probability": 1, "prediction": 1},
            ))

        def stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
            probabilities = [float(row["fraud_probability"]) for row in rows]
            predictions = [int(row["prediction"]) for row in rows]
            return {
                "count": len(rows),
                "avg_probability": sum(probabilities) / len(probabilities) if probabilities else None,
                "fraud_rate": sum(predictions) / len(predictions) if predictions else None,
                "probabilities": probabilities,
            }

        return {"current": stats(read(current_start, now)), "previous": stats(read(previous_start, current_start))}

    def get_feature_monitoring(
        self,
        window_hours: int = 24,
        fields: tuple[str, ...] = ("TransactionAmt", "TransactionDT"),
    ) -> dict[str, Any]:
        """Return numeric feature samples for current vs previous windows."""
        previous_start, current_start, now = self._window_bounds(window_hours)
        collection = self._collection("transactions")

        def read(start: datetime, end: datetime) -> list[dict[str, Any]]:
            return list(collection.find(
                {"created_at": {"$gte": start, "$lt": end}},
                {"_id": 0, "data": 1},
            ))

        def values(rows: list[dict[str, Any]], field: str) -> list[float]:
            output: list[float] = []
            for row in rows:
                value = row.get("data", {}).get(field)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    output.append(float(value))
            return output

        current_rows = read(current_start, now)
        previous_rows = read(previous_start, current_start)
        return {
            "current": {field: values(current_rows, field) for field in fields},
            "previous": {field: values(previous_rows, field) for field in fields},
        }

    def save_audit(self, action: str, details: dict[str, Any]) -> None:
        self._collection("audit_logs").insert_one(
            {"action": action, "details": details, "created_at": datetime.now(timezone.utc)}
        )

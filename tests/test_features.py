import numpy as np
import pandas as pd

from src.data.features import (
    add_frequency_encodings,
    add_hourly_txn_count,
    add_time_since_last_txn,
    build_features,
)


def test_frequency_encoding_uses_only_previous_rows():
    df = pd.DataFrame(
        {
            "TransactionDT": [1, 2, 3],
            "card1": [10, 10, 10],
        }
    )

    result = add_frequency_encodings(df, ["card1"])

    assert result["card1_freq"].tolist() == [0, 1, 2]


def test_time_since_last_transaction_is_chronological():
    df = pd.DataFrame(
        {
            "TransactionDT": [30, 10, 20],
            "card1": [1, 1, 1],
        }
    )

    result = add_time_since_last_txn(df)

    assert result["TransactionDT"].tolist() == [10, 20, 30]
    assert result["time_since_last_txn"].tolist() == [0, 10, 10]


def test_hourly_count_is_available_without_future_lookup():
    df = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "TransactionDT": [0, 3600, 7200],
            "card1": [100, 100, 100],
            "hour": [0, 1, 2],
        }
    )

    result = add_hourly_txn_count(df)

    assert result["card1_hour_count"].tolist() == [1, 1, 1]


def test_build_features_returns_numeric_columns():
    cfg = {
        "features": {
            "time_windows": [3600],
            "freq_encode_cols": ["card1"],
        }
    }

    df = pd.DataFrame(
        {
            "TransactionID": [1, 2],
            "TransactionDT": [100, 200],
            "TransactionAmt": [10.0, 25.0],
            "card1": [100, 100],
            "addr1": [1, 1],
            "P_emaildomain": ["gmail.com", "gmail.com"],
            "R_emaildomain": ["gmail.com", "yahoo.com"],
            "M1": ["T", "F"],
            "V1": [1.0, np.nan],
        }
    )

    result = build_features(df, cfg)

    assert "TransactionID" not in result.columns
    assert result.select_dtypes(exclude=np.number).empty
    assert "email_mismatch" in result.columns

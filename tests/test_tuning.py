import pandas as pd
import pytest

from src.tune import temporal_inner_split


def test_temporal_inner_split_preserves_order_and_separation():
    X = pd.DataFrame({"TransactionDT": [10, 20, 30, 40, 50]})
    y = pd.Series([0, 1, 0, 1, 0])

    X_train, X_val, y_train, y_val = temporal_inner_split(X, y, 0.4)

    assert X_train["TransactionDT"].tolist() == [10, 20, 30]
    assert X_val["TransactionDT"].tolist() == [40, 50]
    assert y_train.tolist() == [0, 1, 0]
    assert y_val.tolist() == [1, 0]
    assert X_train["TransactionDT"].max() < X_val["TransactionDT"].min()


def test_temporal_inner_split_rejects_invalid_size():
    X = pd.DataFrame({"x": [1, 2, 3]})
    y = pd.Series([0, 1, 0])

    with pytest.raises(ValueError):
        temporal_inner_split(X, y, 0.0)

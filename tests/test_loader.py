import pandas as pd

from src.data.loader import train_val_split


def test_train_val_split_is_chronological():
    cfg = {
        "features": {"target_col": "isFraud"},
        "data": {"test_size": 0.4},
    }

    df = pd.DataFrame(
        {
            "TransactionDT": [50, 10, 40, 20, 30],
            "feature": [5, 1, 4, 2, 3],
            "isFraud": [1, 0, 0, 1, 0],
        }
    )

    X_train, X_val, y_train, y_val = train_val_split(df, cfg)

    assert X_train["TransactionDT"].tolist() == [10, 20, 30]
    assert X_val["TransactionDT"].tolist() == [40, 50]
    assert y_train.tolist() == [0, 1, 0]
    assert y_val.tolist() == [0, 1]

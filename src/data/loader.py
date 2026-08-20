from __future__ import annotations

import os
import pickle
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def load_raw(cfg: dict) -> pd.DataFrame:
    raw_dir = Path(cfg["data"]["raw_dir"])
    txn = pd.read_csv(raw_dir / "train_transaction.csv")
    identity = pd.read_csv(raw_dir / "train_identity.csv")
    df = txn.merge(identity, on="TransactionID", how="left")
    return df


def load_test_raw(cfg: dict) -> pd.DataFrame:
    raw_dir = Path(cfg["data"]["raw_dir"])
    txn = pd.read_csv(raw_dir / "test_transaction.csv")
    identity = pd.read_csv(raw_dir / "test_identity.csv")
    df = txn.merge(identity, on="TransactionID", how="left")
    return df


def train_val_split(
    df: pd.DataFrame, cfg: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

    target = cfg["features"]["target_col"]

    # Fraud detection should be evaluated chronologically.
    # Earlier transactions → training
    # Later transactions → validation
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    split_idx = int(len(df) * (1 - cfg["data"]["test_size"]))

    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]

    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]

    X_val = val_df.drop(columns=[target])
    y_val = val_df[target]

    return X_train, X_val, y_train, y_val


def save_processed(obj: object, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_processed(path: str | Path) -> object:
    with open(path, "rb") as f:
        return pickle.load(f)


def processed_exists(cfg: dict) -> bool:
    proc = Path(cfg["data"]["processed_dir"])
    return (proc / "features_train.pkl").exists() and (proc / "features_val.pkl").exists()

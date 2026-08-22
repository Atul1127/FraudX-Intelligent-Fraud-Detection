from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd


def _require_files(raw_dir: Path, filenames: list[str]) -> None:
    missing = [name for name in filenames if not (raw_dir / name).exists()]
    if missing:
        expected = "\n".join(f"  - {raw_dir / name}" for name in missing)
        raise FileNotFoundError(
            "FraudX raw dataset files are missing.\n"
            "Download the IEEE-CIS Fraud Detection dataset and place these files "
            f"under '{raw_dir}':\n{expected}\n\n"
            "The dataset is intentionally excluded from Git because of its size "
            "and Kaggle distribution restrictions."
        )


def load_raw(cfg: dict) -> pd.DataFrame:
    raw_dir = Path(cfg["data"]["raw_dir"])
    txn_name = cfg["data"].get("train_file", "train_transaction.csv")
    identity_name = cfg["data"].get("train_identity_file", "train_identity.csv")
    _require_files(raw_dir, [txn_name, identity_name])

    txn = pd.read_csv(raw_dir / txn_name)
    identity = pd.read_csv(raw_dir / identity_name)
    return txn.merge(identity, on="TransactionID", how="left")


def load_test_raw(cfg: dict) -> pd.DataFrame:
    raw_dir = Path(cfg["data"]["raw_dir"])
    txn_name = cfg["data"].get("test_file", "test_transaction.csv")
    identity_name = cfg["data"].get("test_identity_file", "test_identity.csv")
    _require_files(raw_dir, [txn_name, identity_name])

    txn = pd.read_csv(raw_dir / txn_name)
    identity = pd.read_csv(raw_dir / identity_name)
    return txn.merge(identity, on="TransactionID", how="left")


def train_val_split(
    df: pd.DataFrame, cfg: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    target = cfg["features"]["target_col"]

    # Fraud detection should be evaluated chronologically.
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

from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE


def add_log_amount(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = (df["TransactionDT"] // 3600) % 24
    df["day_of_week"] = (df["TransactionDT"] // 86400) % 7
    return df


def add_email_mismatch(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    p = df["P_emaildomain"].fillna("") if "P_emaildomain" in df else pd.Series("", index=df.index)
    r = df["R_emaildomain"].fillna("") if "R_emaildomain" in df else pd.Series("", index=df.index)
    df["email_mismatch"] = (p != r).astype(int)
    return df


def add_velocity_features(df: pd.DataFrame, time_windows: list[int]) -> pd.DataFrame:
    """Add causal card-level amount statistics using current/prior transactions."""
    df = df.copy().sort_values("TransactionDT").reset_index(drop=True)
    if "card1" not in df or "TransactionAmt" not in df:
        return df

    for window in time_windows:
        label = f"{window // 3600}h" if window < 86400 else f"{window // 86400}d"
        cols = [
            f"card1_amt_count_{label}",
            f"card1_amt_sum_{label}",
            f"card1_amt_mean_{label}",
            f"card1_amt_std_{label}",
        ]
        result = np.zeros((len(df), 4), dtype=float)

        for _, group in df.groupby("card1", sort=False):
            positions = group.index.to_numpy()
            dt = group["TransactionDT"].to_numpy()
            amt = group["TransactionAmt"].to_numpy()
            for i, t in enumerate(dt):
                mask = (dt[: i + 1] <= t) & (dt[: i + 1] > t - window)
                window_amt = amt[: i + 1][mask]
                result[positions[i], 0] = len(window_amt)
                result[positions[i], 1] = window_amt.sum()
                result[positions[i], 2] = window_amt.mean() if len(window_amt) else 0.0
                result[positions[i], 3] = window_amt.std() if len(window_amt) > 1 else 0.0

        for i, col in enumerate(cols):
            df[col] = result[:, i]

    return df


def add_time_since_last_txn(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("TransactionDT").reset_index(drop=True)
    if "card1" in df and "TransactionDT" in df:
        df["time_since_last_txn"] = df.groupby("card1")["TransactionDT"].diff().fillna(0)
    else:
        df["time_since_last_txn"] = 0.0
    return df


def add_hourly_txn_count(df: pd.DataFrame) -> pd.DataFrame:
    """Historical count for a card/hour combination; excludes the current row."""
    df = df.copy().sort_values("TransactionDT").reset_index(drop=True)
    if not {"card1", "hour"}.issubset(df.columns):
        df["card1_hour_count"] = 0
        return df
    df["card1_hour_count"] = df.groupby(["card1", "hour"], sort=False).cumcount()
    return df


def add_frequency_encodings(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy().sort_values("TransactionDT").reset_index(drop=True)
    for col in cols:
        if col in df.columns:
            df[f"{col}_freq"] = df.groupby(col, sort=False).cumcount()
    return df


def add_combination_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("TransactionDT").reset_index(drop=True)
    if not {"card1", "addr1"}.issubset(df.columns):
        df["card1_addr1_freq"] = 0
        return df
    key = df["card1"].astype(str) + "_" + df["addr1"].fillna("nan").astype(str)
    df["card1_addr1_freq"] = key.groupby(key, sort=False).cumcount()
    return df


def encode_match_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    match_cols = [c for c in df.columns if c.startswith("M") and c[1:].isdigit()]
    for col in match_cols:
        df[col] = df[col].map({"T": 1, "F": 0}).fillna(-1).astype(int)
    return df


def select_v_features(df: pd.DataFrame, keep: int = 50) -> pd.DataFrame:
    """Keep the most populated Vesta features."""
    v_cols = [c for c in df.columns if c.startswith("V")]
    if not v_cols:
        return df
    selected = df[v_cols].isnull().mean().nsmallest(keep).index
    return df.drop(columns=[c for c in v_cols if c not in selected])


def drop_id_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["TransactionID"], errors="ignore")


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    categorical_cols = df.select_dtypes(
        include=["object", "string", "category"]
    ).columns
    for col in categorical_cols:
        df[col] = pd.Categorical(df[col]).codes.astype(float)
        df[col] = df[col].replace(-1, np.nan)
    return df


def build_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Build deterministic, time-ordered features without future-row counts."""
    df = df.copy().sort_values("TransactionDT").reset_index(drop=True)
    df = add_log_amount(df)
    df = add_time_features(df)
    df = add_email_mismatch(df)
    df = add_time_since_last_txn(df)
    df = add_velocity_features(df, cfg["features"]["time_windows"])
    df = add_hourly_txn_count(df)
    df = add_frequency_encodings(df, cfg["features"]["freq_encode_cols"])
    df = add_combination_feature(df)
    df = encode_match_features(df)
    df = select_v_features(df, keep=50)
    df = drop_id_cols(df)
    return encode_categoricals(df)


def apply_smote(
    X_train: pd.DataFrame, y_train: pd.Series, cfg: dict
) -> tuple[pd.DataFrame, pd.Series]:
    sm = SMOTE(
        sampling_strategy=cfg["smote"]["sampling_strategy"],
        k_neighbors=cfg["smote"]["k_neighbors"],
        random_state=cfg["data"]["random_seed"],
    )
    X_filled = X_train.fillna(X_train.median(numeric_only=True))
    X_res, y_res = sm.fit_resample(X_filled, y_train)
    X_res = pd.DataFrame(X_res, columns=X_train.columns)
    y_res = pd.Series(y_res, name=y_train.name)
    print(
        f"  SMOTE: {y_train.sum()} → {y_res.sum()} fraud samples "
        f"({y_train.value_counts().to_dict()} → {y_res.value_counts().to_dict()})"
    )
    return X_res, y_res

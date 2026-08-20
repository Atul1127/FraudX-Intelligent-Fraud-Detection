from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE


# ── Transaction-level features ──────────────────────────────────────────────

def add_log_amount(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # TransactionDT is seconds since a reference datetime (not epoch), but
    # modular arithmetic still gives valid cyclic time features.
    df["hour"] = (df["TransactionDT"] // 3600) % 24
    df["day_of_week"] = (df["TransactionDT"] // 86400) % 7
    return df


def add_email_mismatch(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    p = df["P_emaildomain"].fillna("") if "P_emaildomain" in df.columns else ""
    r = df["R_emaildomain"].fillna("") if "R_emaildomain" in df.columns else ""
    df["email_mismatch"] = (p != r).astype(int)
    return df


# ── Velocity / time-window aggregations ─────────────────────────────────────

def add_velocity_features(df: pd.DataFrame, time_windows: list[int]) -> pd.DataFrame:
    """Rolling aggregations of TransactionAmt per card1 within time windows."""
    df = df.copy().sort_values("TransactionDT").reset_index(drop=True)

    for window in time_windows:
        label = f"{window // 3600}h" if window < 86400 else f"{window // 86400}d"
        grp = df.groupby("card1")

        # For each row, count/sum/mean/std of TransactionAmt by card1
        # within the past `window` seconds using a merge-asof approach.
        records: dict[str, list] = {
            f"card1_amt_count_{label}": [],
            f"card1_amt_sum_{label}": [],
            f"card1_amt_mean_{label}": [],
            f"card1_amt_std_{label}": [],
        }

        for _, group in grp:
            dt = group["TransactionDT"].values
            amt = group["TransactionAmt"].values
            counts, sums, means, stds = [], [], [], []
            for i, t in enumerate(dt):
                mask = (dt <= t) & (dt > t - window)
                window_amt = amt[mask]
                counts.append(len(window_amt))
                sums.append(window_amt.sum())
                means.append(window_amt.mean() if len(window_amt) > 0 else 0.0)
                stds.append(window_amt.std() if len(window_amt) > 1 else 0.0)
            records[f"card1_amt_count_{label}"].extend(counts)
            records[f"card1_amt_sum_{label}"].extend(sums)
            records[f"card1_amt_mean_{label}"].extend(means)
            records[f"card1_amt_std_{label}"].extend(stds)

        for col, vals in records.items():
            tmp = pd.Series(vals, index=df.index)
            df[col] = tmp

    return df


def add_time_since_last_txn(df: pd.DataFrame) -> pd.DataFrame:
    """Seconds since this card1 last appeared."""
    df = df.copy().sort_values("TransactionDT")
    df["time_since_last_txn"] = (
        df.groupby("card1")["TransactionDT"].diff().fillna(0)
    )
    return df


def add_hourly_txn_count(df: pd.DataFrame) -> pd.DataFrame:
    """How many times card1 transacted in this particular hour-of-day historically."""
    df = df.copy()
    if "card1" not in df.columns or "hour" not in df.columns or "TransactionID" not in df.columns:
        df["card1_hour_count"] = 1
        return df
    df["card1_hour_count"] = df.groupby(["card1", "hour"])["TransactionID"].transform("count")
    return df


# ── Frequency / count encodings ──────────────────────────────────────────────

def add_frequency_encodings(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy().sort_values("TransactionDT").reset_index(drop=True)

    for col in cols:
        if col in df.columns:
            # Historical frequency only — never use future transactions
            df[f"{col}_freq"] = (
                df.groupby(col).cumcount()
            )

    return df


def add_combination_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("TransactionDT").reset_index(drop=True)

    if "card1" not in df.columns or "addr1" not in df.columns:
        df["card1_addr1_freq"] = 0
        return df

    key = (
        df["card1"].astype(str)
        + "_"
        + df["addr1"].fillna("nan").astype(str)
    )

    df["card1_addr1_freq"] = key.groupby(key).cumcount()

    return df


# ── Match features (M1-M9) ───────────────────────────────────────────────────

def encode_match_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    match_cols = [c for c in df.columns if c.startswith("M") and c[1:].isdigit()]
    for col in match_cols:
        df[col] = df[col].map({"T": 1, "F": 0}).fillna(-1).astype(int)
    return df


# ── V-feature selection (reduce 339 Vesta features) ──────────────────────────

def select_v_features(df: pd.DataFrame, keep: int = 50) -> pd.DataFrame:
    """Keep V-columns with the lowest null rates (most populated)."""
    v_cols = [c for c in df.columns if c.startswith("V")]
    if not v_cols:
        return df
    null_rates = df[v_cols].isnull().mean().sort_values()
    selected = null_rates.head(keep).index.tolist()
    drop_cols = [c for c in v_cols if c not in selected]
    return df.drop(columns=drop_cols)


# ── Drop low-signal identity columns ─────────────────────────────────────────

def drop_id_cols(df: pd.DataFrame) -> pd.DataFrame:
    drop = ["TransactionID"]
    return df.drop(columns=[c for c in drop if c in df.columns])


# ── Encode categoricals → numeric ────────────────────────────────────────────

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        df[col] = pd.Categorical(df[col]).codes.astype(float)
        df[col] = df[col].replace(-1, np.nan)
    return df


# ── Pipeline ─────────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
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
    df = encode_categoricals(df)
    return df


def apply_smote(
    X_train: pd.DataFrame, y_train: pd.Series, cfg: dict
) -> tuple[pd.DataFrame, pd.Series]:
    sm = SMOTE(
        sampling_strategy=cfg["smote"]["sampling_strategy"],
        k_neighbors=cfg["smote"]["k_neighbors"],
        random_state=cfg["data"]["random_seed"],
    )
    # SMOTE requires no NaNs — fill with column median
    X_filled = X_train.fillna(X_train.median(numeric_only=True))
    X_res, y_res = sm.fit_resample(X_filled, y_train)
    X_res = pd.DataFrame(X_res, columns=X_train.columns)
    y_res = pd.Series(y_res, name=y_train.name)
    print(
        f"  SMOTE: {y_train.sum()} → {y_res.sum()} fraud samples "
        f"({y_train.value_counts().to_dict()} → {y_res.value_counts().to_dict()})"
    )
    return X_res, y_res

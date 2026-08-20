# Fraud Detection Project — Complete Code Deep Dive

**Location:** `C:\Users\abhir\projects\frauddetect`
**Generated:** 2026-07-09

This document walks through every file in the project, explains every function, the algorithms and math behind them, the libraries/tools used, what runs on the actual trained artifacts (this project **has already been trained** — checkpoints and processed data exist on disk), and what happens when each step fails.

> **Scope note:** There is **no LLM** anywhere in this codebase. This is a classical machine learning project — two gradient-boosted decision tree models (XGBoost, LightGBM) combined in a soft-voting ensemble, with SHAP for explainability. If you were expecting an LLM component, it does not exist in this project; everything below is tree-ensemble ML.

## Table of Contents

1. [High-level architecture](#1-high-level-architecture)
2. [Libraries & tools inventory](#2-libraries--tools-inventory)
3. [config.yaml — every setting explained](#3-configyaml--every-setting-explained)
4. [src/data/loader.py — raw data I/O](#4-srcdataloaderpy--raw-data-io)
5. [src/data/features.py — feature engineering](#5-srcdatafeaturespy--feature-engineering-the-algorithmic-core)
6. [SMOTE explained in depth](#6-smote--the-imbalance-handling-algorithm-explained-in-depth)
7. [src/models/ensemble.py — the ensemble](#7-srcmodelsensemblepy--fraudensemble)
8. [Gradient boosting internals](#8-gradient-boosting-internals--what-xgboostlightgbm-actually-compute)
9. [train.py + src/train.py — orchestration](#9-trainpy--srctrainpy--training-orchestration)
10. [src/evaluate.py — metrics & threshold search](#10-srcevaluatepy--metrics-and-threshold-search)
11. [src/explain.py — SHAP explained in depth](#11-srcexplainpy--shap-explainability-explained-in-depth)
12. [app/ — the Streamlit dashboard](#12-app--the-streamlit-dashboard-tab-by-tab)
13. [End-to-end flow diagram](#13-end-to-end-controldata-flow-diagram)
14. [Actual results from the trained model](#14-actual-results-from-the-trained-model-on-disk)
15. [Full failure-mode catalogue](#15-full-failure-mode-catalogue)

---

## 1. High-level architecture

```
train_transaction.csv ─┐
                        ├─ merge on TransactionID ─► raw DataFrame (df)
train_identity.csv    ─┘
                                │
                                ▼
                   build_features() [src/data/features.py]
                    (12 sequential transforms, see §5)
                                │
                                ▼
                train_val_split() [80/20 stratified]
                                │
                 ┌──────────────┴──────────────┐
                 ▼                              ▼
          X_train, y_train                X_val, y_val
                 │
                 ▼
          apply_smote() [oversample minority class, TRAIN ONLY]
                 │
                 ▼
          Trainer.run() [src/train.py]
                 │
    ┌────────────┴────────────┐
    ▼                         ▼
XGBClassifier.fit()      LGBMClassifier.fit()
(early stopping on        (early stopping on
 X_val/y_val)              X_val/y_val)
    │                         │
    └────────────┬────────────┘
                  ▼
     FraudEnsemble.predict_proba()
     = (w_xgb·p_xgb + w_lgbm·p_lgbm) / (w_xgb+w_lgbm)
                  │
                  ▼
     find_best_threshold() → threshold maximizing F1
                  │
                  ▼
  models/checkpoints/{xgb.pkl, lgbm.pkl, ensemble_meta.json,
                       training_report.json}
  config.yaml updated with best_threshold
                  │
                  ▼
     streamlit run app/streamlit_app.py
     (loads checkpoints, serves 3-tab interactive dashboard)
```

Two independent entry points:

- `python train.py` → trains and persists the model
- `streamlit run app/streamlit_app.py` → serves predictions/explanations for a trained model already on disk

---

## 2. Libraries & tools inventory

From `requirements.txt`:

```
xgboost>=2.0
lightgbm>=4.0
shap>=0.44
imbalanced-learn>=0.12
scikit-learn>=1.4
pandas>=2.0
numpy>=1.26
streamlit>=1.35
matplotlib>=3.8
seaborn>=0.13
pyyaml>=6.0
joblib>=1.3
```

**xgboost>=2.0** — Gradient-boosted decision tree library written in C++ with a Python binding. Used here via the sklearn-compatible `xgb.XGBClassifier`. `tree_method: hist` (set in config.yaml) makes it build trees using histogram-binned feature values instead of exact greedy splitting — much faster on datasets this size (hundreds of thousands of rows, 169 features) with negligible accuracy loss.

**lightgbm>=4.0** — Microsoft's gradient-boosted tree library. Structurally similar to XGBoost but grows trees leaf-wise (best-first) rather than level-wise, and uses histogram-based splitting by default. Used via `lgb.LGBMClassifier`.

**shap>=0.44** — SHapley Additive exPlanations. Used exclusively via `shap.TreeExplainer`, which implements the exact "Tree SHAP" polynomial-time algorithm for computing Shapley values on decision tree ensembles (see §11).

**imbalanced-learn>=0.12** — Provides `imblearn.over_sampling.SMOTE`, used to synthetically oversample the fraud (minority) class before training (see §6).

**scikit-learn>=1.4** — Used for: `train_test_split` (stratified train/val split), and the entire `sklearn.metrics` suite (ROC-AUC, PR-AUC, precision, recall, F1, confusion matrix, ROC/PR curve point extraction).

**pandas>=2.0 / numpy>=1.26** — DataFrame manipulation and vectorized numeric operations throughout feature engineering.

**streamlit>=1.35** — Turns the plain Python script `app/streamlit_app.py` into a reactive web app. Streamlit re-runs the entire script top-to-bottom on every user interaction (button click, slider drag, file upload); `@st.cache_data` / `@st.cache_resource` decorators in `app/utils.py` are what make this affordable (see §12).

**matplotlib>=3.8 / seaborn>=0.13** — Plotting. `matplotlib.use("Agg")` is set explicitly in `explain.py` and `streamlit_app.py` because there is no GUI backend available in a server/headless context — Agg renders to an in-memory raster buffer instead of a window, which Streamlit then embeds via `st.pyplot`.

**pyyaml>=6.0** — Reads/writes `config.yaml`. Notably, both `train.py` and the dashboard's "Save threshold" button use `yaml.dump(..., default_flow_style=False)` to persist the tuned threshold back into `config.yaml` — the config file is not static, it's a read-write store for the best-known threshold.

**joblib>=1.3** — Serializes the fitted XGBoost/LightGBM model objects to disk (`xgb.pkl`, `lgbm.pkl`). Preferred over raw pickle for scikit-learn-style estimators because it handles large numpy arrays more efficiently.

---

## 3. config.yaml — every setting explained

```yaml
data:
  processed_dir: data/processed
  random_seed: 42
  raw_dir: data/raw
  test_size: 0.2
ensemble:
  default_threshold: 0.4259135130282319
  lgbm_weight: 0.5
  xgb_weight: 0.5
features:
  freq_encode_cols:
  - card1
  - card2
  - addr1
  - P_emaildomain
  - R_emaildomain
  - DeviceType
  - DeviceInfo
  target_col: isFraud
  time_windows:
  - 3600
  - 86400
  - 604800
lightgbm:
  colsample_bytree: 0.8
  early_stopping_rounds: 50
  is_unbalance: true
  learning_rate: 0.05
  max_depth: 8
  metric: auc
  n_estimators: 500
  num_leaves: 63
  subsample: 0.8
  verbose: -1
smote:
  k_neighbors: 5
  sampling_strategy: 0.1
xgboost:
  colsample_bytree: 0.8
  early_stopping_rounds: 50
  eval_metric: auc
  learning_rate: 0.05
  max_depth: 6
  n_estimators: 500
  scale_pos_weight: 9
  subsample: 0.8
  tree_method: hist
```

| Key | Value | Meaning |
|---|---|---|
| `data.processed_dir` | `data/processed` | cache location for engineered features |
| `data.random_seed` | `42` | used by `train_test_split`, SMOTE, both models |
| `data.raw_dir` | `data/raw` | where `train_transaction.csv` / `train_identity.csv` live |
| `data.test_size` | `0.2` | 20% held out as validation set |
| `ensemble.default_threshold` | `0.4259...` | decision boundary on P(fraud); **overwritten** by `train.py` after every run to the F1-optimal value found by `find_best_threshold()` |
| `ensemble.xgb_weight` / `lgbm_weight` | `0.5` / `0.5` | equal soft-vote weighting between the two models |
| `features.freq_encode_cols` | 7 columns | columns that get count-encoded (§5) |
| `features.target_col` | `isFraud` | binary label column name |
| `features.time_windows` | `3600, 86400, 604800` | 1 hour, 1 day, 7 days (seconds) — velocity feature windows |
| `lightgbm.colsample_bytree` | `0.8` | each tree sees a random 80% of features (regularization) |
| `lightgbm.early_stopping_rounds` | `50` | stop if val AUC hasn't improved in 50 rounds |
| `lightgbm.is_unbalance` | `true` | LightGBM auto-computes class weights ~ 1/class_freq |
| `lightgbm.learning_rate` | `0.05` | shrinkage applied to each tree's contribution |
| `lightgbm.max_depth` | `8` | secondary cap on tree depth |
| `lightgbm.num_leaves` | `63` | primary complexity control (leaf-wise growth); 63 = 2⁶−1, roughly matching `max_depth=8` |
| `lightgbm.n_estimators` | `500` | max boosting rounds (ceiling; early stopping usually cuts this short) |
| `smote.k_neighbors` | `5` | SMOTE interpolates each synthetic point from its 5 nearest same-class neighbors (§6) |
| `smote.sampling_strategy` | `0.1` | oversample minority class until it is 10% of the majority class count (not 50/50 — deliberately realistic) |
| `xgboost.max_depth` | `6` | shallower than LightGBM's 8 — XGBoost grows level-wise so depth alone controls tree size (2⁶=64 leaves max, comparable to LightGBM's 63) |
| `xgboost.scale_pos_weight` | `9` | up-weights the fraud class in the loss function; README states actual fraud rate ~3.5%, so 1/0.035≈28 would be the "textbook" value — 9 is a deliberately gentler weighting |
| `xgboost.tree_method` | `hist` | histogram-based split finding (fast, approximate) |

> **Note:** class imbalance is addressed **three separate ways simultaneously** — SMOTE (10% target), XGBoost's `scale_pos_weight=9`, and LightGBM's `is_unbalance=true`. This is a "belt and suspenders" approach; each technique addresses imbalance at a different stage (data level vs. loss-function level), and their combined effect is evaluated empirically via validation AUC/F1, not derived analytically.

---

## 4. src/data/loader.py — raw data I/O

```python
def load_raw(cfg: dict) -> pd.DataFrame:
    raw_dir = Path(cfg["data"]["raw_dir"])
    txn = pd.read_csv(raw_dir / "train_transaction.csv")
    identity = pd.read_csv(raw_dir / "train_identity.csv")
    df = txn.merge(identity, on="TransactionID", how="left")
    return df
```

The IEEE-CIS dataset ships as two separate CSVs — transaction-level fields (amount, card, product, the C/D/M/V feature blocks) and identity-level fields (device, browser, the `id_01..id_38` block) — linked by `TransactionID`. Not every transaction has a matching identity row (identity verification wasn't always collected), so this is a **left** merge: every transaction row survives, and identity columns are `NaN` where there's no match. This is why so much of feature engineering downstream has to be NaN-tolerant.

```python
def load_test_raw(cfg: dict) -> pd.DataFrame:
    ...  # identical pattern for test_transaction.csv / test_identity.csv
```

Defined but **not called anywhere else** in the codebase — there is no inference/submission script that uses the Kaggle test set. This project's scoring path (the Streamlit "Score a Transaction" tab) builds a synthetic one-row DataFrame from user input instead. Dead code relative to the current pipeline, likely kept for future Kaggle-submission use.

```python
def train_val_split(df, cfg):
    target = cfg["features"]["target_col"]
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_seed"], stratify=y,
    )
    return X_train, X_val, y_train, y_val
```

Thin wrapper around scikit-learn's `train_test_split`. `stratify=y` is the important detail: because fraud is ~3.5% of transactions, a plain random split could by chance under/over-represent fraud in the validation set and produce a misleadingly noisy AUC. Stratification forces the ~3.5% fraud ratio to hold in both splits.

```python
def save_processed(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_processed(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def processed_exists(cfg):
    proc = Path(cfg["data"]["processed_dir"])
    return (proc / "features_train.pkl").exists() and (proc / "features_val.pkl").exists()
```

Simple pickle-based caching layer. Feature engineering (§5) is computationally expensive (the velocity-window loop is effectively O(n) per card-group but with heavy per-row Python overhead — see the warning in §5), so `train.py` checks `processed_exists()` first and skips straight to cached `(X_train, y_train)`/`(X_val, y_val)` tuples unless `--force-preprocess` is passed.

> ⚠️ **This cache has no invalidation logic** — if you edit `features.py` or `config.yaml`'s feature settings, the cached pickles will **not** reflect those changes unless you pass `--force-preprocess` or manually delete `data/processed/*.pkl`. This is a real footgun: a silent stale-cache bug.

---

## 5. src/data/features.py — feature engineering (the algorithmic core)

This file is the most "handwritten algorithm" part of the codebase. It runs 12 transforms in a fixed pipeline via `build_features()`. Every function takes and returns a DataFrame and (mostly) makes a defensive `.copy()` first to avoid mutating the caller's frame.

### 5.1 add_log_amount

```python
def add_log_amount(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])
    return df
```

`log1p(x) = log(1+x)`, chosen over plain `log(x)` because `TransactionAmt` can legitimately be very small (fractions of a dollar) and `log1p` is numerically stable near zero (avoids `log(0) = -inf`). Purpose: transaction amounts are heavily right-skewed (a few very large purchases dominate the raw scale); log-transforming compresses that skew so tree splits on this feature aren't dominated by a handful of outliers.

### 5.2 add_time_features

```python
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # TransactionDT is seconds since a reference datetime (not epoch), but
    # modular arithmetic still gives valid cyclic time features.
    df["hour"] = (df["TransactionDT"] // 3600) % 24
    df["day_of_week"] = (df["TransactionDT"] // 86400) % 7
    return df
```

`TransactionDT` in this dataset is **not** a Unix timestamp — it's seconds elapsed since an arbitrary, undisclosed reference point (a known quirk of the IEEE-CIS dataset). The code comment explicitly flags this: modular arithmetic (`// 3600 % 24`) still produces a valid cyclic hour-of-day feature (0–23) and day-of-week-like feature (0–6) even though the absolute calendar date is unrecoverable, because the 24-hour and 7-day periodicities of human transaction behavior are preserved under modular reduction regardless of what the reference epoch actually was.

### 5.3 add_email_mismatch

```python
def add_email_mismatch(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    p = df["P_emaildomain"].fillna("") if "P_emaildomain" in df.columns else ""
    r = df["R_emaildomain"].fillna("") if "R_emaildomain" in df.columns else ""
    df["email_mismatch"] = (p != r).astype(int)
    return df
```

`P_emaildomain` = purchaser's email domain, `R_emaildomain` = recipient's email domain (e.g. for gifting). A binary flag: 1 if they differ. Fraud intuition: legitimate self-purchases usually have matching or blank/blank domains; a mismatch (especially non-blank vs. non-blank different domains) can indicate account takeover or gift-card fraud patterns. NaN is filled to `""` before comparison, so `NaN==NaN` ambiguity collapses safely to `"" == ""` → not a mismatch.

### 5.4 add_velocity_features (the most expensive function)

```python
def add_velocity_features(df: pd.DataFrame, time_windows: list[int]) -> pd.DataFrame:
    """Rolling aggregations of TransactionAmt per card1 within time windows."""
    df = df.copy().sort_values("TransactionDT").reset_index(drop=True)

    for window in time_windows:
        label = f"{window // 3600}h" if window < 86400 else f"{window // 86400}d"
        grp = df.groupby("card1")

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
```

**Algorithm:** for every `card1` (a proxy for "this specific card/account") and for every one of the 3 time windows (1h, 1d, 7d), and for every transaction made by that card, this computes a **trailing (causal, look-back-only)** window: "of all transactions by this same card1 in the `[t-window, t]` interval, what's the count / sum / mean / stddev of `TransactionAmt`?" This is a classic *velocity feature* from fraud detection: fraud rings often burst many transactions on a stolen card in a short window, so "how many times / how much has this card been used in the last hour" is one of the single most predictive fraud signals in practice.

> ⚠️ **Complexity warning:** for each `card1` group of size *m*, this is an O(m²) nested-loop boolean mask over the group's own transaction timestamps. Across ~590k total transactions this is bounded by per-card group sizes (most cards transact few times, so in practice it's far below worst-case O(n²) over the whole dataset), but it's still a Python-level double loop — this is the reason `build_features()` is slow enough that `loader.py`'s processed-feature caching exists at all. Not vectorized; a production rewrite would use `pd.merge_asof` or a sorted two-pointer sliding window per group for O(m log m) or O(m).

Produces 4 columns × 3 windows = **12 velocity features**: `card1_amt_count_1h/1d/7d`, `card1_amt_sum_1h/1d/7d`, `card1_amt_mean_1h/1d/7d`, `card1_amt_std_1h/1d/7d` (std defaults to 0.0 when fewer than 2 transactions are in the window, avoiding NaN from a single-sample stddev).

### 5.5 add_time_since_last_txn

```python
def add_time_since_last_txn(df: pd.DataFrame) -> pd.DataFrame:
    """Seconds since this card1 last appeared."""
    df = df.copy().sort_values("TransactionDT")
    df["time_since_last_txn"] = (
        df.groupby("card1")["TransactionDT"].diff().fillna(0)
    )
    return df
```

Vectorized (uses pandas' built-in `.diff()` per group — no Python loop). For each `card1`, how many seconds since that card's previous transaction. The very first transaction for a card gets 0 (`fillna`), which is a slight semantic wart: 0 seconds-since-last is indistinguishable from "an extremely rapid repeat transaction" (both would signal high-velocity fraud), yet here 0 actually means "first time this card appears in the dataset" (no signal at all). Minor label-noise source the model has to learn around.

### 5.6 add_hourly_txn_count

```python
def add_hourly_txn_count(df: pd.DataFrame) -> pd.DataFrame:
    """How many times card1 transacted in this particular hour-of-day historically."""
    df = df.copy()
    if "card1" not in df.columns or "hour" not in df.columns or "TransactionID" not in df.columns:
        df["card1_hour_count"] = 1
        return df
    df["card1_hour_count"] = df.groupby(["card1", "hour"])["TransactionID"].transform("count")
    return df
```

For each `(card1, hour-of-day)` combination, count how many rows in the **entire dataset** (train+val, since this runs before the split) share that combination.

> ⚠️ This is subtly different from a "historical" count (as the docstring claims) — because `build_features` runs *before* `train_val_split`, this uses the full future+past distribution, i.e. it is **not** a strictly causal feature like the velocity windows are. This is a mild form of information leakage (using future data to describe a "historical" pattern), though its severity is limited since it's an aggregate frequency rather than a direct label leak.

### 5.7 add_frequency_encodings

```python
def add_frequency_encodings(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            freq = df[col].value_counts(dropna=False)
            df[f"{col}_freq"] = df[col].map(freq).fillna(0).astype(int)
    return df
```

Classic "count encoding": replace each categorical value with how often it appears in the dataset (`card1`, `card2`, `addr1`, `P_emaildomain`, `R_emaildomain`, `DeviceType`, `DeviceInfo`). Rare card numbers / addresses / email domains / devices are often correlated with fraud. `dropna=False` means NaN itself is counted as a category — "how many rows have a missing `DeviceInfo`" becomes informative too, since missing device info is itself a fraud signal in this dataset.

### 5.8 add_combination_feature

```python
def add_combination_feature(df: pd.DataFrame) -> pd.DataFrame:
    """card1 + addr1 pair — captures card-issuer / billing-zip combinations."""
    df = df.copy()
    if "card1" not in df.columns or "addr1" not in df.columns:
        df["card1_addr1_freq"] = 0
        return df
    df["card1_addr1"] = (
        df["card1"].astype(str) + "_" + df["addr1"].fillna("nan").astype(str)
    )
    freq = df["card1_addr1"].value_counts(dropna=False)
    df["card1_addr1_freq"] = df["card1_addr1"].map(freq).fillna(0).astype(int)
    df = df.drop(columns=["card1_addr1"])
    return df
```

A 2-way interaction count-encoding: frequency of the specific (card, billing-address) *pair*, not each independently. Captures "this card has never been used with this billing address before" — a stronger fraud signal than either `card1_freq` or `addr1_freq` alone.

### 5.9 encode_match_features (M1–M9)

```python
def encode_match_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    match_cols = [c for c in df.columns if c.startswith("M") and c[1:].isdigit()]
    for col in match_cols:
        df[col] = df[col].map({"T": 1, "F": 0}).fillna(-1).astype(int)
    return df
```

The IEEE-CIS dataset's M1–M9 columns are "match" flags (e.g. does the billing name match the shipping name) encoded as string `"T"`/`"F"`/NaN. Mapped to 1/0/−1. Using −1 (not NaN, not 0) for missing is deliberate: it keeps a **third distinct state** visible to the tree splitter (missingness itself can be informative — e.g. a match check that wasn't even performed), whereas encoding missing as 0 would conflate "verified false" with "not verified at all."

### 5.10 select_v_features

```python
def select_v_features(df: pd.DataFrame, keep: int = 50) -> pd.DataFrame:
    """Keep V-columns with the lowest null rates (most populated)."""
    v_cols = [c for c in df.columns if c.startswith("V")]
    if not v_cols:
        return df
    null_rates = df[v_cols].isnull().mean().sort_values()
    selected = null_rates.head(keep).index.tolist()
    drop_cols = [c for c in v_cols if c not in selected]
    return df.drop(columns=drop_cols)
```

The Vesta-engineered "V" columns (V1–V339 in the raw data) are a large, largely redundant, heavily-null block of proprietary engineered features from the original Kaggle sponsor. Rather than feeding all ~339 into the model, this keeps only the 50 V-columns with the **lowest missing-value rate**, as a crude but cheap proxy for "most reliably measured, probably most useful." This is a heuristic, not a supervised feature-selection method (it ignores correlation with the target entirely) — a null-rate filter, nothing more.

### 5.11 drop_id_cols

```python
def drop_id_cols(df: pd.DataFrame) -> pd.DataFrame:
    drop = ["TransactionID"]
    return df.drop(columns=[c for c in drop if c in df.columns])
```

Removes the raw identifier — a unique ID column would otherwise let a tree model "memorize" individual rows, so it's dropped before modeling.

### 5.12 encode_categoricals

```python
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        df[col] = pd.Categorical(df[col]).codes.astype(float)
        df[col] = df[col].replace(-1, np.nan)
    return df
```

Any remaining string/object columns (`ProductCD`, `card4`, `card6`, `DeviceType`, `DeviceInfo`, etc.) get integer-coded via pandas' `Categorical` codes. `pd.Categorical(...).codes` assigns −1 to NaN by convention, which is then explicitly mapped back to `np.nan` — this matters because XGBoost/LightGBM both natively handle NaN as "missing" with their own learned split-direction logic, whereas leaving it as the arbitrary integer −1 would make the model treat "missing" as just another ordinary category value.

> 🐛 **Latent bug:** this is a raw label/ordinal encoding (arbitrary integer per unique string value), **not** one-hot and **not** target/mean encoding. The specific integer assigned to each category is **not stable across separate calls** because `pd.Categorical(...).codes` depends on which unique values are *present* in whatever slice of data is passed in. A single-row inference request in the Streamlit app re-runs `pd.Categorical(single_row_column).codes`, which assigns code `0` to whatever category is present in that one row — **not** the same code that value received during training. Categorical columns are almost certainly being scored with silently wrong/inconsistent encodings at inference time in the dashboard's Tab 1. The dashboard partially masks this by aligning columns by *name* afterward, but the categorical *values* within those columns are re-coded independently each time, with no persisted mapping. (See §15.)

### 5.13 build_features — the pipeline

```python
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
    X_filled = X_train.fillna(X_train.median(numeric_only=True))
    X_res, y_res = sm.fit_resample(X_filled, y_train)
    X_res = pd.DataFrame(X_res, columns=X_train.columns)
    y_res = pd.Series(y_res, name=y_train.name)
    return X_res, y_res
```

Order matters here in a few subtle ways: time/velocity features are computed **before** the frame is later split into train/val (so, as noted in §5.6, some aggregate features see the full dataset rather than being computed strictly within-split); `select_v_features` runs after `encode_match_features` but M-columns don't start with "V" so no conflict; categorical encoding runs last, after all string-producing helper columns (like the temporary `card1_addr1` in §5.8) have already been created and dropped.

**Result on this project's actual data:** 169 total feature columns are fed to the models (confirmed from `models/checkpoints/ensemble_meta.json`). That count = original transaction/identity columns (minus `TransactionID`, minus 289 dropped V-columns) + 50 kept V-columns + the ~16 new engineered columns from steps 5.1–5.8.

---

## 6. SMOTE — the imbalance-handling algorithm, explained in depth

```python
def apply_smote(X_train, y_train, cfg):
    sm = SMOTE(
        sampling_strategy=cfg["smote"]["sampling_strategy"],  # 0.1
        k_neighbors=cfg["smote"]["k_neighbors"],               # 5
        random_state=cfg["data"]["random_seed"],
    )
    X_filled = X_train.fillna(X_train.median(numeric_only=True))
    X_res, y_res = sm.fit_resample(X_filled, y_train)
    ...
```

**SMOTE** = Synthetic Minority Over-sampling Technique (Chawla et al., 2002).

**Algorithm:** for each minority-class (fraud) sample *x*, find its `k_neighbors` (=5) nearest minority-class neighbors in feature space (Euclidean distance over all 169 columns, unweighted/unscaled). Pick one neighbor *x_nn* at random, generate a synthetic point:

```
x_new = x + λ * (x_nn - x),   where λ ~ Uniform(0, 1)
```

i.e., a random point on the line segment between the two real minority samples. Repeat until the minority class reaches `sampling_strategy` fraction of the majority class count — here 0.1, meaning fraud is oversampled up to **10%** of legitimate transactions (not to 50/50 — a deliberate choice to avoid over-correcting into an unrealistic class balance that would bias the model's probability calibration too far from the true ~3.5% base rate).

**Why `fillna(median)` first:** SMOTE's nearest-neighbor search cannot handle NaN (undefined distance). Median imputation is applied *only* for this step's neighbor search and synthetic interpolation — a crude choice, but standard practice for this algorithm.

**Critical design choice** — SMOTE is applied to `X_train` **only, after the train/val split**. This is important: applying SMOTE before splitting would let synthetic points derived from a validation-set fraud case leak information into the training set (a synthetic neighbor interpolated between two real points can be very close to one that ends up validated against), inflating validation metrics artificially. Doing it post-split, train-only, keeps validation metrics honest.

> **Caveat:** SMOTE here operates on the full 169-dimensional engineered feature space, including one-hot/ordinal-encoded categoricals and count/frequency columns. Interpolating "halfway between" two categorical codes (e.g., `ProductCD` code 2 and code 4 → synthetic code 3) does not correspond to any real category — this is a well-known theoretical weakness of vanilla SMOTE on mixed categorical/continuous data (SMOTENC exists specifically to fix this, but is not used here).

---

## 7. src/models/ensemble.py — FraudEnsemble

```python
class FraudEnsemble:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        xgb_params = {k: v for k, v in cfg["xgboost"].items() if k != "eval_metric"}
        self.xgb_model = xgb.XGBClassifier(
            **xgb_params,
            eval_metric=cfg["xgboost"]["eval_metric"],
            random_state=cfg["data"]["random_seed"],
        )
        lgbm_params = {k: v for k, v in cfg["lightgbm"].items() if k not in ("metric", "early_stopping_rounds")}
        self.lgbm_model = lgb.LGBMClassifier(
            **lgbm_params,
            random_state=cfg["data"]["random_seed"],
        )
        self.xgb_weight: float = cfg["ensemble"]["xgb_weight"]
        self.lgbm_weight: float = cfg["ensemble"]["lgbm_weight"]
        self.feature_names: list[str] = []
```

Both sub-models are constructed directly from the config dict, with a couple of key-name incompatibilities filtered out (XGBoost's `eval_metric` is re-added explicitly; LightGBM's `metric` and `early_stopping_rounds` are handled via `fit()`'s `callbacks=` instead of the constructor).

```python
    def _fill_nan(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.fillna(X.median(numeric_only=True))
```

Used before **every** fit/predict call. Note: XGBoost and LightGBM **both** have native support for missing values (they learn an optimal default split direction for NaN during training) — this codebase deliberately overrides that native capability with median imputation instead. Simpler/more predictable, but discards the models' built-in missingness handling. The median is recomputed fresh from whatever `X` is passed each call — see §15 for the single-row inference edge case this creates.

```python
    def fit(self, X_train, y_train, X_val, y_val):
        self.feature_names = X_train.columns.tolist()
        X_tr = self._fill_nan(X_train)
        X_v = self._fill_nan(X_val)

        print("Training XGBoost...")
        self.xgb_model.fit(
            X_tr, y_train,
            eval_set=[(X_v, y_val)],
            verbose=False,
        )

        print("Training LightGBM...")
        self.lgbm_model.fit(
            X_tr, y_train,
            eval_set=[(X_v, y_val)],
            callbacks=[
                lgb.early_stopping(self.cfg["lightgbm"]["early_stopping_rounds"], verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
```

Both models train independently and are **not** stacked/blended during training — each sees the same `X_train`/`y_train`/`X_val`/`y_val` and optimizes its own loss (XGBoost's default `binary:logistic` / LightGBM's default binary log loss) against its own AUC-tracked early stopping. There is no shared gradient or joint objective; the "ensemble" is purely a post-hoc prediction-time combination (see §8).

```python
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_filled = self._fill_nan(X)
        p_xgb = self.xgb_model.predict_proba(X_filled)[:, 1]
        p_lgbm = self.lgbm_model.predict_proba(X_filled)[:, 1]
        total = self.xgb_weight + self.lgbm_weight
        return (self.xgb_weight * p_xgb + self.lgbm_weight * p_lgbm) / total

    def predict(self, X: pd.DataFrame, threshold: float | None = None) -> np.ndarray:
        if threshold is None:
            threshold = self.cfg["ensemble"]["default_threshold"]
        return (self.predict_proba(X) >= threshold).astype(int)
```

This is **soft voting** (weighted arithmetic mean of predicted probabilities) — the simplest possible ensembling strategy. `[:, 1]` selects the "positive class" (fraud) column from sklearn's `predict_proba`. `predict()` converts probability → binary label using `>=` against the config's persisted best-F1 threshold (0.4259...) unless overridden — not the naive 0.5 cutoff, which matters a lot for an imbalanced problem like fraud (see §10).

```python
    def save(self, checkpoint_dir):
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.xgb_model, checkpoint_dir / "xgb.pkl")
        joblib.dump(self.lgbm_model, checkpoint_dir / "lgbm.pkl")
        meta = {
            "xgb_weight": self.xgb_weight,
            "lgbm_weight": self.lgbm_weight,
            "feature_names": self.feature_names,
            "default_threshold": self.cfg["ensemble"]["default_threshold"],
        }
        with open(checkpoint_dir / "ensemble_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, checkpoint_dir, cfg):
        checkpoint_dir = Path(checkpoint_dir)
        obj = cls.__new__(cls)
        obj.cfg = cfg
        obj.xgb_model = joblib.load(checkpoint_dir / "xgb.pkl")
        obj.lgbm_model = joblib.load(checkpoint_dir / "lgbm.pkl")
        with open(checkpoint_dir / "ensemble_meta.json") as f:
            meta = json.load(f)
        obj.xgb_weight = meta["xgb_weight"]
        obj.lgbm_weight = meta["lgbm_weight"]
        obj.feature_names = meta["feature_names"]
        return obj
```

Persists `xgb_model` and `lgbm_model` via `joblib.dump`/`joblib.load` separately (two `.pkl` files), plus a small `ensemble_meta.json` holding the weights, the trained feature name list (used later to align columns at inference — see §12's Tab 1), and the threshold. `load()` uses `cls.__new__(cls)` to construct an instance **without** calling `__init__` (bypassing model construction/hyperparameter setup entirely, since the already-fitted models are being loaded from disk, not rebuilt).

---

## 8. Gradient boosting internals — what XGBoost/LightGBM actually compute

Both are implementations of **gradient boosted decision trees (GBDT)** for binary classification. The shared core algorithm:

1. Start with a constant baseline prediction (log-odds of the base fraud rate).
2. For each of up to `n_estimators` (500) rounds:
   - Compute the gradient and Hessian of the logistic loss w.r.t. the current ensemble's predictions, for every training row.
   - Fit a new regression tree to predict these gradients (this is what "gradient boosting" means — each new tree corrects the residual error of all previous trees combined).
   - Scale the new tree's output by `learning_rate` (0.05) before adding it to the running prediction — a small learning rate means each tree contributes only a small correction, requiring more trees but generalizing better.
   - Evaluate AUC on the held-out validation set (`eval_set`); if it hasn't improved for `early_stopping_rounds` (50) consecutive rounds, stop early and roll back to the best-scoring iteration.

**Where the two libraries differ:**

- **XGBoost** (`tree_method: hist`, `max_depth: 6`) — grows trees **level-wise** (breadth-first): every leaf at a given depth is split before the next depth is considered, so depth is the main complexity control. Uses histogram-binned feature values to find near-optimal split points in O(features × bins) instead of scanning every sorted unique value — faster and more memory-efficient at the cost of split-point precision.
- **LightGBM** (`num_leaves: 63`, `max_depth: 8`) — grows trees **leaf-wise** (best-first): at each step it splits whichever single leaf yields the largest loss reduction, regardless of depth, which tends to reach lower training loss faster per tree but is more prone to overfitting on noisy data, hence why `num_leaves` (not depth) is its primary regularizer and `max_depth` is a secondary cap.

`colsample_bytree: 0.8` / `subsample: 0.8` (both models) — classic bagging-style regularization: each tree only sees a random 80% of columns and 80% of rows, decorrelating the trees in the ensemble and reducing overfitting variance (on top of, and independent from, gradient boosting's own sequential-correction mechanism).

`scale_pos_weight: 9` (XGBoost) / `is_unbalance: true` (LightGBM) — both modify the **loss function itself** to penalize false negatives (missed fraud) more than false positives, multiplying the gradient/Hessian contribution of positive-class (fraud) rows. This works *in addition to* SMOTE's data-level rebalancing (§6) — the model sees both an already-partially-rebalanced training set (10% fraud after SMOTE) *and* a further loss-weighted penalty on the remaining imbalance.

`eval_metric: auc` / `metric: auc` — AUC-ROC is used for early stopping rather than accuracy or log loss specifically because accuracy is useless on a ~3.5%-positive dataset (predicting "never fraud" gets 96.5% accuracy). AUC-ROC measures ranking quality independent of the classification threshold, which is the right metric to optimize during training since the actual decision threshold is tuned separately, after training, against F1 (§10).

---

## 9. train.py + src/train.py — training orchestration

`train.py` (project root) is the CLI entry point:

```python
def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    from src.data.loader import (
        load_raw, train_val_split, save_processed, load_processed, processed_exists,
    )
    from src.data.features import build_features, apply_smote
    from src.train import Trainer

    proc = Path(cfg["data"]["processed_dir"])
    target = cfg["features"]["target_col"]

    if not args.force_preprocess and processed_exists(cfg):
        print("Loading cached features...")
        X_train, y_train = load_processed(proc / "features_train.pkl")
        X_val, y_val = load_processed(proc / "features_val.pkl")
    else:
        print("Loading raw data...")
        df = load_raw(cfg)
        df_feat = build_features(df, cfg)
        X_train, X_val, y_train, y_val = train_val_split(df_feat, cfg)
        save_processed((X_train, y_train), proc / "features_train.pkl")
        save_processed((X_val, y_val), proc / "features_val.pkl")

    if not args.skip_smote:
        print("Applying SMOTE...")
        X_train, y_train = apply_smote(X_train, y_train, cfg)

    print("\nTraining ensemble...")
    trainer = Trainer(cfg)
    report = trainer.run(X_train, y_train, X_val, y_val)

    ckpt_dir = Path("models/checkpoints")
    trainer.save(ckpt_dir, report)

    # Persist best threshold back to config
    cfg["ensemble"]["default_threshold"] = report["best_threshold"]
    with open(args.config, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
```

Note the local imports (`from src.data.loader import ...` etc.) happen **inside** `main()`, not at module top-level — this defers the (somewhat heavy) xgboost/lightgbm/pandas import cost until after argument parsing, so `python train.py --help` returns instantly rather than waiting on those imports.

> The **last** thing `main()` does is overwrite `config.yaml`'s `ensemble.default_threshold` with whatever this run's F1-optimal threshold turned out to be. This means `config.yaml` **is a training artifact**, not just input — running `train.py` repeatedly (different random seeds, or after code changes) will silently change the threshold value checked into `config.yaml`. If you're tracking `config.yaml` in git, every training run produces a diff on that one field.

`src/train.py` — the `Trainer` class:

```python
class Trainer:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.model = FraudEnsemble(cfg)

    def run(self, X_train, y_train, X_val, y_val) -> dict:
        self.model.fit(X_train, y_train, X_val, y_val)

        proba = self.model.predict_proba(X_val)
        best_threshold, best_f1 = find_best_threshold(y_val.values, proba)
        metrics = compute_metrics(y_val.values, proba, threshold=best_threshold)

        self.cfg["ensemble"]["default_threshold"] = float(best_threshold)

        feat_imp = dict(
            zip(self.model.feature_names, self.model.xgb_model.feature_importances_.tolist())
        )
        top_features = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:25]

        report = {**metrics, "best_threshold": float(best_threshold), "top_features": top_features}
        return report

    def save(self, checkpoint_dir, report) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        self.model.save(checkpoint_dir)
        with open(checkpoint_dir / "training_report.json", "w") as f:
            json.dump(report, f, indent=2)
```

Thin orchestration: fit the ensemble, score the validation set once, derive the F1-optimal threshold from that same validation set (§10), recompute final metrics at that threshold, and extract the top-25 feature importances — but **only** from `xgb_model.feature_importances_` (XGBoost's default "gain"-based importance).

> LightGBM's own feature importances are never used or reported anywhere in this codebase — the "Top Predictive Features" chart in the dashboard (§12, Tab 2) is XGBoost-only, not a true ensemble-level importance measure.

---

## 10. src/evaluate.py — metrics and threshold search

```python
def compute_metrics(y_true, y_proba, threshold=0.5) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    return {
        "auc_roc": float(roc_auc_score(y_true, y_proba)),
        "auc_pr": float(average_precision_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
    }
```

- **AUC-ROC** — area under the True-Positive-Rate vs. False-Positive-Rate curve across all possible thresholds — a threshold-independent measure of how well the model ranks fraud above non-fraud. 0.5 = random, 1.0 = perfect.
- **AUC-PR (average precision)** — area under the Precision-Recall curve — more informative than AUC-ROC on highly imbalanced data like this (~3.5% positive), because ROC's false-positive-rate denominator is dominated by the huge majority class and can look deceptively good even when precision is poor. This is why the README leads with AUC-PR (0.65) alongside AUC-ROC (0.94) — the gap between the two numbers **is** the imbalance signal.
- **precision/recall/f1 at a specific threshold** — `zero_division=0` guards against an undefined-metric warning/NaN when a threshold is so extreme that zero transactions are predicted positive.

```python
def find_best_threshold(y_true, y_proba) -> tuple[float, float]:
    """Return threshold that maximises F1 on the provided data."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = np.where(
        (precision + recall) == 0,
        0.0,
        2 * precision * recall / (precision + recall),
    )
    best_idx = int(np.argmax(f1_scores[:-1]))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])
```

**Algorithm:** `precision_recall_curve` returns precision/recall at *every* distinct probability value in `y_proba` — i.e., it implicitly evaluates every possible threshold that would change at least one prediction. This is exhaustive, not a coarse grid search. F1 (harmonic mean of precision and recall) is computed at each candidate threshold, and the threshold with the single highest F1 is selected via `argmax`.

Why `[:-1]`: `precision_recall_curve` returns arrays where `precision`/`recall` have one **more** element than `thresholds` (sklearn appends a final `(precision=1, recall=0)` point corresponding to an implicit threshold of +infinity) — slicing off the last element aligns `f1_scores` and `thresholds` before calling `argmax`, avoiding an off-by-one bug.

> This is exactly what determines `config.yaml`'s `ensemble.default_threshold` (0.4259...) — it is **not** a hand-tuned or arbitrary value, it's the exact F1-maximizing point on the validation set's precision-recall curve from the most recent training run.

```python
def get_roc_curve(y_true, y_proba): ...
def get_pr_curve(y_true, y_proba): ...
def get_confusion_matrix(y_true, y_proba, threshold): ...
```

Thin wrappers around sklearn's `roc_curve`, `precision_recall_curve`, and `confusion_matrix` — used only by the dashboard for plotting (§12).

---

## 11. src/explain.py — SHAP explainability, explained in depth

**What is SHAP:** SHapley Additive exPlanations, based on Shapley values from cooperative game theory (Lloyd Shapley, 1953). For a single prediction, SHAP answers: "how much did each individual feature contribute to pushing this prediction away from the model's average (base) output?" Formally, for a model *f* and instance *x*, the Shapley value φᵢ of feature *i* is the **average marginal contribution** of feature *i* across all possible orderings (coalitions) of features being "added" one at a time — this guarantees the values sum *exactly* to `(prediction − base_value)` (a property called *local accuracy*), and is the unique attribution method satisfying a specific set of fairness axioms (efficiency, symmetry, dummy, additivity).

```python
def build_explainers(model, X_background: pd.DataFrame) -> dict:
    """Build SHAP TreeExplainers for both sub-models."""
    X_bg = X_background.fillna(X_background.median(numeric_only=True))
    return {
        "xgb": shap.TreeExplainer(model.xgb_model, X_bg),
        "lgbm": shap.TreeExplainer(model.lgbm_model, X_bg),
        "xgb_weight": model.xgb_weight,
        "lgbm_weight": model.lgbm_weight,
    }
```

Naively computing exact Shapley values requires evaluating the model on 2^N feature subsets (N=169 here — computationally impossible: 2¹⁶⁹). `shap.TreeExplainer` instead implements **Tree SHAP** (Lundberg et al., 2018/2020), a *polynomial-time* algorithm — O(T·L·D²) where T=number of trees, L=max leaves per tree, D=max tree depth — that computes the **exact** Shapley values (not an approximation) by exploiting the tree structure itself. This is only tractable because both XGBoost and LightGBM models are literally made of decision trees — `TreeExplainer` would not work on an arbitrary black-box model (that requires the much slower model-agnostic KernelSHAP instead, not used here).

The `X_bg` background dataset (a 200-row sample of the validation set, per `app/utils.py`'s `load_explainers`) establishes the "base value" — the model's expected/average output over that background distribution — against which every individual explanation's feature contributions are measured as deviations.

```python
def explain_transaction(row, explainers, feature_names) -> shap.Explanation:
    """Return a weighted-average SHAP Explanation for a single row."""
    row_filled = row.fillna(row.median(numeric_only=True))

    sv_xgb = explainers["xgb"].shap_values(row_filled)
    sv_lgbm = explainers["lgbm"].shap_values(row_filled)

    # LightGBM returns list[array] for binary; XGBoost returns array
    if isinstance(sv_lgbm, list):
        sv_lgbm = sv_lgbm[1]
    if isinstance(sv_xgb, list):
        sv_xgb = sv_xgb[1]

    total = explainers["xgb_weight"] + explainers["lgbm_weight"]
    sv_combined = (
        explainers["xgb_weight"] * sv_xgb + explainers["lgbm_weight"] * sv_lgbm
    ) / total

    base_xgb = float(explainers["xgb"].expected_value)
    base_lgbm = float(
        explainers["lgbm"].expected_value[1]
        if hasattr(explainers["lgbm"].expected_value, "__len__")
        else explainers["lgbm"].expected_value
    )
    base = (explainers["xgb_weight"] * base_xgb + explainers["lgbm_weight"] * base_lgbm) / total

    return shap.Explanation(
        values=sv_combined[0],
        base_values=base,
        data=row_filled.values[0],
        feature_names=feature_names,
    )
```

Two library API quirks handled explicitly: XGBoost's binary-classifier SHAP values come back as a plain array (contribution to the positive class's log-odds), while older LightGBM SHAP APIs can return a **list** of two arrays (one per class) — the `isinstance(..., list)` checks normalize both to "contribution to P(fraud)" before combining.

> **Mathematical caveat worth knowing:** SHAP values are additive and exact for a *single* model, but the ensemble's actual prediction is `(w_xgb·p_xgb + w_lgbm·p_lgbm)`, a weighted average of two *probabilities*, while each model's raw SHAP values are computed in that model's own internal margin/log-odds space (before the sigmoid). This code averages the two models' SHAP values directly (`w_xgb·sv_xgb + w_lgbm·sv_lgbm`) — a reasonable, standard approximation for explaining an averaged-probability ensemble, but it is **not** a mathematically exact Shapley decomposition of the final ensembled probability (Shapley values don't compose linearly through a sigmoid in general). In practice this approximation is good enough for *directional, relative* feature-importance explanations (what the waterfall plot is used for) — just don't over-interpret the exact numeric magnitude of a single feature's SHAP contribution as rigorously exact for the combined ensemble.

```python
def plot_waterfall(explanation: shap.Explanation, max_display: int = 15) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    fig = plt.gcf()
    plt.tight_layout()
    return fig
```

Renders SHAP's built-in waterfall chart: starts at `base_values` (the average prediction), stacks each feature's signed contribution in descending order of magnitude (top `max_display`=15 shown individually, the rest collapsed into one "N other features" bar), ending exactly at the final predicted value — a visual, additive audit trail of "why did the model say this transaction is X% likely fraud."

```python
def plot_summary(model, X, explainers, max_display=20, plot_type="bar") -> plt.Figure:
    ...  # computes SHAP values for MANY rows (not just one), then shap.summary_plot(...)
```

Dataset-level (global) explanation rather than per-transaction: shows which features have the largest average absolute SHAP impact across many transactions.

> This function is defined but **not actually wired up** anywhere in the Streamlit dashboard (Tab 2 uses XGBoost's built-in `feature_importances_` instead — see §9). `app/components/shap_chart.py` has its own near-duplicate `render_summary`, which is *also* unused by `streamlit_app.py`. Both look like leftover/planned-but-unwired functionality.

---

## 12. app/ — the Streamlit dashboard, tab by tab

### Streamlit's execution model (important context for the whole file)

Streamlit re-executes `streamlit_app.py` from top to bottom on **every** interaction (a slider drag, button click, file upload). Nothing is a persistent server process reacting to individual events the way, say, a Flask route handler would — the entire script body runs again each time. This is why `app/utils.py` wraps expensive operations (`load_model`, `compute_val_probas`, `load_explainers`) in `@st.cache_resource` / `@st.cache_data` — without these decorators, every slider tick on the Tab 3 threshold slider would reload both pickled models and recompute all validation-set predictions from scratch.

### app/utils.py caching helpers

```python
@st.cache_resource
def load_model(checkpoint_dir: str, _cfg_key: str):
    from src.models.ensemble import FraudEnsemble
    cfg = load_config()
    return FraudEnsemble.load(checkpoint_dir, cfg)
```

`@st.cache_resource` is for objects that shouldn't be copied/serialized (like a live model with C++ bindings) — Streamlit caches the object itself (by reference) keyed on the function's arguments. The `_cfg_key` parameter's leading underscore is a Streamlit convention meaning "don't hash this argument for the cache key" (config dicts aren't reliably hashable) — a dummy passed through purely to allow manually invalidating the cache when needed.

```python
@st.cache_data
def compute_val_probas(checkpoint_dir: str, proc_dir: str):
    import numpy as np
    model = load_model(checkpoint_dir, checkpoint_dir)
    X_val, y_val = load_val_data(proc_dir)
    probas = model.predict_proba(X_val)
    return y_val.values, probas
```

`@st.cache_data` is for plain serializable data (numpy arrays here) — Streamlit hashes the function's inputs, and if unchanged, returns a cached *copy* without recomputation. Running `predict_proba` over the entire validation set happens **once** per Streamlit session, not on every threshold-slider tick in Tab 3 — this is the single biggest performance-relevant design detail in the whole dashboard.

```python
@st.cache_resource
def load_explainers(checkpoint_dir: str, proc_dir: str, _cfg_key: str):
    from src.explain import build_explainers
    import pickle
    model = load_model(checkpoint_dir, checkpoint_dir)
    with open(Path(proc_dir) / "features_val.pkl", "rb") as f:
        X_val, _ = pickle.load(f)
    background = X_val.sample(min(200, len(X_val)), random_state=42)
    return build_explainers(model, background)


def checkpoint_ready(checkpoint_dir: str = "models/checkpoints") -> bool:
    ckpt = Path(checkpoint_dir)
    return (ckpt / "xgb.pkl").exists() and (ckpt / "lgbm.pkl").exists()


def processed_data_ready(proc_dir: str = "data/processed") -> bool:
    proc = Path(proc_dir)
    return (proc / "features_val.pkl").exists()
```

`load_explainers` is only computed lazily (called only inside Tab 1's "Score a Transaction" flow) since building two `TreeExplainer`s over a 200-row background is nontrivial work that's wasted if the user never uploads a transaction.

### streamlit_app.py: readiness gates

```python
if not checkpoint_ready(CKPT_DIR):
    st.error("No trained model found. Run `python train.py` first to train the ensemble.")
    st.stop()

if not processed_data_ready(PROC_DIR):
    st.error("Processed validation data not found. Run `python train.py` first.")
    st.stop()
```

`st.stop()` halts script execution immediately, preventing every downstream tab from trying to load nonexistent files and crashing with an unhandled traceback — the user instead sees a clean, actionable error message. On this machine, both checks currently pass (checkpoints and processed data already exist on disk from a prior training run — see §14).

### Tab 1 — "Score a Transaction"

Two input paths:

**(a) CSV upload** — user uploads a single-row CSV matching the raw `train_transaction.csv`+`train_identity.csv` schema:

```python
uploaded = st.file_uploader("Upload transaction CSV", type="csv")

if uploaded is not None:
    try:
        row_df = pd.read_csv(uploaded)
        if len(row_df) > 1:
            st.warning(f"File has {len(row_df)} rows — using only the first row.")
            row_df = row_df.iloc[:1]

        with st.spinner("Engineering features..."):
            from src.data.features import build_features
            row_feat = build_features(row_df, cfg)

            # Align to trained feature names
            for col in model.feature_names:
                if col not in row_feat.columns:
                    row_feat[col] = np.nan
            row_feat = row_feat[model.feature_names]

        fraud_proba = float(model.predict_proba(row_feat)[0])
        threshold = cfg["ensemble"]["default_threshold"]
        is_fraud = fraud_proba >= threshold
        ...
        st.subheader("SHAP Explanation")
        with st.spinner("Computing SHAP values..."):
            explainers = load_explainers(CKPT_DIR, PROC_DIR, CKPT_DIR)
            explanation = explain_transaction(row_feat, explainers, model.feature_names)
            fig = render_waterfall(explanation, max_display=15)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error processing transaction: {e}")
```

1. Warns and truncates if >1 row uploaded (only the first row used).
2. Calls `build_features(row_df, cfg)` — the **same** feature pipeline used during training, run on just this one row.
3. Aligns columns: for every column the trained model expects (`model.feature_names`, loaded from `ensemble_meta.json`) that's missing from this row's engineered features, injects `np.nan`; then reorders/selects exactly those 169 columns. This step is critical because feature engineering run on one row produces very different aggregates than on the full training set (e.g. `card1_freq` on a lone row is always 1; velocity windows on a single row are always count=1) — but it only fixes *missing columns*, not semantically wrong *values* in columns that do exist (see §15's categorical-encoding bug).
4. `model.predict_proba(row_feat)` → fraud probability, compared against `default_threshold` → FRAUD / LEGITIMATE verdict.
5. Builds a SHAP waterfall explanation via `explain_transaction` and renders it with `st.pyplot`.

Wrapped in a broad `try/except Exception as e: st.error(...)` — any failure anywhere in this chain (malformed CSV, missing required raw columns, a SHAP computation error) is caught and surfaced as a red error banner rather than crashing the whole app, but the exception message shown is whatever `str(e)` happens to be (not curated) and there's no logging of the full traceback anywhere the developer could see it later.

**(b) Manual input form** — a minimal fallback:

```python
with st.expander("Manual input (key fields only)"):
    amt = st.number_input("TransactionAmt", min_value=0.0, value=50.0, step=1.0)
    card1 = st.number_input("card1", min_value=0, value=10000, step=1)
    p_email = st.text_input("P_emaildomain", value="gmail.com")
    r_email = st.text_input("R_emaildomain", value="gmail.com")
    dt = st.number_input("TransactionDT", min_value=0, value=86400, step=3600)

    if st.button("Score transaction"):
        row_df = pd.DataFrame([{
            "TransactionAmt": amt, "card1": card1,
            "P_emaildomain": p_email, "R_emaildomain": r_email,
            "TransactionDT": dt, "TransactionID": 0,
        }])
        from src.data.features import build_features
        row_feat = build_features(row_df, cfg)
        for col in model.feature_names:
            if col not in row_feat.columns:
                row_feat[col] = np.nan
        row_feat = row_feat[model.feature_names]

        fraud_proba = float(model.predict_proba(row_feat)[0])
        threshold = cfg["ensemble"]["default_threshold"]
        st.metric("Fraud Probability", f"{fraud_proba:.1%}")
        if fraud_proba >= threshold:
            st.error("FRAUD")
        else:
            st.success("LEGITIMATE")
```

Only 5 raw fields plus a hardcoded `TransactionID: 0`. This produces a DataFrame with almost entirely missing columns relative to the full schema — `build_features` runs on this sparse frame (many of its helper functions guard against missing columns exactly for this reason, e.g. `add_hourly_txn_count`'s `if "card1" not in df.columns` check), and the resulting prediction is scored the same way as (a) but **without** a SHAP explanation (that block only exists in path (a)).

### Tab 2 — "Model Performance"

```python
y_true, y_proba = compute_val_probas(CKPT_DIR, PROC_DIR)
threshold = cfg["ensemble"]["default_threshold"]
metrics = compute_metrics(y_true, y_proba, threshold)
# AUC-ROC, AUC-PR, Precision, Recall as st.metric() tiles
# ROC curve (render_roc_curve) beside a seaborn confusion-matrix heatmap
# bar chart of report["top_features"] (XGBoost gain importance, top 25)
```

Purely retrospective/static view of the model's performance on the same validation split from the most recent training run — does not re-evaluate on any new data.

### Tab 3 — "Threshold Tuning"

```python
threshold = st.slider(
    "Decision Threshold", min_value=0.01, max_value=0.99,
    value=float(cfg["ensemble"]["default_threshold"]), step=0.01,
)
metrics = compute_metrics(y_true, y_proba, threshold)
# live Precision/Recall/F1/Flagged-count metrics
# PR curve with the current threshold marked (render_pr_curve)

if st.button("Save threshold to config.yaml"):
    cfg["ensemble"]["default_threshold"] = float(threshold)
    import yaml
    with open("config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    st.success(f"Threshold {threshold:.2f} saved to config.yaml")
```

Because `y_true`/`y_proba` come from the cached `compute_val_probas` call, dragging this slider is cheap — it's just re-thresholding already-computed probabilities, an O(n) operation, not re-running the models. Saving overwrites `config.yaml` directly from the running Streamlit process — a direct filesystem write triggered by a UI button, which will race/conflict if `train.py` is run again afterward (whichever writes last wins, no locking).

### app/components/shap_chart.py

```python
def render_waterfall(explanation, max_display=15) -> plt.Figure:
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    fig = plt.gcf()
    plt.tight_layout()
    return fig

def render_summary(shap_values, X, max_display=20) -> plt.Figure:
    shap.summary_plot(shap_values, X, max_display=max_display, show=False)
    fig = plt.gcf()
    plt.tight_layout()
    return fig
```

`render_waterfall` is a thin wrapper (duplicated logic vs. `explain.py`'s `plot_waterfall`, but this is the one actually imported by the app for Tab 1). `render_summary` is defined but never called from `streamlit_app.py` — same duplication/dead-code situation as `explain.py`'s `plot_summary` (§11).

### app/components/pr_curve.py

```python
def render_pr_curve(precision, recall, thresholds, current_threshold, auc_pr) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(recall, precision, color="#1f77b4", linewidth=2, label=f"PR curve (AUC={auc_pr:.3f})")
    dists = np.abs(thresholds - current_threshold)
    idx = int(np.argmin(dists))
    ax.scatter(recall[idx], precision[idx], color="#d62728", zorder=5, s=120,
               label=f"Threshold = {current_threshold:.2f}")
    ...
    return fig

def render_roc_curve(fpr, tpr, auc_roc) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fpr, tpr, color="#1f77b4", linewidth=2, label=f"ROC (AUC={auc_roc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ...
    return fig
```

`render_pr_curve` marks the currently-selected threshold by finding the closest *actual* threshold value in the `precision_recall_curve` output via `np.argmin(np.abs(thresholds - current_threshold))` (nearest-neighbor lookup, since the slider's continuous value won't exactly match one of sklearn's returned discrete threshold values). `render_roc_curve` plots a diagonal "Random" reference line (y=x, representing AUC=0.5, a coin-flip classifier) for visual comparison.

---

## 13. End-to-end control/data flow diagram

**Training** (`python train.py`):

```
CSV files → load_raw() → build_features() [12 steps, §5]
  → train_val_split() [stratified 80/20]
  → apply_smote() [train only, §6]
  → Trainer.run()
    → FraudEnsemble.fit() [XGBoost + LightGBM, each with early stopping, §7-8]
    → predict_proba(X_val)
    → find_best_threshold() [exhaustive F1 search, §10]
    → compute_metrics()
    → top-25 XGBoost feature importances
  → save models/checkpoints/*.pkl + ensemble_meta.json + training_report.json
  → rewrite config.yaml's default_threshold
```

**Serving** (`streamlit run app/streamlit_app.py`):

```
readiness checks [checkpoint + processed data exist]
  → load cached model/explainers/val-probas [Streamlit cache decorators]
  → 3 tabs:
      Tab1: user CSV/form → build_features() on 1 row
            → align to trained feature_names → predict_proba
            → threshold compare → SHAP TreeExplainer waterfall
      Tab2: static metrics/curves/confusion-matrix/feature-importance chart
            from the cached validation predictions
      Tab3: interactive threshold re-scoring of the same cached validation
            predictions → optional write-back to config.yaml
```

---

## 14. Actual results from the trained model on disk

This project has **already been trained** at least once — the following are the real numbers currently sitting in `models/checkpoints/training_report.json` (not hypothetical):

| Metric | Value |
|---|---|
| AUC-ROC | 0.9405 |
| AUC-PR | 0.6515 |
| Precision | 0.6677 |
| Recall | 0.5824 |
| F1 | 0.6221 |
| Best threshold | 0.4259 (currently mirrored into `config.yaml`) |

Total engineered feature count actually used by the trained model: **169** (from `ensemble_meta.json`'s `feature_names` list) — includes the raw/passed-through transaction+identity columns, 50 selected V-columns, and all the engineered columns from §5.

Top-5 features by XGBoost gain importance (from `training_report.json`):

| Rank | Feature | Importance |
|---|---|---|
| 1 | `C8` | 0.1230 |
| 2 | `V317` | 0.0657 |
| 3 | `C4` | 0.0504 |
| 4 | `V287` | 0.0420 |
| 5 | `C12` | 0.0387 |

> Notably, **none** of this project's own hand-engineered features (velocity windows, frequency encodings, `email_mismatch`, etc.) crack the top 5 — the raw Vesta-provided C-columns and a few V-columns dominate XGBoost's gain importance, though several engineered features (`M1`, `card4`, C-family) do appear further down the top-25 list. A genuine, mildly humbling finding about this specific dataset: the proprietary Vesta features the competition sponsors already engineered are still doing most of the heavy lifting relative to this project's additions, at least by this one importance measure.

---

## 15. Full failure-mode catalogue

### Data loading (loader.py)

- Missing `data/raw/*.csv` → `pd.read_csv` raises `FileNotFoundError`, uncaught, crashes `train.py` immediately with a Python traceback (no friendly message). The README explicitly warns the user to download and place these files first.
- Malformed/corrupted CSV → pandas raises a `ParserError`, uncaught, same crash-with-traceback behavior.

### Stale cache (loader.py / train.py)

- If `features.py` or `config.yaml`'s feature settings change but `data/processed/*.pkl` already exists, `train.py` **silently** loads the old cached features unless `--force-preprocess` is passed. No hash/fingerprint check exists to detect staleness. This can produce a model trained on a feature schema that no longer matches the current code — a genuinely dangerous silent-failure mode for anyone iterating on `features.py`.

### Feature engineering (features.py)

- `add_velocity_features`'s nested Python loop (§5.4) is the main performance bottleneck; on very large datasets or many small windows it could become impractically slow — no timeout, no progress bar beyond the coarse print statements in `train.py`/`loader.py`.
- `pd.Categorical(...).codes` non-determinism across calls (§5.12) means categorical feature *values* scored by the dashboard's Tab 1 are not guaranteed to match the integer codes the model was actually trained on for the same real-world category — a silent correctness bug that would degrade single-transaction scoring accuracy for any row touching a categorical column. Does not raise an exception; predictions are simply less accurate than they should be, with no visible symptom.
- `add_hourly_txn_count`'s use of full-dataset (not train-only) counts (§5.6) is a mild information-leakage source baked into both the reported validation metrics and any future single-row scoring.

### SMOTE (features.py `apply_smote`, §6)

- If `y_train` has fewer real minority-class samples than `k_neighbors` (5) after the train/val split, `SMOTE.fit_resample` raises a `ValueError` — uncaught, crashes `train.py`. Not currently a risk with the full IEEE-CIS dataset (~20k+ fraud rows even after an 80/20 split), but would surface immediately when training on a small custom/sampled subset.
- `--skip-smote` flag exists precisely so a user can bypass this whole code path (e.g. for debugging or if SMOTE proves harmful/unnecessary on a given data subset).

### Model training (ensemble.py / train.py, §7-9)

- Both XGBoost and LightGBM `fit()` calls have no try/except — any internal error crashes `train.py` with a raw traceback. There is no partial-checkpoint recovery — if LightGBM fails *after* XGBoost already trained successfully, that XGBoost work is lost (nothing is saved until both models finish and `Trainer.save()` is explicitly called at the very end).
- `early_stopping_rounds` on both models means training could stop far short of `n_estimators=500` — by design, not a failure, but the actual number of trees is data-dependent and only discoverable by inspecting the saved model afterward.

### Ensemble NaN handling (ensemble.py `_fill_nan`, §7)

- For a single-row DataFrame with a genuinely missing value in some column, `X.median(numeric_only=True)` of that one row's column is itself NaN (median of a length-1 all-NaN series is NaN) — so `fillna(NaN)` is a no-op and the actual NaN passes straight into `predict_proba`. XGBoost handles NaN inputs gracefully natively, but LightGBM's sklearn wrapper can raise or warn depending on version/config. This is a latent single-row-inference edge case, most likely to surface via the dashboard's manual-input form (Tab 1B) or a CSV upload missing several columns.

### Dashboard (streamlit_app.py, §12)

- `checkpoint_ready()`/`processed_data_ready()` gate the whole app with `st.stop()` if `models/checkpoints` or `data/processed` aren't populated — clean failure, clear user-facing message, no crash.
- Tab 1's CSV-upload path wraps everything in a broad `except Exception as e: st.error(...)` — any failure (a `KeyError` from a missing required raw column, a SHAP internal error, a dtype mismatch) degrades to the same generic-looking red banner with just the raw exception string — no differentiation between "your CSV is malformed" vs. "there's a genuine bug in feature engineering," which makes this failure mode hard for an end user to self-diagnose.
- Tab 1's manual-input form has no explanation/SHAP output at all — if a user only ever uses the manual form, they never see feature attributions, silently.
- Tab 3's "Save threshold to config.yaml" button writes directly to the config file with no locking — if `python train.py` is run concurrently (or afterward), whichever process's write happens last overwrites the other's threshold value with no warning or merge.
- Streamlit's cache decorators key on function *arguments*, not on file modification times — if you retrain the model (overwriting the same checkpoint file paths) while a Streamlit session is already running, the app will keep serving predictions from the **old cached model** until the server process is manually restarted (or Streamlit's cache is manually cleared via its UI) — a real "stale model in production" risk during iterative development.

### Explainability (explain.py, §11)

- If `X_background` passed to `build_explainers` contains a column with all values NaN, `fillna(median)` leaves that column entirely NaN — SHAP's `TreeExplainer` would then receive NaN in its background data, which can produce degenerate/undefined base values for that feature's contribution. Not currently observed (the 200-row validation sample is unlikely to have an entirely-NaN column), but a latent risk if the background sample size were reduced.
- The weighted SHAP-value averaging across two different models' margin spaces (§11's mathematical caveat) is not a "failure" in the crash sense, but is a source of approximation error in the numbers shown to end users — worth knowing if you ever need to defend or audit the exact magnitude of a specific SHAP contribution shown in the dashboard.

---

*End of document.*

# Interpretable Fraud Detection

XGBoost + LightGBM ensemble trained on the [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) dataset, with SHAP explainability and an interactive Streamlit dashboard.

**Validation results:** AUC-ROC 0.94 | AUC-PR 0.65 | F1 0.62 (threshold 0.43)

![Dashboard](https://img.shields.io/badge/Streamlit-dashboard-red) ![Python](https://img.shields.io/badge/Python-3.10%2B-blue)

---

## Features

- **Ensemble model** — soft-voting average of XGBoost and LightGBM, each trained with early stopping on a held-out validation set
- **25+ engineered features** — transaction velocity (1h / 24h / 7d windows per card), time-since-last-transaction, hour/day-of-week, email domain mismatch, count encodings for card/address/email, M1–M9 match flags
- **Class imbalance handling** — SMOTE oversampling applied to training split only (fraud rate: 3.5% → 10%)
- **SHAP explainability** — per-transaction waterfall plots showing which features drove each prediction
- **Interactive dashboard** — real-time transaction scoring, ROC/PR curves, live threshold tuning

---

## Project Structure

```
frauddetect/
├── config.yaml              # All hyperparameters
├── train.py                 # Training entry point
├── requirements.txt
├── src/
│   ├── data/
│   │   ├── loader.py        # Load and merge raw CSVs
│   │   └── features.py      # Feature engineering + SMOTE
│   ├── models/
│   │   └── ensemble.py      # FraudEnsemble class
│   ├── train.py             # Trainer class
│   ├── evaluate.py          # Metrics (AUC-ROC, PR curve, threshold search)
│   └── explain.py           # SHAP TreeExplainer + plots
└── app/
    ├── streamlit_app.py     # Three-tab dashboard
    ├── utils.py             # Streamlit caching helpers
    └── components/
        ├── shap_chart.py    # Waterfall plot component
        └── pr_curve.py      # PR / ROC curve components
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the dataset

The dataset is from the [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) Kaggle competition. You need a Kaggle account and to accept the competition rules before downloading.

```bash
# Using the Kaggle CLI
kaggle competitions download -c ieee-fraud-detection
```

Unzip and place these four files into `data/raw/`:

```
data/raw/
├── train_transaction.csv
├── train_identity.csv
├── test_transaction.csv
└── test_identity.csv
```

### 3. Train

```bash
python train.py
```

Options:
- `--force-preprocess` — recompute features even if a cache exists
- `--skip-smote` — train without SMOTE oversampling

Training takes 10–20 minutes depending on hardware. Outputs:
- `models/checkpoints/xgb.pkl` — trained XGBoost model
- `models/checkpoints/lgbm.pkl` — trained LightGBM model
- `models/checkpoints/training_report.json` — metrics + top features
- `config.yaml` updated with the best F1 threshold

### 4. Launch the dashboard

```bash
streamlit run app/streamlit_app.py
```

---

## Dashboard

**Tab 1 — Score a Transaction**
Upload a CSV row (same schema as `train_transaction.csv` merged with `train_identity.csv`) to get a fraud probability and a SHAP waterfall plot explaining the prediction. A minimal manual input form is also available for quick testing.

**Tab 2 — Model Performance**
ROC curve, confusion matrix, per-class metrics, and a bar chart of the top 25 most important features.

**Tab 3 — Threshold Tuning**
Drag a slider to explore the precision/recall tradeoff at any threshold. The PR curve updates live. Save your chosen threshold back to `config.yaml` with one click.

---

## How It Works

```
train_transaction.csv ─┐
                        ├─ merge ─► feature engineering ─► SMOTE ─► XGBoost ──┐
train_identity.csv    ─┘                                              LightGBM ─┴─► avg ─► probability ─► SHAP
```

Feature engineering builds velocity signals (how fast is this card spending?), frequency encodings (how common is this card/email/address?), time features (hour of day, days since last use), and match flags (do the card details on file match what was submitted?).

SMOTE is applied only to the training split to preserve realistic evaluation metrics on the validation set.

SHAP TreeExplainer computes exact Shapley values for both sub-models; the per-transaction explanation is the weighted average matching the ensemble weights.

---

## Configuration

All hyperparameters live in `config.yaml`. Key settings:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `xgboost.n_estimators` | 500 | Max trees (early stopping may use fewer) |
| `xgboost.scale_pos_weight` | 9 | Up-weights fraud samples (~1/fraud_rate) |
| `lightgbm.is_unbalance` | true | Auto class-weight balancing |
| `smote.sampling_strategy` | 0.1 | Target fraud fraction after oversampling |
| `ensemble.default_threshold` | auto | Set by training to maximise F1 |

---

## Notes on the Dataset

The IEEE-CIS dataset is subject to Kaggle's [competition rules](https://www.kaggle.com/competitions/ieee-fraud-detection/rules). The data files are excluded from this repository via `.gitignore` and must be downloaded separately after accepting those rules.

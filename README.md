# FraudX — Intelligent Fraud Detection

An interpretable fraud detection system built on the IEEE-CIS Fraud Detection dataset, combining temporal validation, leakage-safe feature engineering, SMOTE, gradient-boosting ensembles, Optuna tuning, stacking, threshold optimization, and SHAP explainability.

## Results

### Temporal Validation

The main evaluation uses a chronological split so the model is evaluated on later transactions.

| Model | ROC-AUC | PR-AUC | F1 |
|---|---:|---:|---:|
| XGBoost | 0.8750 | 0.3604 | 0.3958 |
| LightGBM | 0.8780 | 0.4482 | 0.4618 |
| CatBoost | 0.8807 | 0.4612 | 0.4753 |
| Weighted Ensemble | 0.9011 | 0.4946 | 0.4980 |
| **Stacked Ensemble** | **0.9199** | **0.5504** | **0.4834** |

### Random Validation Benchmark

A separate random-split benchmark is included to compare against the conventional validation setup.

| Model | ROC-AUC | PR-AUC | F1 |
|---|---:|---:|---:|
| XGBoost | 0.9587 | 0.7723 | 0.7304 |
| LightGBM | 0.9579 | 0.7588 | 0.7210 |
| CatBoost | 0.9561 | 0.7212 | 0.6856 |
| Weighted Ensemble | **0.9610** | **0.7717** | **0.7290** |

The difference between random and temporal validation highlights the difficulty of predicting future fraud transactions and helps expose overly optimistic validation results.

## Key Features

- **Temporal validation** using chronological transaction ordering
- **Leakage-safe feature engineering**
- Transaction velocity and time-based features
- Frequency and combination encodings
- Email, card, address and device-related signals
- **SMOTE** for training-set class imbalance
- **XGBoost + LightGBM + CatBoost**
- **Optuna hyperparameter optimization**
- Weighted soft-voting ensemble
- **Stacking with Logistic Regression meta-model**
- Automatic fraud threshold optimization
- **SHAP explainability**
- Streamlit dashboard for interactive analysis

## Architecture

```text
IEEE-CIS Dataset
       │
       ▼
Data Loading & Merge
       │
       ▼
Leakage-Safe Feature Engineering
       │
       ▼
Chronological Train / Validation Split
       │
       ▼
SMOTE on Training Data
       │
       ├──────────────┐
       ▼              ▼
   XGBoost        LightGBM
       │              │
       └──────┬───────┘
              │
           CatBoost
              │
              ▼
      Weighted Ensemble
              │
              ▼
       Stacking Meta-Model
              │
              ▼
      Threshold Optimization
              │
              ▼
      Fraud Probability
              │
              ▼
        SHAP Explanation

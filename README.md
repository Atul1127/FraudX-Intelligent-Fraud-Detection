# FraudX — Intelligent Fraud Detection

FraudX is an interpretable fraud-detection system built on the IEEE-CIS Fraud Detection dataset. It combines **time-aware validation, leakage-conscious feature engineering, class-imbalance handling, gradient-boosting ensembles, Optuna tuning, threshold optimization, SHAP explanations, automated tests, and a Streamlit dashboard**.

## Results

### Main temporal evaluation

The primary benchmark uses chronological validation: earlier transactions are used for training and later transactions are held out for validation.

| Model | ROC-AUC | PR-AUC | F1 |
|---|---:|---:|---:|
| XGBoost | 0.8750 | 0.3604 | 0.3958 |
| LightGBM | 0.8780 | 0.4482 | 0.4618 |
| CatBoost | 0.8807 | 0.4612 | 0.4753 |
| **Weighted Ensemble** | **0.9011** | **0.4946** | **0.4980** |

### Stacking experiment

A separate chronological stacking experiment achieved **0.9199 ROC-AUC** and **0.5504 PR-AUC**. Stacking is kept as an explicit experiment rather than silently replacing the default weighted ensemble used by the training pipeline and dashboard.

### Random-split benchmark

| Model | ROC-AUC | PR-AUC | F1 |
|---|---:|---:|---:|
| XGBoost | 0.9587 | 0.7723 | 0.7304 |
| LightGBM | 0.9579 | 0.7588 | 0.7210 |
| CatBoost | 0.9561 | 0.7212 | 0.6856 |
| **Weighted Ensemble** | **0.9610** | **0.7717** | **0.7290** |

The gap between random and temporal validation demonstrates why future-transaction validation is the primary evaluation protocol.

## Architecture

```text
IEEE-CIS Transactions + Identity
              │
              ▼
        Data Loading / Merge
              │
              ▼
     Time-Ordered Feature Engineering
              │
              ├── Time / amount features
              ├── Historical frequency features
              ├── Card / address / email signals
              ├── Transaction velocity
              └── Vesta / match features
              │
              ▼
     Chronological Train / Validation Split
              │
              ▼
       SMOTE on Training Data
              │
       ┌──────┼────────┐
       ▼      ▼        ▼
    XGBoost LightGBM CatBoost
       └──────┼────────┘
              ▼
       Weighted Ensemble
              │
              ▼
      Threshold Optimization
              │
              ▼
       Fraud Probability
              │
       ┌──────┴──────┐
       ▼             ▼
   SHAP Explain    Dashboard
```

## Project Structure

```text
FraudX-Intelligent-Fraud-Detection/
├── app/
│   ├── streamlit_app.py
│   ├── utils.py
│   └── components/
│       ├── pr_curve.py
│       └── shap_chart.py
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   └── features.py
│   ├── models/
│   │   └── ensemble.py
│   ├── evaluate.py
│   ├── explain.py
│   ├── train.py
│   ├── tune.py
│   ├── stacking.py
│   └── benchmark_random.py
├── tests/
│   ├── test_features.py
│   ├── test_loader.py
│   ├── test_evaluate.py
│   └── test_ensemble.py
├── config.yaml
├── pytest.ini
├── requirements.txt
├── train.py
└── README.md
```

Generated datasets and model artifacts are intentionally excluded from Git.

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download IEEE-CIS data

Download the competition files from Kaggle and place them under `data/raw/`:

```text
data/raw/
├── train_transaction.csv
├── train_identity.csv
├── test_transaction.csv
└── test_identity.csv
```

The dataset is not included because of its size and Kaggle distribution restrictions.

### 3. Train the main ensemble

```bash
python train.py
```

The pipeline reuses cached processed features when available and writes model checkpoints plus a training report under `models/checkpoints/`.

### 4. Tune models

```bash
python src/tune.py
```

Optuna optimizes **PR-AUC**, which is more informative than accuracy for highly imbalanced fraud detection. Best parameters are saved to `models/optuna/best_params.json`.

### 5. Run the stacking experiment

```bash
python src/stacking.py
```

Results are written to `models/stacking/stacking_metrics.json`.

### 6. Run the random benchmark

```bash
python -m src.benchmark_random
```

This is a comparison benchmark only; temporal validation remains the primary evaluation protocol.

### 7. Run tests

```bash
pytest -q
```

The test suite covers chronological splitting, historical feature behavior, numeric feature output, evaluation metrics, threshold selection, and the ensemble prediction contract.

### 8. Launch the dashboard

```bash
streamlit run app/streamlit_app.py
```

## Feature Engineering

FraudX generates transaction-level signals including:

- Log-transformed transaction amount
- Hour and day-of-week features
- Time since the previous transaction for a card
- Historical card and combination frequencies
- Card-level transaction velocity over configurable windows
- Card/hour historical activity
- Email-domain mismatch
- M1–M9 match indicators
- Selected Vesta features based on missingness
- Missing-value-aware categorical encoding

Time-dependent count features are calculated in transaction order and do not use future rows. This is important because fraud patterns can drift over time.

## Imbalanced Learning

Fraud transactions are heavily underrepresented. FraudX applies **SMOTE only to the training split**, while validation remains untouched. The boosting models also use class-balancing mechanisms.

```text
Training data → SMOTE → Model fitting
Validation    ─────────→ Evaluation
```

## Model Optimization

The system uses three complementary gradient-boosting models:

- **XGBoost**
- **LightGBM**
- **CatBoost**

Optuna searches a compact hyperparameter space using PR-AUC as the optimization objective. The default ensemble uses weighted soft voting:

```text
XGBoost  ── 35% ──┐
LightGBM ── 35% ──┼──► Fraud Probability
CatBoost ── 30% ──┘
```

The decision threshold is optimized on the validation set for F1 and persisted in the configuration used by the dashboard.

## Stacking Experiment

`src/stacking.py` provides a separate chronological stacking experiment. Base models are trained on earlier transactions, predictions from a later training segment train the logistic-regression meta-model, and the final model is evaluated on the held-out future validation segment.

This keeps the experimental result separate from the default weighted ensemble so the repository's main training path remains reproducible and easy to understand.

## Explainability

SHAP is used for model interpretation, including global feature importance, per-transaction explanations, waterfall plots, and feature contribution analysis.

## Testing

The repository includes lightweight unit tests designed to catch common ML pipeline regressions:

- Temporal ordering and train/validation separation
- Historical frequency and transaction-time features
- Numeric feature output after preprocessing
- Evaluation metric and threshold contracts
- Ensemble probability shape and bounds

Run the suite with:

```bash
pytest -q
```

## Why Temporal Validation?

Fraud behavior changes over time. A random split can place highly related transactions from the same period on both sides of the split and produce optimistic estimates.

FraudX instead evaluates the realistic deployment scenario:

```text
Earlier transactions ─────────► Later transactions
        TRAIN                       VALIDATION
```

The random benchmark is retained only to quantify the difference between conventional and time-aware validation.

## Dashboard

The Streamlit application provides:

- Transaction scoring from uploaded CSV rows or key manual inputs
- Fraud probability and optimized threshold
- SHAP transaction explanations
- ROC and PR curves
- Confusion matrix
- Feature importance
- Interactive threshold analysis

## Technology Stack

Python · Pandas · NumPy · Scikit-learn · XGBoost · LightGBM · CatBoost · Optuna · SHAP · imbalanced-learn · Streamlit · Pytest

## Project Goal

FraudX focuses on building a realistic and interpretable fraud-detection workflow rather than optimizing a single benchmark number. It demonstrates **time-aware ML validation, leakage-conscious feature engineering, imbalanced classification, ensemble learning, hyperparameter optimization, threshold selection, stacking, explainability, testing, and model serving through an interactive dashboard**.

# FraudX — Intelligent Fraud Detection

An end-to-end, interpretable fraud detection system built on the IEEE-CIS Fraud Detection dataset. FraudX combines leakage-aware feature engineering, chronological validation, class-imbalance handling, gradient-boosting ensembles, hyperparameter optimization, stacking, threshold tuning, and SHAP explainability.

## Highlights

- Chronological train/validation split for realistic future-transaction evaluation
- Leakage-aware transaction, identity, frequency, velocity, and temporal features
- SMOTE applied only to training data
- XGBoost, LightGBM, and CatBoost base learners
- Weighted soft-voting ensemble and Logistic Regression stacking
- Optuna optimization targeting PR-AUC
- Threshold optimization for fraud-focused precision/recall trade-offs
- SHAP-based global and per-transaction explanations
- Streamlit dashboard for scoring, evaluation, and threshold analysis

## Results

### Temporal Validation

| Model | ROC-AUC | PR-AUC | F1 |
|---|---:|---:|---:|
| XGBoost | 0.8750 | 0.3604 | 0.3958 |
| LightGBM | 0.8780 | 0.4482 | 0.4618 |
| CatBoost | 0.8807 | 0.4612 | 0.4753 |
| Weighted Ensemble | 0.9011 | 0.4946 | 0.4980 |
| **Stacked Ensemble** | **0.9199** | **0.5504** | 0.4834 |

### Random Validation Benchmark

| Model | ROC-AUC | PR-AUC | F1 |
|---|---:|---:|---:|
| XGBoost | 0.9587 | 0.7723 | 0.7304 |
| LightGBM | 0.9579 | 0.7588 | 0.7210 |
| CatBoost | 0.9561 | 0.7212 | 0.6856 |
| **Weighted Ensemble** | **0.9610** | **0.7717** | **0.7290** |

The gap between random and temporal validation demonstrates why chronological evaluation is important for fraud detection: random splits can produce substantially more optimistic estimates when transaction patterns are time-dependent.

## Architecture

```text
IEEE-CIS Transactions + Identity Data
                │
                ▼
        Data Loading & Merge
                │
                ▼
     Leakage-Aware Feature Engineering
                │
                ▼
      Chronological Train / Validation
                │
                ▼
        SMOTE on Training Only
                │
       ┌────────┼─────────┐
       ▼        ▼         ▼
   XGBoost   LightGBM   CatBoost
       └────────┼─────────┘
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
          ┌─────┴─────┐
          ▼           ▼
      SHAP API     Streamlit UI
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
├── config.yaml
├── requirements.txt
├── train.py
└── README.md
```

Generated datasets, cached features, model checkpoints, tuning outputs, and stacking artifacts are intentionally excluded from version control.

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the dataset

FraudX uses the [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) dataset. Accept the competition rules before downloading.

```bash
kaggle competitions download -c ieee-fraud-detection -p data/raw
```

Extract the files into:

```text
data/raw/
├── train_transaction.csv
├── train_identity.csv
├── test_transaction.csv
└── test_identity.csv
```

The dataset is not included in this repository because of its size and Kaggle distribution restrictions.

### 3. Train the pipeline

```bash
python train.py
```

Cached processed features are reused automatically. Use `--force-preprocess` to rebuild them or `--skip-smote` to disable oversampling.

### 4. Tune models

```bash
python src/tune.py
```

Best parameters are written to `models/optuna/best_params.json`.

### 5. Evaluate stacking

```bash
python src/stacking.py
```

Stacking metrics are written to `models/stacking/stacking_metrics.json`.

### 6. Run the random-split benchmark

```bash
python -m src.benchmark_random
```

This benchmark is intentionally separate from the main temporal evaluation.

### 7. Launch the dashboard

```bash
streamlit run app/streamlit_app.py
```

## Feature Engineering

FraudX derives transaction-level signals from time, cards, addresses, emails, devices, identity fields, and historical transaction behavior. The feature pipeline includes temporal features, velocity signals, frequency/combination encodings, identity matching signals, and missing-value indicators.

Historical frequency features are designed to respect transaction ordering and reduce temporal leakage.

## Imbalanced Learning

Fraud is a minority class, so FraudX applies SMOTE only after the chronological training split. Validation remains untouched to preserve the original class distribution. The boosting models also use appropriate class-balancing controls where configured.

## Model Optimization

Optuna searches compact hyperparameter spaces for XGBoost, LightGBM, and CatBoost. PR-AUC is used as the primary optimization objective because it is more informative than accuracy for highly imbalanced fraud detection.

## Stacking

The temporal pipeline combines base-model predictions with a Logistic Regression meta-model. The stacking workflow is designed so the meta-model learns from earlier model predictions rather than directly consuming final validation labels.

## Explainability

SHAP is used to provide:

- Global feature importance
- Per-transaction feature contributions
- Waterfall-style explanations
- Model-level interpretation for fraud investigations

## Dashboard

The Streamlit interface supports:

- Transaction fraud scoring
- Individual prediction explanations
- Model performance inspection
- Confusion matrix and precision/recall analysis
- PR/ROC evaluation
- Threshold analysis
- Feature-importance exploration

## Technologies

Python · Pandas · NumPy · Scikit-learn · XGBoost · LightGBM · CatBoost · Optuna · SHAP · imbalanced-learn · Streamlit

## Project Goal

FraudX is designed as an ML engineering project rather than a single-model benchmark. The focus is on realistic validation, leakage-aware feature engineering, imbalanced classification, ensemble modeling, explainability, and an interactive inference workflow.

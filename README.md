
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
````

## Project Structure

```text
FraudX-Intelligent-Fraud-Detection/
│
├── app/
│   ├── streamlit_app.py
│   ├── utils.py
│   └── components/
│       ├── shap_chart.py
│       └── pr_curve.py
│
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   └── features.py
│   │
│   ├── models/
│   │   └── ensemble.py
│   │
│   ├── evaluate.py
│   ├── explain.py
│   ├── train.py
│   ├── tune.py
│   ├── stacking.py
│   └── benchmark_random.py
│
├── models/
│   ├── checkpoints/
│   ├── optuna/
│   └── stacking/
│
├── data/
│   └── raw/
│
├── config.yaml
├── requirements.txt
├── train.py
└── README.md
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the dataset

FraudX uses the
[IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection)
dataset.

You need a Kaggle account and must accept the competition rules before downloading.

Using the Kaggle CLI:

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

The dataset is excluded from Git because of its size and distribution restrictions.

### 3. Train

```bash
python train.py
```

Cached features are reused when available.

### 4. Tune models

Optuna tuning is intentionally lightweight:

```bash
python src/tune.py
```

Best parameters are saved to:

```text
models/optuna/best_params.json
```

### 5. Evaluate stacking

```bash
python src/stacking.py
```

Results are saved to:

```text
models/stacking/stacking_metrics.json
```

### 6. Random validation benchmark

```bash
python -m src.benchmark_random
```

This is provided only as a comparison against the main chronological evaluation.

### 7. Launch dashboard

```bash
streamlit run app/streamlit_app.py
```

## Feature Engineering

FraudX creates transaction-level signals including:

* Transaction time and temporal patterns
* Hour and day-of-week features
* Transaction velocity
* Historical frequency encodings
* Card/address combinations
* Email-domain relationships
* Identity and transaction matching signals
* Missing-value indicators

Frequency features are calculated using historical transaction information to reduce temporal leakage.

## Class Imbalance

Fraud represents a small fraction of transactions.

FraudX uses:

```text
Training data
     │
     ▼
SMOTE
     │
     ▼
Balanced training distribution
```

SMOTE is applied only to the training data. Validation data remains untouched so evaluation better reflects the original fraud distribution.

The boosting models also use class-balancing mechanisms where appropriate.

## Model Optimization

Optuna searches a compact hyperparameter space for:

* XGBoost
* LightGBM
* CatBoost

The optimization objective is **PR-AUC**, which is more informative than accuracy for highly imbalanced fraud detection.

The tuned models are then used by the ensemble and stacking pipeline.

## Stacking

The final temporal model uses:

```text
XGBoost ──┐
LightGBM ─┼──► Logistic Regression
CatBoost ─┘       Meta-model
                    │
                    ▼
              Fraud Probability
```

The meta-model is trained on predictions from earlier transactions and evaluated on later transactions.

This prevents the meta-model from directly seeing the final validation labels.

## Explainability

FraudX uses SHAP to explain model predictions.

The system provides:

* Global feature importance
* Per-transaction explanations
* SHAP waterfall plots
* Feature contribution analysis

This makes the model easier to interpret when investigating why a transaction was classified as suspicious.

## Why Temporal Validation?

Random validation can produce optimistic fraud-detection results because transactions from the same underlying time period can appear in both training and validation.

FraudX therefore uses:

```text
Earlier Transactions ─────────► Later Transactions
        TRAIN                       VALIDATION
```

The random benchmark is retained separately for comparison.

This gives two useful perspectives:

* **Random split:** conventional benchmark performance
* **Temporal split:** more realistic future-transaction performance

## Dashboard

The Streamlit dashboard provides:

### Transaction Scoring

Estimate fraud probability for a transaction and view its explanation.

### Model Performance

View:

* ROC-AUC
* PR-AUC
* Confusion matrix
* Feature importance
* Model performance

### Threshold Analysis

Explore the precision/recall trade-off at different fraud thresholds.

## Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* LightGBM
* CatBoost
* Optuna
* SHAP
* imbalanced-learn
* Streamlit

## Dataset

IEEE-CIS Fraud Detection
Kaggle Competition

The dataset is subject to Kaggle's competition rules and is not included in this repository.

## Project Goal

FraudX focuses on building a realistic and interpretable fraud detection pipeline rather than optimizing a single validation score.

The project demonstrates:

* Feature engineering
* Imbalanced classification
* Ensemble learning
* Hyperparameter optimization
* Stacking
* Temporal validation
* Model interpretability
* Threshold optimization

````

Commit it on GitHub as:

```text
Rewrite README for FraudX ML pipeline
````

Then in Bash:

```bash
cd ~/FraudX-Intelligent-Fraud-Detection
git pull origin main
```

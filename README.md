# Fraud Detection & Risk Scoring System

> **End-to-end, time-aware fraud detection with ensemble ML, online historical features, explainability, experiment tracking, API serving, monitoring, Docker, and CI/CD.**

[![CI](https://github.com/Atul1127/Fraud-Detection-Risk-Scoring-System/actions/workflows/ci.yml/badge.svg)](https://github.com/Atul1127/Fraud-Detection-Risk-Scoring-System/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-Online%20Features-47A248?logo=mongodb&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-CI%2FCD-2088FF?logo=githubactions&logoColor=white)

FraudX is a production-style fraud detection platform built on the IEEE-CIS Fraud Detection dataset. It combines **chronological validation, leakage-conscious feature engineering, imbalanced learning, XGBoost/LightGBM/CatBoost ensembles, Optuna tuning, threshold optimization, SHAP explainability, MongoDB-backed historical features, FastAPI inference, MLflow tracking, prediction/feature drift monitoring, Docker Compose, and GitHub Actions CI/CD**.

---

## Recruiter Snapshot

- **590,540 transactions** processed with chronological train/validation evaluation.
- **0.8921 ROC-AUC**, **0.4857 PR-AUC**, and **0.5081 F1** from the final tuned weighted ensemble.
- Ensemble F1 improved by approximately **8.5%** after Optuna tuning (**0.4681 → 0.5081**).
- A separate chronological stacking experiment reached **0.9177 ROC-AUC / 0.5445 PR-AUC**.
- **18 automated tests passing** in the verified local suite.
- Real-time inference through **FastAPI** with MongoDB-backed historical features and prediction persistence.
- **MLflow** tracks model parameters, metrics, reports, and artifacts.
- **PSI-based monitoring** detects prediction-score and online numeric-feature distribution shifts.
- Complete local stack containerized with **Docker Compose** and validated through **GitHub Actions**.

---

# Architecture

```text
                    GitHub Push / PR
                           │
                    GitHub Actions
                    ┌──────┴──────┐
                    │ Tests + CI  │
                    │ Docker Build│
                    └──────┬──────┘
                           │
          ┌────────────────┴────────────────┐
          │                                 │
          ▼                                 ▼
   Training Pipeline                    FastAPI :8001
          │                                 │
          ▼                                 ▼
 Feature Engineering                    MongoDB :27017
          │                                 │
          ▼                                 │
 Chronological Split                       │
          │                                 │
          ▼                                 │
 Training-only SMOTE                       │
          │                                 │
    ┌─────┼─────┐                           │
    ▼     ▼     ▼                           │
   XGB   LGBM  CatBoost                     │
    └─────┼─────┘                           │
          ▼                                 │
  Weighted Ensemble ◄───────────────────────┘
          │
          ▼
 Threshold Optimization
          │
          ▼
 Fraud Probability
          │
     ┌────┴────┐
     ▼         ▼
   SHAP     Persistence
               │
               ▼
        Drift Monitoring
               │
               ▼
      stable / warning / drift

          ┌──────────────┐
          │    MLflow    │
          │ Runs/Metrics │
          │  Artifacts   │
          └──────────────┘
```

### Runtime services

| Service | Purpose | Host Port |
|---|---|---:|
| **FraudX API** | Real-time fraud scoring and monitoring | `8001` |
| **MongoDB** | Historical transactions and prediction persistence | `27017` |
| **MLflow** | Experiment tracking and artifacts | `5000` |

Port `8001` is used intentionally so FraudX can coexist with applications using port `8000`.

---

# Core Capabilities

| Capability | Implementation |
|---|---|
| Fraud classification | XGBoost + LightGBM + CatBoost |
| Time-aware evaluation | Chronological train/validation split |
| Imbalanced learning | SMOTE on training data + class balancing |
| Hyperparameter optimization | Optuna, optimized for PR-AUC |
| Decision optimization | Validation-based F1 threshold selection |
| Explainable AI | SHAP |
| Online historical features | MongoDB |
| Model serving | FastAPI + Uvicorn |
| Experiment tracking | MLflow |
| Drift monitoring | PSI for prediction scores and online numeric features |
| Containerization | Docker + Docker Compose |
| Automated validation | Pytest |
| CI/CD | GitHub Actions |
| Interactive analysis | Streamlit |

---

# Model Performance

## Final tuned temporal evaluation

The primary benchmark uses chronological validation: earlier transactions are used for training and later transactions are held out for validation.

| Model | ROC-AUC | PR-AUC | F1 |
|---|---:|---:|---:|
| XGBoost | 0.8832 | 0.4551 | 0.4692 |
| LightGBM | 0.8491 | 0.3748 | 0.3993 |
| CatBoost | 0.8773 | 0.4624 | 0.4778 |
| **Weighted Ensemble** | **0.8921** | **0.4857** | **0.5081** |

**Best ensemble threshold:** `0.426`  
**Precision:** `0.5725` · **Recall:** `0.4567` · **F1:** `0.5081`

SMOTE is applied only to the training split; the chronological validation period remains untouched.

## Optuna improvement

| Metric | Before tuning | After tuning |
|---|---:|---:|
| ROC-AUC | 0.8888 | **0.8921** |
| PR-AUC | 0.4726 | **0.4857** |
| F1 | 0.4681 | **0.5081** |

## Separate stacking experiment

A chronological stacking experiment is retained separately from the default production path.

| Metric | Stacking |
|---|---:|
| ROC-AUC | **0.9177** |
| PR-AUC | **0.5445** |
| F1 | 0.4875 |
| Threshold | 0.810 |

> Results can vary with library versions, cached artifacts, configuration, and feature changes. The weighted ensemble remains the default serving path.

---

# Machine Learning Pipeline

```text
IEEE-CIS Transactions + Identity
              │
              ▼
       Data Loading / Merge
              │
              ▼
   Time-Ordered Feature Engineering
              │
      ┌───────┼─────────┐
      ▼       ▼         ▼
    Time    History   Velocity
   Amount   Frequency  Signals
      └───────┼─────────┘
              ▼
    Chronological Split
              │
              ├──────────────► Validation (untouched)
              │
              ▼
       Training Data
              │
              ▼
             SMOTE
              │
       ┌──────┼──────┐
       ▼      ▼      ▼
     XGB    LGBM   CatBoost
       └──────┼──────┘
              ▼
      Weighted Probability
              │
              ▼
       Threshold Search
              │
              ▼
      Fraud / Not Fraud
              │
              ▼
        SHAP Explanation
```

## Feature Engineering

FraudX generates transaction-level signals including:

- Log-transformed transaction amount.
- Hour and day-of-week features.
- Time since previous transaction for a card.
- Historical card and combination frequencies.
- Card-level transaction velocity over configurable windows.
- Card/hour historical activity.
- Email-domain mismatch signals.
- M1–M9 match indicators.
- Selected Vesta features based on missingness.
- Missing-value-aware categorical encoding.

Time-dependent count features are calculated in transaction order and do not use future rows, reducing temporal leakage risk.

## Imbalanced Learning

FraudX applies **SMOTE only to the training split**, while validation remains untouched. The boosting models also use class-balancing mechanisms.

## Ensemble

```text
XGBoost  ── 35% ──┐
LightGBM ── 35% ──┼──► Fraud Probability
CatBoost ── 30% ──┘
```

Optuna optimizes PR-AUC and the decision threshold is selected on validation data.

---

# Online Features with MongoDB

MongoDB provides historical context during online inference rather than acting only as a persistence layer.

```text
Incoming transaction
        │
        ▼
     FastAPI
        │
        ▼
 MongoDB history
        │
        ▼
Historical frequency / velocity features
        │
        ▼
Model preprocessing
        │
        ▼
Ensemble prediction
        │
        ▼
Persist transaction + prediction
```

Historical queries use earlier transactions, preserving the temporal nature of the feature pipeline.

---

# FastAPI

## Swagger

After starting Docker Compose:

**http://127.0.0.1:8001/docs**

## Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Service/model/MongoDB health |
| `GET` | `/model/info` | Model version, threshold, weights, feature count |
| `GET` | `/monitoring/predictions` | Prediction volume, fraud rate and score drift |
| `GET` | `/monitoring/features` | Online numeric-feature distribution drift |
| `POST` | `/predict` | Score and persist a transaction |

### Example prediction request

```json
{
  "transaction_id": "TX_TEST_001",
  "data": {
    "TransactionDT": 864000,
    "TransactionAmt": 149.99,
    "ProductCD": "W",
    "card1": 13926,
    "card2": 555,
    "card3": 150,
    "card5": 226,
    "addr1": 315,
    "addr2": 87,
    "P_emaildomain": "gmail.com",
    "R_emaildomain": "gmail.com"
  }
}
```

Example response from the verified local deployment:

```json
{
  "transaction_id": "TX_TEST_001",
  "fraud_probability": 0.31567527345000485,
  "prediction": 0,
  "threshold": 0.42570148755502263,
  "model_version": "local-checkpoint",
  "persisted": true
}
```

---

# Drift Monitoring

FraudX includes a lightweight production-style monitoring layer based on **Population Stability Index (PSI)**.

### Prediction monitoring

```http
GET /monitoring/predictions?window_hours=24
```

Tracks:

- Prediction volume.
- Fraud prediction rate.
- Average fraud probability.
- Current vs previous prediction-score distribution.
- PSI drift level.

### Feature monitoring

```http
GET /monitoring/features?window_hours=24
```

Compares online numeric feature distributions across adjacent time windows.

### PSI interpretation

| PSI | Status |
|---:|---|
| `< 0.10` | Stable |
| `0.10–0.25` | Warning |
| `>= 0.25` | Drift |

The monitoring endpoints return `insufficient_data` until both comparison windows contain enough observations.

See [`docs/MONITORING.md`](docs/MONITORING.md) for implementation details and limitations.

---

# MLflow

FraudX tracks training runs with MLflow.

**Experiment:** `FraudX-Fraud-Detection`

Tracked information includes:

- Model parameters.
- ROC-AUC, PR-AUC, precision, recall and F1.
- Optimized threshold.
- Training reports.
- Model checkpoint artifacts.
- Feature metadata.

The verified run produced:

```text
ROC-AUC        0.8921
PR-AUC         0.4857
Precision      0.5725
Recall         0.4567
F1             0.5081
Threshold      0.4257
```

MLflow is containerized locally with a pinned `mlflow==3.15.1` image build and SQLite-backed tracking metadata.

Open the local UI at:

**http://127.0.0.1:5000**

---

# Docker Deployment

FraudX runs as three local services:

```text
┌─────────────────────────────────────────────┐
│              Docker Compose                 │
│                                             │
│  ┌────────────┐  ┌────────────┐  ┌────────┐│
│  │ FraudX API │  │  MongoDB   │  │ MLflow ││
│  │   :8001    │  │   :27017   │  │ :5000  ││
│  └────────────┘  └────────────┘  └────────┘│
└─────────────────────────────────────────────┘
```

The API image includes the Linux OpenMP runtime required by LightGBM/XGBoost. MLflow is built locally from `Dockerfile.mlflow`, avoiding a runtime dependency on an external MLflow container registry.

### Start

```bash
git clone https://github.com/Atul1127/Fraud-Detection-Risk-Scoring-System.git
cd Fraud-Detection-Risk-Scoring-System
docker compose up --build -d
```

### Check

```bash
docker compose ps
```

Expected services:

```text
fraudx-api       Up
fraudx-mlflow    Up (healthy)
fraudx-mongodb   Up (healthy)
```

### Stop

```bash
docker compose down
```

### Local endpoints

| Service | URL |
|---|---|
| FastAPI Swagger | http://127.0.0.1:8001/docs |
| FastAPI health | http://127.0.0.1:8001/health |
| MLflow | http://127.0.0.1:5000 |
| MongoDB | `mongodb://127.0.0.1:27017` |

---

# CI/CD

GitHub Actions runs on pushes and pull requests to `main`.

```text
Push / Pull Request
        │
        ▼
Checkout → Python 3.12 → Install dependencies
        │
        ▼
Compile sources → Pytest → Docker Compose validation
        │
        ▼
Docker Build
        │
        └── main push → publish FraudX API image to GHCR
```

The workflow also uses GitHub Actions cache for Docker builds.

---

# Testing

The automated suite covers:

- Temporal ordering and train/validation separation.
- Historical frequency and transaction-time feature behavior.
- Numeric feature output after preprocessing.
- Evaluation metrics and threshold contracts.
- Ensemble probability shape and bounds.
- API/project configuration.
- MongoDB/online feature behavior.
- Drift/PSI calculations.
- Docker Compose configuration.

**Verified local result: `18 passed`.**

Run locally:

```bash
pytest -q
```

---

# Explainability

SHAP is integrated for model interpretation, including:

- Global feature importance.
- Per-transaction explanations.
- Waterfall-style explanations.
- Feature contribution analysis.

The objective is not only to predict fraud but also to provide evidence for why a transaction received a high fraud score.

---

# Streamlit Dashboard

Run the interactive analysis dashboard with:

```bash
streamlit run app/streamlit_app.py
```


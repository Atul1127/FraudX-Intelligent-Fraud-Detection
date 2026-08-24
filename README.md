# FraudX — Intelligent Fraud Detection Platform

> **End-to-end, time-aware fraud detection with ensemble ML, online historical features, explainability, experiment tracking, API serving, Docker, and CI/CD.**

[![CI](https://github.com/Atul1127/FraudX-Intelligent-Fraud-Detection/actions/workflows/ci.yml/badge.svg)](https://github.com/Atul1127/FraudX-Intelligent-Fraud-Detection/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-Online%20Features-47A248?logo=mongodb&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-CI%2FCD-2088FF?logo=githubactions&logoColor=white)

FraudX is a production-style fraud detection platform built on the IEEE-CIS Fraud Detection dataset. The project combines **chronological validation, leakage-conscious feature engineering, imbalanced learning, XGBoost/LightGBM/CatBoost ensembles, Optuna tuning, threshold optimization, SHAP explainability, MongoDB-backed historical features, FastAPI inference, MLflow tracking, Docker Compose, and GitHub Actions CI/CD**.

---

## Recruiter Snapshot

- **590,540 transactions** processed with chronological train/validation evaluation.
- **0.8921 ROC-AUC**, **0.4857 PR-AUC**, and **0.5081 F1** from the final tuned weighted ensemble.
- Ensemble F1 improved by approximately **8.5%** after Optuna tuning (**0.4681 → 0.5081**).
- A separate chronological stacking experiment reached **0.9177 ROC-AUC / 0.5445 PR-AUC**.
- **8/8 automated tests passing** in the verified local suite.
- Real-time inference exposed through **FastAPI** with MongoDB persistence and historical feature retrieval.
- Experiment parameters, metrics, reports, and model artifacts tracked with **MLflow**.
- Full local stack containerized with **Docker Compose** and validated through **GitHub Actions**.

---

## Architecture

```text
                         ┌──────────────────────┐
                         │     GitHub Push       │
                         └──────────┬───────────┘
                                    │
                              GitHub Actions
                                    │
                         ┌──────────▼───────────┐
                         │ Tests + Docker Build │
                         └──────────┬───────────┘
                                    │
             ┌──────────────────────┴──────────────────────┐
             │                                             │
             ▼                                             ▼
      ┌───────────────┐                            ┌────────────────┐
      │ Training      │                            │ FastAPI        │
      │ Pipeline      │                            │ Inference API  │
      └───────┬───────┘                            └───────┬────────┘
              │                                            │
              ▼                                            ▼
     ┌─────────────────┐                          ┌─────────────────┐
     │ Feature         │                          │ MongoDB         │
     │ Engineering     │◄────────────────────────►│ Historical Data │
     └────────┬────────┘                          └─────────────────┘
              │
              ▼
      Chronological Split
              │
              ▼
        Training Only
           SMOTE
              │
       ┌──────┼──────┐
       ▼      ▼      ▼
    XGBoost LightGBM CatBoost
       └──────┼──────┘
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
     SHAP        MongoDB
  Explanation    Prediction Log

              ┌──────────────────┐
              │      MLflow      │
              │ Runs / Metrics / │
              │ Artifacts        │
              └──────────────────┘
```

### Runtime services

| Service | Purpose | Host Port |
|---|---|---:|
| **FraudX API** | Real-time fraud scoring | `8001` |
| **MongoDB** | Historical transactions / prediction persistence | `27017` |
| **MLflow** | Experiment tracking and artifacts | `5000` |

> Port `8001` is intentional so FraudX can coexist with other local applications using port `8000`.

---

## Core Capabilities

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
| Containerization | Docker + Docker Compose |
| Automated validation | Pytest |
| CI/CD | GitHub Actions |
| Interactive analysis | Streamlit |

---

# Model Performance

### Final tuned temporal evaluation

The primary benchmark uses chronological validation: earlier transactions are used for training and later transactions are held out for validation.

| Model | ROC-AUC | PR-AUC | F1 |
|---|---:|---:|---:|
| XGBoost | 0.8832 | 0.4551 | 0.4692 |
| LightGBM | 0.8491 | 0.3748 | 0.3993 |
| CatBoost | 0.8773 | 0.4624 | 0.4778 |
| **Weighted Ensemble** | **0.8921** | **0.4857** | **0.5081** |

**Best ensemble threshold:** `0.426`  
**Precision:** `0.5725` · **Recall:** `0.4567` · **F1:** `0.5081`

FraudX applies SMOTE only to the training split, leaving the chronological validation period untouched.

### Why PR-AUC matters

Fraud detection is an imbalanced classification problem. ROC-AUC is useful, but **PR-AUC, precision, recall, and F1** provide a more informative view of minority-class performance.

### Optuna improvement

| Metric | Before tuning | After tuning |
|---|---:|---:|
| ROC-AUC | 0.8888 | **0.8921** |
| PR-AUC | 0.4726 | **0.4857** |
| F1 | 0.4681 | **0.5081** |

### Separate stacking experiment

A chronological stacking experiment is retained separately rather than silently replacing the default ensemble.

| Metric | Stacking |
|---|---:|
| ROC-AUC | **0.9177** |
| PR-AUC | **0.5445** |
| F1 | 0.4875 |
| Threshold | 0.810 |

> Results can vary with library versions, cached artifacts, configuration, and feature changes. The weighted ensemble remains the project's default production path.

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

Time-dependent count features are calculated in transaction order and do not use future rows. This is critical for avoiding temporal leakage.

## Imbalanced Learning

Fraud transactions are heavily underrepresented. FraudX applies **SMOTE only to the training split**, while validation remains untouched. The boosting models also use class-balancing mechanisms.

```text
Training data ──► SMOTE ──► Model fitting
Validation    ─────────────────► Evaluation
```

## Model Ensemble

The default ensemble combines three complementary gradient-boosting learners:

```text
XGBoost  ── 35% ──┐
LightGBM ── 35% ──┼──► Fraud Probability
CatBoost ── 30% ──┘
```

Optuna searches a compact hyperparameter space using **PR-AUC** as the optimization objective. The decision threshold is optimized on validation data and persisted in configuration.

---

# Online Feature Store with MongoDB

FraudX does not treat MongoDB as a generic database add-on. It provides historical context during online inference.

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
Persist prediction + transaction
```

Historical queries use earlier transactions rather than future records, preserving the temporal nature of the feature pipeline.

---

# FastAPI Inference API

FraudX exposes the trained ensemble through a REST API.

## Swagger

After starting Docker Compose:

**http://127.0.0.1:8001/docs**

## Health check

```http
GET /health
```

Expected response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "mongodb_connected": true
}
```

## Prediction

Example request:

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

A later transaction with historical context produced a different probability, demonstrating that the online historical feature path is active.

---

# MLflow Experiment Tracking

FraudX tracks training runs with MLflow.

**Experiment:** `FraudX-Fraud-Detection`

Tracked information includes:

- Model parameters.
- ROC-AUC.
- PR-AUC.
- Precision.
- Recall.
- F1.
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

Local MLflow uses SQLite for run metadata and persistent local artifact storage.

Launch the UI with the project's configured MLflow setup and open:

**http://127.0.0.1:5000**

---

# Dockerized Deployment

FraudX runs as a multi-service Docker Compose stack:

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

The API image includes the Linux OpenMP runtime required by LightGBM/XGBoost.

### Start the stack

```bash
git clone https://github.com/Atul1127/FraudX-Intelligent-Fraud-Detection.git
cd FraudX-Intelligent-Fraud-Detection
docker compose up --build -d
```

### Check services

```bash
docker compose ps
```

### Stop services

```bash
docker compose down
```

### Endpoints

| Service | URL |
|---|---|
| FastAPI Swagger | http://127.0.0.1:8001/docs |
| FastAPI health | http://127.0.0.1:8001/health |
| MLflow | http://127.0.0.1:5000 |
| MongoDB | `mongodb://127.0.0.1:27017` |

---

# CI/CD

GitHub Actions validates the repository on pushes and pull requests to `main`.

```text
Git Push / Pull Request
          │
          ▼
   Checkout Repository
          │
          ▼
     Python 3.12
          │
          ▼
  Install Dependencies
          │
          ▼
    Compile Sources
          │
          ▼
      Run Pytest
          │
          ▼
 Validate Docker Compose
          │
          ▼
    Build Docker Image
          │
          ▼
         PASS
```

The workflow also publishes the Docker image to GitHub Container Registry when the CI/CD path succeeds.

---

# Testing

The project includes automated tests covering important ML contracts:

- Temporal ordering and train/validation separation.
- Historical frequency and transaction-time feature behavior.
- Numeric feature output after preprocessing.
- Evaluation metric and threshold contracts.
- Ensemble probability shape and bounds.
- API/project configuration validation.
- Docker Compose configuration validation in CI.

Current verified local suite:

```text
8 passed
```

Run locally:

```bash
pytest -q
```

---

# Explainability

SHAP is integrated for model interpretation.

The project supports:

- Global feature importance.
- Per-transaction explanations.
- Waterfall-style explanations.
- Feature contribution analysis.

The goal is not only to predict fraud, but also to provide evidence for **why a transaction received a high fraud score**.

---

# Streamlit Dashboard

The repository also contains an interactive Streamlit application for model exploration and explainability.

```bash
streamlit run app/streamlit_app.py
```

The dashboard provides:

- Transaction scoring.
- Fraud probability and optimized threshold.
- SHAP explanations.
- ROC and PR curves.
- Confusion matrix.
- Feature importance.
- Interactive threshold analysis.

---

# Project Structure

```text
FraudX-Intelligent-Fraud-Detection/
│
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── aws-deploy.yml
│
├── api/
│   ├── main.py
│   └── dependencies.py
│
├── app/
│   ├── streamlit_app.py
│   ├── utils.py
│   └── components/
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── evaluate.py
│   ├── explain.py
│   ├── mlflow_tracker.py
│   ├── stacking.py
│   └── tune.py
│
├── tests/
├── models/
├── config.yaml
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── train.py
└── README.md
```

Large datasets, generated artifacts, and local databases are intentionally excluded from Git.

---

# Local Development

## Install dependencies

```bash
pip install -r requirements.txt
```

## Download IEEE-CIS data

Download the competition files from Kaggle and place them under:

```text
data/raw/
├── train_transaction.csv
├── train_identity.csv
├── test_transaction.csv
└── test_identity.csv
```

The dataset is not included because of its size and Kaggle distribution restrictions.

## Train the main ensemble

```bash
python train.py
```

The pipeline reuses cached processed features when available and writes model checkpoints plus a training report under the configured model directory.

## Tune models

```bash
python src/tune.py
```

Optuna optimizes PR-AUC using an inner chronological validation split so the final validation period is not used for hyperparameter selection.

## Run stacking experiment

```bash
python src/stacking.py
```

## Run random benchmark

```bash
python -m src.benchmark_random
```

The random benchmark is comparison-only; temporal validation remains the primary evaluation protocol.

## Run tests

```bash
pytest -q
```

---

# Why Temporal Validation?

Fraud behavior changes over time. A random split can place highly related transactions from the same period on both sides of the split and produce optimistic estimates.

FraudX instead evaluates a more realistic deployment scenario:

```text
Earlier transactions ─────────────► Later transactions
        TRAIN                           VALIDATION
```

The random benchmark is retained only to quantify the difference between conventional and time-aware validation.

---

# Technology Stack

### Programming

Python

### Machine Learning

Pandas · NumPy · SciPy · Scikit-learn · XGBoost · LightGBM · CatBoost · Optuna · imbalanced-learn

### Explainability

SHAP

### Serving

FastAPI · Uvicorn · Pydantic

### Data / Persistence

MongoDB

### MLOps

MLflow · Docker · Docker Compose · GitHub Actions · GitHub Container Registry

### Visualization

Streamlit · Matplotlib · Seaborn

---

# Design Decisions

### Why an ensemble?

XGBoost, LightGBM, and CatBoost have different tree-growing and optimization characteristics. Combining their probabilities can improve robustness over relying on a single learner.

### Why optimize the threshold?

A default probability threshold of `0.5` is not necessarily appropriate for an imbalanced fraud problem. FraudX selects a validation threshold based on the project's evaluation objective instead.

### Why MongoDB?

The online inference path needs historical transaction context. MongoDB provides a simple persistence layer for retrieving earlier transactions and storing predictions.

### Why MLflow?

Fraud detection experiments involve many model, feature, threshold, and preprocessing choices. MLflow provides reproducible run-level tracking instead of relying on manually recorded results.

### Why Docker?

Docker makes the API, database, and experiment-tracking environment reproducible across machines and simplifies local deployment.

### Why no Kubernetes?

FraudX is intentionally scoped as a single-service ML platform with supporting infrastructure. Kubernetes would add operational complexity without being required by the current workload. It is better treated as a separate MLOps learning project than forced into this system.

### Why no AWS?

The project is designed to demonstrate the complete ML-to-serving-to-MLOps workflow locally. Cloud deployment is intentionally optional rather than adding infrastructure solely for a resume keyword.

---

# Future Improvements

Potential next iterations include:

- Prediction drift monitoring.
- Feature distribution monitoring.
- Prometheus/Grafana observability.
- Kafka-based streaming ingestion.
- Batch inference jobs.
- Model registry promotion workflows.
- Automated retraining triggers.
- Additional calibration experiments.

These are deliberately separated from the current stable production-style path.

---

# Resume-Ready Summary

**FraudX — Intelligent Fraud Detection Platform**

Built an end-to-end fraud detection platform using an ensemble of XGBoost, LightGBM, and CatBoost with Optuna hyperparameter optimization, SHAP explainability, time-aware validation, and threshold optimization; achieved **0.892 ROC-AUC / 0.486 PR-AUC** on chronological validation. Developed a **FastAPI** inference service with **MongoDB-backed historical features and prediction persistence**, integrated **MLflow** for experiment tracking and model artifacts, containerized the platform with **Docker Compose**, and automated testing and image builds through **GitHub Actions CI/CD**.

---

## License

This project is intended for educational and portfolio use. Review the IEEE-CIS dataset's original competition terms before redistributing any dataset files.

---

## Author

**Atul**  
Machine Learning · Deep Learning · MLOps · Generative AI

[GitHub Repository](https://github.com/Atul1127/FraudX-Intelligent-Fraud-Detection)

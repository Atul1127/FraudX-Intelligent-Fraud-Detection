from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
)


def compute_metrics(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5
) -> dict:
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


def find_best_threshold(
    y_true: np.ndarray, y_proba: np.ndarray
) -> tuple[float, float]:
    """Return threshold that maximises F1 on the provided data."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = np.where(
        (precision + recall) == 0,
        0.0,
        2 * precision * recall / (precision + recall),
    )
    best_idx = int(np.argmax(f1_scores[:-1]))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def get_roc_curve(
    y_true: np.ndarray, y_proba: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    return fpr, tpr, thresholds


def get_pr_curve(
    y_true: np.ndarray, y_proba: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    return precision, recall, thresholds


def get_confusion_matrix(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float
) -> np.ndarray:
    y_pred = (y_proba >= threshold).astype(int)
    return confusion_matrix(y_true, y_pred)

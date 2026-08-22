import numpy as np

from src.evaluate import compute_metrics, find_best_threshold


def test_find_best_threshold_returns_valid_threshold():
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.05, 0.20, 0.80, 0.95])

    threshold, score = find_best_threshold(y_true, probabilities)

    assert 0.0 <= threshold <= 1.0
    assert 0.0 <= score <= 1.0


def test_compute_metrics_contains_core_fraud_metrics():
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.05, 0.20, 0.80, 0.95])

    metrics = compute_metrics(y_true, probabilities, threshold=0.5)

    for key in ["auc_roc", "auc_pr", "precision", "recall", "f1"]:
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0

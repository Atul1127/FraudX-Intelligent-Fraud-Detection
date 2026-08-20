from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def render_pr_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    thresholds: np.ndarray,
    current_threshold: float,
    auc_pr: float,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(recall, precision, color="#1f77b4", linewidth=2, label=f"PR curve (AUC={auc_pr:.3f})")

    # Mark current threshold
    dists = np.abs(thresholds - current_threshold)
    idx = int(np.argmin(dists))
    ax.scatter(
        recall[idx],
        precision[idx],
        color="#d62728",
        zorder=5,
        s=120,
        label=f"Threshold = {current_threshold:.2f}",
    )

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def render_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_roc: float,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fpr, tpr, color="#1f77b4", linewidth=2, label=f"ROC (AUC={auc_roc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig

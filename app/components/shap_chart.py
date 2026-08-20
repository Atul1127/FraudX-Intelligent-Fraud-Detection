from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import shap


def render_waterfall(
    explanation: shap.Explanation,
    max_display: int = 15,
) -> plt.Figure:
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    fig = plt.gcf()
    plt.tight_layout()
    return fig


def render_summary(
    shap_values: "np.ndarray",
    X: pd.DataFrame,
    max_display: int = 20,
) -> plt.Figure:
    shap.summary_plot(shap_values, X, max_display=max_display, show=False)
    fig = plt.gcf()
    plt.tight_layout()
    return fig

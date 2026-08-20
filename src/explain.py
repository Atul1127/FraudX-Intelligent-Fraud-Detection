from __future__ import annotations

import numpy as np
import pandas as pd
import shap
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")


def build_explainers(model, X_background: pd.DataFrame) -> dict:
    """Build SHAP TreeExplainers for both sub-models."""
    X_bg = X_background.fillna(X_background.median(numeric_only=True))
    return {
        "xgb": shap.TreeExplainer(model.xgb_model, X_bg),
        "lgbm": shap.TreeExplainer(model.lgbm_model, X_bg),
        "xgb_weight": model.xgb_weight,
        "lgbm_weight": model.lgbm_weight,
    }


def explain_transaction(
    row: pd.DataFrame,
    explainers: dict,
    feature_names: list[str],
) -> shap.Explanation:
    """Return a weighted-average SHAP Explanation for a single row."""
    row_filled = row.fillna(row.median(numeric_only=True))

    sv_xgb = explainers["xgb"].shap_values(row_filled)
    sv_lgbm = explainers["lgbm"].shap_values(row_filled)

    # LightGBM returns list[array] for binary; XGBoost returns array
    if isinstance(sv_lgbm, list):
        sv_lgbm = sv_lgbm[1]
    if isinstance(sv_xgb, list):
        sv_xgb = sv_xgb[1]

    total = explainers["xgb_weight"] + explainers["lgbm_weight"]
    sv_combined = (
        explainers["xgb_weight"] * sv_xgb + explainers["lgbm_weight"] * sv_lgbm
    ) / total

    base_xgb = float(explainers["xgb"].expected_value)
    base_lgbm = float(
        explainers["lgbm"].expected_value[1]
        if hasattr(explainers["lgbm"].expected_value, "__len__")
        else explainers["lgbm"].expected_value
    )
    base = (explainers["xgb_weight"] * base_xgb + explainers["lgbm_weight"] * base_lgbm) / total

    return shap.Explanation(
        values=sv_combined[0],
        base_values=base,
        data=row_filled.values[0],
        feature_names=feature_names,
    )


def plot_waterfall(explanation: shap.Explanation, max_display: int = 15) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    fig = plt.gcf()
    plt.tight_layout()
    return fig


def plot_summary(
    model,
    X: pd.DataFrame,
    explainers: dict,
    max_display: int = 20,
    plot_type: str = "bar",
) -> plt.Figure:
    X_filled = X.fillna(X.median(numeric_only=True))
    sv_xgb = explainers["xgb"].shap_values(X_filled)
    sv_lgbm = explainers["lgbm"].shap_values(X_filled)

    if isinstance(sv_lgbm, list):
        sv_lgbm = sv_lgbm[1]
    if isinstance(sv_xgb, list):
        sv_xgb = sv_xgb[1]

    total = explainers["xgb_weight"] + explainers["lgbm_weight"]
    sv_combined = (
        explainers["xgb_weight"] * sv_xgb + explainers["lgbm_weight"] * sv_lgbm
    ) / total

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        sv_combined,
        X_filled,
        plot_type=plot_type,
        max_display=max_display,
        show=False,
    )
    fig = plt.gcf()
    plt.tight_layout()
    return fig

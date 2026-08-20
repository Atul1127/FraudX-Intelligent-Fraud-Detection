from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import shap


def build_explainers(model, X_background: pd.DataFrame) -> dict:
    X_bg = X_background.fillna(
        X_background.median(numeric_only=True)
    )

    return {
        "xgb": shap.TreeExplainer(
            model.xgb_model,
            X_bg,
        ),
        "lgbm": shap.TreeExplainer(
            model.lgb_model,
            X_bg,
        ),
        "cat": shap.TreeExplainer(
            model.cat_model,
            X_bg,
        ),
    }


def _get_shap_values(explainer, X):
    values = explainer.shap_values(X)

    if isinstance(values, list):
        values = values[1]

    return values


def explain_transaction(
    row: pd.DataFrame,
    explainers: dict,
    feature_names: list[str],
) -> shap.Explanation:

    row_filled = row.fillna(
        row.median(numeric_only=True)
    )

    sv_xgb = _get_shap_values(
        explainers["xgb"],
        row_filled,
    )

    sv_lgbm = _get_shap_values(
        explainers["lgbm"],
        row_filled,
    )

    sv_cat = _get_shap_values(
        explainers["cat"],
        row_filled,
    )

    combined = (
        0.35 * sv_xgb
        + 0.35 * sv_lgbm
        + 0.30 * sv_cat
    )

    base_xgb = float(
        explainers["xgb"].expected_value
    )

    base_lgbm = explainers["lgbm"].expected_value

    if hasattr(base_lgbm, "__len__"):
        base_lgbm = float(base_lgbm[1])
    else:
        base_lgbm = float(base_lgbm)

    base_cat = explainers["cat"].expected_value

    if hasattr(base_cat, "__len__"):
        base_cat = float(base_cat[1])
    else:
        base_cat = float(base_cat)

    base = (
        0.35 * base_xgb
        + 0.35 * base_lgbm
        + 0.30 * base_cat
    )

    return shap.Explanation(
        values=combined[0],
        base_values=base,
        data=row_filled.values[0],
        feature_names=feature_names,
    )


def plot_waterfall(
    explanation: shap.Explanation,
    max_display: int = 15,
) -> plt.Figure:

    shap.plots.waterfall(
        explanation,
        max_display=max_display,
        show=False,
    )

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

    X_filled = X.fillna(
        X.median(numeric_only=True)
    )

    sv_xgb = _get_shap_values(
        explainers["xgb"],
        X_filled,
    )

    sv_lgbm = _get_shap_values(
        explainers["lgbm"],
        X_filled,
    )

    sv_cat = _get_shap_values(
        explainers["cat"],
        X_filled,
    )

    combined = (
        0.35 * sv_xgb
        + 0.35 * sv_lgbm
        + 0.30 * sv_cat
    )

    shap.summary_plot(
        combined,
        X_filled,
        plot_type=plot_type,
        max_display=max_display,
        show=False,
    )

    fig = plt.gcf()
    plt.tight_layout()

    return fig

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.utils import (
    load_config,
    load_model,
    load_val_data,
    compute_val_probas,
    load_explainers,
    load_training_report,
    checkpoint_ready,
    processed_data_ready,
)
from app.components.shap_chart import render_waterfall
from app.components.pr_curve import render_pr_curve, render_roc_curve
from src.evaluate import (
    compute_metrics,
    get_roc_curve,
    get_pr_curve,
    get_confusion_matrix,
)
from src.explain import explain_transaction

CKPT_DIR = "models/checkpoints"
PROC_DIR = "data/processed"

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
)
st.title("Fraud Detection — Interpretable Ensemble")

# ── Readiness checks ─────────────────────────────────────────────────────────
if not checkpoint_ready(CKPT_DIR):
    st.error(
        "No trained model found. Run `python train.py` first to train the ensemble."
    )
    st.stop()

if not processed_data_ready(PROC_DIR):
    st.error(
        "Processed validation data not found. Run `python train.py` first."
    )
    st.stop()

cfg = load_config()
model = load_model(CKPT_DIR, CKPT_DIR)
report = load_training_report(CKPT_DIR)

tab1, tab2, tab3 = st.tabs(["Score a Transaction", "Model Performance", "Threshold Tuning"])


# ── Tab 1: Score a Transaction ────────────────────────────────────────────────
with tab1:
    st.subheader("Real-Time Transaction Scoring")
    st.markdown(
        "Upload a CSV file with a **single transaction row** (same columns as "
        "`train_transaction.csv` merged with `train_identity.csv`)."
    )

    uploaded = st.file_uploader("Upload transaction CSV", type="csv")

    if uploaded is not None:
        try:
            row_df = pd.read_csv(uploaded)
            if len(row_df) > 1:
                st.warning(f"File has {len(row_df)} rows — using only the first row.")
                row_df = row_df.iloc[:1]

            # Feature engineering
            with st.spinner("Engineering features..."):
                from src.data.features import build_features
                row_feat = build_features(row_df, cfg)

                # Align to trained feature names
                for col in model.feature_names:
                    if col not in row_feat.columns:
                        row_feat[col] = np.nan
                row_feat = row_feat[model.feature_names]

            fraud_proba = float(model.predict_proba(row_feat)[0])
            threshold = cfg["ensemble"]["default_threshold"]
            is_fraud = fraud_proba >= threshold

            col1, col2, col3 = st.columns(3)
            col1.metric("Fraud Probability", f"{fraud_proba:.1%}")
            col2.metric("Threshold", f"{threshold:.2f}")
            col3.metric(
                "Verdict",
                "FRAUD" if is_fraud else "LEGITIMATE",
                delta=None,
            )
            if is_fraud:
                st.error("Transaction flagged as **FRAUDULENT**")
            else:
                st.success("Transaction appears **LEGITIMATE**")

            # SHAP waterfall
            st.subheader("SHAP Explanation")
            with st.spinner("Computing SHAP values..."):
                explainers = load_explainers(CKPT_DIR, PROC_DIR, CKPT_DIR)
                explanation = explain_transaction(row_feat, explainers, model.feature_names)
                fig = render_waterfall(explanation, max_display=15)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error processing transaction: {e}")
    else:
        st.info(
            "No file uploaded yet. You can also use the raw input form below for quick testing."
        )
        with st.expander("Manual input (key fields only)"):
            amt = st.number_input("TransactionAmt", min_value=0.0, value=50.0, step=1.0)
            card1 = st.number_input("card1", min_value=0, value=10000, step=1)
            p_email = st.text_input("P_emaildomain", value="gmail.com")
            r_email = st.text_input("R_emaildomain", value="gmail.com")
            dt = st.number_input("TransactionDT", min_value=0, value=86400, step=3600)

            if st.button("Score transaction"):
                row_df = pd.DataFrame(
                    [{
                        "TransactionAmt": amt,
                        "card1": card1,
                        "P_emaildomain": p_email,
                        "R_emaildomain": r_email,
                        "TransactionDT": dt,
                        "TransactionID": 0,
                    }]
                )
                from src.data.features import build_features
                row_feat = build_features(row_df, cfg)
                for col in model.feature_names:
                    if col not in row_feat.columns:
                        row_feat[col] = np.nan
                row_feat = row_feat[model.feature_names]

                fraud_proba = float(model.predict_proba(row_feat)[0])
                threshold = cfg["ensemble"]["default_threshold"]
                st.metric("Fraud Probability", f"{fraud_proba:.1%}")
                if fraud_proba >= threshold:
                    st.error("FRAUD")
                else:
                    st.success("LEGITIMATE")


# ── Tab 2: Model Performance ──────────────────────────────────────────────────
with tab2:
    st.subheader("Validation Set Performance")

    y_true, y_proba = compute_val_probas(CKPT_DIR, PROC_DIR)
    threshold = cfg["ensemble"]["default_threshold"]
    metrics = compute_metrics(y_true, y_proba, threshold)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AUC-ROC", f"{metrics['auc_roc']:.4f}")
    c2.metric("AUC-PR", f"{metrics['auc_pr']:.4f}")
    c3.metric("Precision", f"{metrics['precision']:.4f}")
    c4.metric("Recall", f"{metrics['recall']:.4f}")

    col_roc, col_cm = st.columns(2)

    with col_roc:
        fpr, tpr, _ = get_roc_curve(y_true, y_proba)
        fig_roc = render_roc_curve(fpr, tpr, metrics["auc_roc"])
        st.pyplot(fig_roc)

    with col_cm:
        import seaborn as sns
        import matplotlib.pyplot as plt
        cm = get_confusion_matrix(y_true, y_proba, threshold)
        fig_cm, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Legit", "Fraud"],
            yticklabels=["Legit", "Fraud"],
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix (threshold={threshold:.2f})")
        plt.tight_layout()
        st.pyplot(fig_cm)

    if report:
        st.subheader("Top Predictive Features (XGBoost importance)")
        top_df = pd.DataFrame(report["top_features"], columns=["Feature", "Importance"])
        st.bar_chart(top_df.set_index("Feature"))


# ── Tab 3: Threshold Tuning ───────────────────────────────────────────────────
with tab3:
    st.subheader("Precision-Recall Threshold Tuning")

    y_true, y_proba = compute_val_probas(CKPT_DIR, PROC_DIR)

    threshold = st.slider(
        "Decision Threshold",
        min_value=0.01,
        max_value=0.99,
        value=float(cfg["ensemble"]["default_threshold"]),
        step=0.01,
        help="Increase to reduce false positives; decrease to catch more fraud.",
    )

    metrics = compute_metrics(y_true, y_proba, threshold)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision", f"{metrics['precision']:.4f}")
    c2.metric("Recall", f"{metrics['recall']:.4f}")
    c3.metric("F1", f"{metrics['f1']:.4f}")
    c4.metric(
        "Flagged",
        f"{int((y_proba >= threshold).sum()):,}",
        delta=f"{(y_proba >= threshold).mean():.1%} of transactions",
    )

    precision, recall, thresholds = get_pr_curve(y_true, y_proba)
    fig_pr = render_pr_curve(precision, recall, thresholds, threshold, metrics["auc_pr"])
    st.pyplot(fig_pr)

    if st.button("Save threshold to config.yaml"):
        cfg["ensemble"]["default_threshold"] = float(threshold)
        import yaml
        with open("config.yaml", "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)
        st.success(f"Threshold {threshold:.2f} saved to config.yaml")

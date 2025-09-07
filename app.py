import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_fscore_support
)

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")
st.title("💳 Credit Card Fraud Detection")
st.caption("Imbalanced classification on the Kaggle credit card dataset (Scikit-learn / XGBoost + SMOTE).")

ART_DIR = Path("artifacts")
MODEL_PATH = ART_DIR / "model_best.pkl"
PREPROC_PATH = ART_DIR / "preprocessor.pkl"
FEATS_PATH = ART_DIR / "feature_names.json"
METRICS_PATH = ART_DIR / "metrics.json"

@st.cache_resource(show_spinner=False)
def load_artifacts():
    if not (MODEL_PATH.exists() and PREPROC_PATH.exists() and FEATS_PATH.exists()):
        return None, None, None
    preproc = joblib.load(PREPROC_PATH)
    model = joblib.load(MODEL_PATH)
    features = json.loads(FEATS_PATH.read_text())
    return preproc, model, features

preproc, model, features = load_artifacts()

colL, colR = st.columns([2,1])
with colR:
    st.subheader("Model status")
    if model is None:
        st.error("Artifacts not found. Run `python train_fraud.py` first.")
    else:
        st.success("Model loaded")
        if METRICS_PATH.exists():
            m = json.loads(METRICS_PATH.read_text())
            st.metric("ROC-AUC", f"{m['best']['roc_auc']:.3f}")
            st.metric("Recall (fraud)", f"{m['best']['recall_pos']:.3f}")
            st.metric("Precision (fraud)", f"{m['best']['precision_pos']:.3f}")
        threshold = st.slider("Decision threshold", 0.01, 0.99, 0.50, 0.01,
                              help="Score ≥ threshold ⇒ classify as fraud")

with colL:
    st.subheader("Single transaction (upload 1-row CSV)")
    st.caption("Required columns: V1..V28, Amount, Time (order doesn’t matter).")
    one = st.file_uploader("Upload single-row CSV", type=["csv"], key="one")
    if one is not None and model is not None:
        try:
            df1 = pd.read_csv(one)
            if len(df1) != 1:
                st.error("CSV must contain exactly 1 row.")
            else:
                miss = [c for c in features if c not in df1.columns]
                if miss:
                    st.error(f"Missing required columns: {miss}")
                else:
                    X = preproc.transform(df1[features])
                    score = float(model.predict_proba(X)[0,1])
                    pred = int(score >= threshold)
                    st.markdown(f"### Fraud probability: **{score:.3f}** → **{'FRAUD' if pred else 'LEGIT'}**")
                    st.progress(min(max(score,0),1))
        except Exception as e:
            st.error(f"Failed to score row: {e}")

st.markdown("---")
st.subheader("Batch scoring (CSV)")
st.caption("Upload a CSV of transactions. If it includes `Class` (0/1), app will also compute evaluation metrics.")

upl = st.file_uploader("Upload CSV", type=["csv"], key="batch")
if upl is not None and model is not None:
    try:
        df = pd.read_csv(upl)
        miss = [c for c in features if c not in df.columns]
        if miss:
            st.error(f"Missing required columns: {miss}")
        else:
            X = preproc.transform(df[features])
            scores = model.predict_proba(X)[:,1]
            preds = (scores >= threshold).astype(int)

            out = df.copy()
            out["fraud_score"] = scores
            out["prediction"] = preds
            st.dataframe(out.head(30), use_container_width=True)

            # If labeled, show metrics
            if "Class" in df.columns:
                y_true = df["Class"].astype(int).values
                try:
                    auc = roc_auc_score(y_true, scores)
                except Exception:
                    auc = None
                pr, rc, f1, _ = precision_recall_fscore_support(
                    y_true, preds, average="binary", zero_division=0
                )
                st.write("**Evaluation on uploaded labeled data:**")
                if auc is not None:
                    st.metric("ROC-AUC", f"{auc:.3f}")
                st.metric("Recall (fraud)", f"{rc:.3f}")
                st.metric("Precision (fraud)", f"{pr:.3f}")
                st.metric("F1 (fraud)", f"{f1:.3f}")

                # Confusion matrix figure
                cm = confusion_matrix(y_true, preds, labels=[0,1])
                fig = plt.figure()
                plt.imshow(cm, cmap="Blues")
                plt.title("Confusion Matrix (0=Legit, 1=Fraud)")
                plt.xlabel("Predicted")
                plt.ylabel("True")
                for (i, j), v in np.ndenumerate(cm):
                    plt.text(j, i, int(v), ha="center", va="center")
                plt.tight_layout()
                st.pyplot(fig)

                st.text("Classification report:")
                st.text(classification_report(y_true, preds, zero_division=0))

            st.download_button(
                "Download predictions CSV",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="fraud_predictions.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Failed to process CSV: {e}")

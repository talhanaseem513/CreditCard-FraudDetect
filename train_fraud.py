"""
Train fraud detection models on Kaggle Credit Card Fraud dataset.

Input:
  data/creditcard.csv  (columns: Time, V1..V28, Amount, Class)
Output (artifacts/):
  - preprocessor.pkl          (RobustScaler for Time/Amount)
  - model_best.pkl            (best by ROC-AUC)
  - model_logreg.pkl, model_rf.pkl, model_xgb.pkl (if xgboost installed)
  - feature_names.json
  - metrics.json              (per-model metrics + best)
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, precision_recall_fscore_support, classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

DATA_PATH = Path("data/creditcard.csv")
ART_DIR = Path("artifacts")
ART_DIR.mkdir(exist_ok=True, parents=True)

# Features per Kaggle dataset
V_FEATS = [f"V{i}" for i in range(1,29)]
FEATURES = ["Time"] + V_FEATS + ["Amount"]
TARGET = "Class"

def load_data():
    assert DATA_PATH.exists(), f"Missing {DATA_PATH}. Download Kaggle creditcard.csv to data/."
    df = pd.read_csv(DATA_PATH)
    needed = set(FEATURES + [TARGET])
    miss = [c for c in needed if c not in df.columns]
    assert not miss, f"CSV missing required columns: {miss}"
    return df[FEATURES + [TARGET]].copy()

def build_preprocessor():
    # PCA features (V1..V28) typically clean; scale only Time & Amount
    time_amount = ["Time", "Amount"]
    preprocess = ColumnTransformer(
        transformers=[
            ("ta", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", RobustScaler())
            ]), time_amount),
            ("rest", "passthrough", V_FEATS)
        ],
        remainder="drop"
    )
    return preprocess

def fit_with_smote(model, X_train, y_train):
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    model.fit(X_res, y_res)
    return model

def evaluate_and_store(name, model, X_test, y_test, store=True):
    proba = model.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, proba)
    preds = (proba >= 0.5).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary", zero_division=0)
    print(f"\n=== {name} ===")
    print(f"ROC-AUC: {auc:.3f} | Precision: {pr:.3f} | Recall: {rc:.3f} | F1: {f1:.3f}")
    print(classification_report(y_test, preds, zero_division=0))
    if store:
        joblib.dump(model, ART_DIR / f"model_{name}.pkl")
    return {"roc_auc": auc, "precision_pos": pr, "recall_pos": rc, "f1_pos": f1}

def main():
    df = load_data()
    X = df[FEATURES]
    y = df[TARGET].astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    preproc = build_preprocessor()
    X_tr_t = preproc.fit_transform(X_tr)
    X_te_t = preproc.transform(X_te)
    joblib.dump(preproc, ART_DIR / "preprocessor.pkl")
    Path(ART_DIR / "feature_names.json").write_text(json.dumps(FEATURES), encoding="utf-8")

    results = {}

    # Logistic Regression
    logreg = LogisticRegression(max_iter=5000, class_weight="balanced")
    logreg = fit_with_smote(logreg, X_tr_t, y_tr)
    results["logreg"] = evaluate_and_store("logreg", logreg, X_te_t, y_te)

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=600, max_depth=None, random_state=42,
        class_weight="balanced_subsample", n_jobs=-1
    )
    rf = fit_with_smote(rf, X_tr_t, y_tr)
    results["rf"] = evaluate_and_store("rf", rf, X_te_t, y_te)

    # XGBoost (optional)
    if HAS_XGB:
        pos = y_tr.sum()
        neg = len(y_tr) - pos
        spw = max(1.0, neg / max(pos, 1))
        xgb = XGBClassifier(
            n_estimators=800, learning_rate=0.05, max_depth=4,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            objective="binary:logistic", eval_metric="auc",
            tree_method="hist", random_state=42, scale_pos_weight=spw,
            n_jobs=-1
        )
        xgb = fit_with_smote(xgb, X_tr_t, y_tr)
        results["xgb"] = evaluate_and_store("xgb", xgb, X_te_t, y_te)
    else:
        print("\n[!] xgboost not installed; skipping XGBClassifier.")

    # Pick best by ROC-AUC
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = joblib.load(ART_DIR / f"model_{best_name}.pkl")
    joblib.dump(best_model, ART_DIR / "model_best.pkl")

    out = {"by_model": results, "best_model": best_name, "best": results[best_name]}
    Path(ART_DIR / "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nBest model: {best_name}  ROC-AUC={results[best_name]['roc_auc']:.3f}")
    print(f"Artifacts saved to: {ART_DIR.resolve()}")

if __name__ == "__main__":
    main()

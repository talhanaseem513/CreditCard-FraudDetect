# 💳 Credit Card Fraud Detection (Finance Classification)

Detect fraudulent transactions on the highly imbalanced Kaggle dataset (0.17% fraud).  
Includes SMOTE for imbalance, multiple models, a Streamlit app with threshold tuning, and batch CSV scoring.

## ✨ Features
- Models: Logistic Regression, Random Forest, (optional) XGBoost
- Imbalance handling: **SMOTE**
- Metrics: ROC-AUC, Precision/Recall/F1 (fraud class)
- App: **Threshold slider**, single-row & batch scoring, optional evaluation if `Class` present

## 📊 Data
Download from Kaggle: `creditcard.csv`  
Place it at: `data/creditcard.csv`  
Columns used: `Time, V1..V28, Amount, Class`

## ⚙️ Setup
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

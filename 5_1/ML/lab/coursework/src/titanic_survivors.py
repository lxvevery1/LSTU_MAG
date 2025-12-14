import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from data_preprocessing import get_processed_data

# ---------------------------------------------------------
# Prepare output dir
# ---------------------------------------------------------
os.makedirs("models", exist_ok=True)

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
X_train, X_val, y_train, y_val, df, scaler = get_processed_data(
    data_dir="titanic/", test_size=0.2, random_state=42
)

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Save scaler
joblib.dump(scaler, "models/scaler.pkl")


# ============================================================
# Random Forest
# ============================================================
rf = RandomForestClassifier(n_estimators=400, random_state=42)
rf.fit(X_train, y_train)

pred_rf = rf.predict(X_val)
proba_rf = rf.predict_proba(X_val)[:, 1]

print("=== Random Forest ===")
print("Acc:", accuracy_score(y_val, pred_rf))
print("AUC:", roc_auc_score(y_val, proba_rf))
print()

joblib.dump(rf, "models/random_forest.pkl")

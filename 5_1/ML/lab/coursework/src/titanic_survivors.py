import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

data_dir = "titanic/"

# ---------------- Load data ----------------
df = pd.read_csv(data_dir + "train.csv")

# ---------------- Cabin processing ----------------
df["Cabin"] = df["Cabin"].fillna("Unknown")

# Deck = первая буква
df["Deck"] = df["Cabin"].str[0]

# Числовая часть каюты
df["CabinNumber"] = (
    df["Cabin"]
    .str.extract(r"(\d+)", expand=False)
    .fillna(-1)
    .astype(int)
)

df = pd.get_dummies(df, columns=["Deck"], drop_first=True)

df = df.drop(columns=["Cabin"])

# ---------------- Basic cleaning ----------------
df = df.drop(columns=["PassengerId", "Name", "Ticket"])

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

# ---------------- Feature/target split ----------------
X = df.drop(columns=["Survived"])
y = df["Survived"]

# ---------------- Train/val split ----------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ---------------- Random Forest ----------------
rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

pred = rf.predict(X_val)
proba = rf.predict_proba(X_val)[:, 1]

print("=== Random Forest ===")
print("Acc:", accuracy_score(y_val, pred))
print("AUC:", roc_auc_score(y_val, proba))

# ---------------- Feature importance ----------------
print("\nTop features:")
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print(imp.head(10))

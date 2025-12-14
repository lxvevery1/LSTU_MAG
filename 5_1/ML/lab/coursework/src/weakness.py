import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ---------------- Load and preprocess ----------------
def preprocess_titanic(data_dir="titanic/"):
    df = pd.read_csv(data_dir + "train.csv")

    df = df.drop(columns=["PassengerId", "Name", "Ticket"])
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    df["Cabin"] = df["Cabin"].fillna("Unknown")
    df["Deck"] = df["Cabin"].str[0]
    df["CabinNumber"] = df["Cabin"].str.extract(r"(\d+)", expand=False).fillna(-1).astype(int)
    df = pd.get_dummies(df, columns=["Deck", "Embarked"], drop_first=True)
    df = df.drop(columns=["Cabin"])

    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ---------------- Main ----------------
X_train, X_val, y_train, y_val = preprocess_titanic()

# Масштабирование не обязательно для Random Forest, но можно
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train_scaled, y_train)

# Важность признаков
importances = rf.feature_importances_
features = X_train.columns
feature_importance_df = pd.DataFrame({"feature": features, "importance": importances})
feature_importance_df = feature_importance_df.sort_values(by="importance", ascending=False)

print("Top influential features for survival:")
print(feature_importance_df.head(10))

# Optional: визуализация
plt.figure(figsize=(10,6))
plt.barh(feature_importance_df["feature"][:10][::-1], feature_importance_df["importance"][:10][::-1])
plt.xlabel("Importance")
plt.title("Top 10 features for Titanic survival (Random Forest)")
plt.show()

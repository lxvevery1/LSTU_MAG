import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(data_dir="titanic/"):
    """
    Load and preprocess Titanic data.

    Parameters:
    -----------
    data_dir : str
        Directory containing the data file

    Returns:
    --------
    tuple : (X_train, X_val, y_train, y_val, df)
    """
    # ---------------- Load data ----------------
    df = pd.read_csv(data_dir + "train.csv")

    # ---------------- Cabin processing ----------------
    df["Cabin"] = df["Cabin"].fillna("Unknown")

    # Deck = первая буква
    df["Deck"] = df["Cabin"].str[0]

    # Числовая часть каюты
    df["CabinNumber"] = (
        df["Cabin"].str.extract(r"(\d+)", expand=False).fillna(-1).astype(int)
    )

    df = pd.get_dummies(df, columns=["Deck"], drop_first=True)
    df = df.drop(columns=["Cabin"])

    # ---------------- Basic cleaning ----------------
    df = df.drop(columns=["PassengerId", "Name", "Ticket"])

    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

    return df

def split_data(df, test_size=0.2, random_state=42):
    """
    Split data into training and validation sets.

    Parameters:
    -----------
    df : pandas.DataFrame
        Preprocessed dataframe with 'Survived' column
    test_size : float
        Proportion of data for validation set
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    tuple : (X_train, X_val, y_train, y_val)
    """
    # ---------------- Feature/target split ----------------
    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    # ---------------- Train/val split ----------------
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    return X_train, X_val, y_train, y_val

def get_processed_data(data_dir="titanic/", test_size=0.2, random_state=42):
    """
    Main function to get all processed data.

    Parameters:
    -----------
    data_dir : str
        Directory containing the data file
    test_size : float
        Proportion of data for validation set
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    tuple : (X_train, X_val, y_train, y_val, df)
    """
    df = load_and_preprocess_data(data_dir)
    X_train, X_val, y_train, y_val = split_data(df, test_size, random_state)
    scaler = StandardScaler()

    return X_train, X_val, y_train, y_val, df, scaler

def load_models_from_file(data_dir="models/"):
    scaler = joblib.load(data_dir + "scaler.pkl")
    dt = joblib.load(data_dir + "decision_tree.pkl")
    gnb = joblib.load(data_dir + "gaussian_nb.pkl")
    logreg = joblib.load(data_dir + "logreg.pkl")
    rf = joblib.load(data_dir + "random_forest.pkl")
    svm = joblib.load(data_dir + "svm.pkl")

    return scaler, dt, gnb, logreg, rf, svm

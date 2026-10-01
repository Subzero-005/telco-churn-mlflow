"""Loading and preprocessing for the Telco Customer Churn dataset."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "telco_churn.csv"

TARGET = "Churn"
ID_COLUMN = "customerID"

NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
RAW_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_raw(path=DATA_PATH):
    return pd.read_csv(path)


def clean(df):
    """Fix types and fill missing values on raw records."""
    df = df.copy()
    # TotalCharges is blank for customers with zero tenure, so it loads as text.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()
    return df


def encode(df, feature_columns=None):
    """One-hot encode categoricals. Pass feature_columns to align new data with the training layout."""
    X = pd.get_dummies(df[RAW_FEATURES], columns=CATEGORICAL_FEATURES, dtype=int)
    X.columns = [c.replace(" ", "_") for c in X.columns]
    if feature_columns is not None:
        X = X.reindex(columns=feature_columns, fill_value=0)
    return X.astype(float)


def load_train_test(test_size=0.2, random_state=42):
    df = clean(load_raw())
    y = (df[TARGET] == "Yes").astype(int)
    X = encode(df)
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def category_options():
    """Unique values for each categorical column, used to build the app's dropdowns."""
    df = clean(load_raw())
    return {col: sorted(df[col].unique().tolist()) for col in CATEGORICAL_FEATURES}

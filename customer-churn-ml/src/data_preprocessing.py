"""
Data Preprocessing Module for Customer Churn Prediction
======================================================
Handles data loading, initial quality checks, missing value resolution,
datatype coercion, and train-test splitting to prevent data leakage.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_raw_data(filepath: str = "customer-churn-ml/data/dataset.csv") -> pd.DataFrame:
    """Load the customer churn dataset from CSV."""
    df = pd.read_csv(filepath)
    return df


def clean_raw_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform initial data hygiene:
    1. Drop customerID (pure identifier, non-predictive)
    2. Convert TotalCharges from object/string to float, filling empty spaces with 0.0 (where tenure=0)
    3. Ensure SeniorCitizen is treated as categorical/integer consistently
    4. Strip any leading/trailing whitespace from categorical strings
    """
    df_clean = df.copy()

    # Drop customerID if present
    if "customerID" in df_clean.columns:
        df_clean = df_clean.drop(columns=["customerID"])

    # TotalCharges contains ' ' for new customers with tenure = 0
    if "TotalCharges" in df_clean.columns:
        df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"].astype(str).str.strip(), errors="coerce")
        # Impute missing TotalCharges with 0.0 (these are customers with tenure 0)
        df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(0.0)

    # Standardize string columns
    str_cols = df_clean.select_dtypes(include=["object"]).columns
    for col in str_cols:
        df_clean[col] = df_clean[col].astype(str).str.strip()

    return df_clean


def get_feature_types(df: pd.DataFrame):
    """Identify numerical and categorical column names excluding target."""
    target_col = "Churn"
    feature_cols = [c for c in df.columns if c != target_col]

    numerical_cols = []
    categorical_cols = []

    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(df[c]) and c not in ["SeniorCitizen"]:
            numerical_cols.append(c)
        else:
            categorical_cols.append(c)

    return numerical_cols, categorical_cols


def prepare_train_test(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
):
    """
    Split clean dataframe into Train and Test sets BEFORE any transformations
    that compute aggregates (to prevent data leakage).
    Stratify on target variable 'Churn'.
    """
    df_clean = clean_raw_dataframe(df)

    X = df_clean.drop(columns=["Churn"])
    y = df_clean["Churn"].apply(lambda val: 1 if str(val).strip().lower() in ["yes", "1", "true"] else 0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    raw = load_raw_data()
    print("Loaded raw data shape:", raw.shape)
    X_tr, X_te, y_tr, y_te = prepare_train_test(raw)
    print(f"Train set: {X_tr.shape}, Test set: {X_te.shape}")
    print(f"Train Churn Rate: {y_tr.mean():.4f}, Test Churn Rate: {y_te.mean():.4f}")

"""
Feature Engineering Module for Customer Churn Prediction
=========================================================
Implements domain-driven feature engineering:
1. tenure_group: Segments customer lifecycle (new, developing, established, loyal).
2. total_services: Count of active tech/security/entertainment add-ons (higher engagement = lower churn).
3. charges_per_tenure: Average spend progression per month of tenure.
4. has_family_support: Compound flag for both Partner and Dependents (family stability indicator).
5. streaming_bundle: Flag indicating subscription to both StreamingTV and StreamingMovies.
6. monthly_charges_ratio: Ratio of MonthlyCharges relative to expected base.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that applies feature engineering
    without leaking target data or test distribution statistics.
    """

    def __init__(self):
        self.service_columns = [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Allow DataFrame or dict input
        if isinstance(X, dict):
            df = pd.DataFrame([X])
        elif isinstance(X, list):
            df = pd.DataFrame(X)
        else:
            df = X.copy()

        # Ensure TotalCharges is float
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(
                df["TotalCharges"].astype(str).str.strip(), errors="coerce"
            ).fillna(0.0)

        # 1. Tenure Group Segmentation
        # New customers (0-12m) have highest churn risk; long-tenure (>48m) have loyalty inertia
        if "tenure" in df.columns:
            df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0)
            bins = [-1, 12, 24, 48, 60, 100]
            labels = ["0-12m", "13-24m", "25-48m", "49-60m", ">60m"]
            df["tenure_group"] = pd.cut(df["tenure"], bins=bins, labels=labels).astype(str)
        else:
            df["tenure_group"] = "0-12m"

        # 2. Total Services Count
        # Counts how many value-added services the customer uses (OnlineSecurity, Backup, etc.)
        def count_services(row):
            cnt = 0
            for col in self.service_columns:
                if col in row and str(row[col]).strip().lower() == "yes":
                    cnt += 1
            return cnt

        df["total_services"] = df.apply(count_services, axis=1)

        # 3. Monthly to Total Charges / Average spend progression
        # Indicates if monthly spend is higher than their historical average rate
        if "tenure" in df.columns and "MonthlyCharges" in df.columns and "TotalCharges" in df.columns:
            monthly = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(0.0)
            tenure_safe = df["tenure"].replace(0, 1)
            expected_total = monthly * tenure_safe
            actual_total = df["TotalCharges"]
            # Ratio of actual total paid vs estimated total
            df["charges_ratio"] = np.where(
                expected_total > 0,
                (actual_total / expected_total).clip(0.1, 2.5),
                1.0
            )
            # Spend rate per month of tenure
            df["charges_per_tenure"] = monthly / tenure_safe
        else:
            df["charges_ratio"] = 1.0
            df["charges_per_tenure"] = 0.0

        # 4. Has Family Support (Partner & Dependents)
        if "Partner" in df.columns and "Dependents" in df.columns:
            has_p = df["Partner"].astype(str).str.lower() == "yes"
            has_d = df["Dependents"].astype(str).str.lower() == "yes"
            df["has_family_support"] = np.where(has_p & has_d, "Yes", "No")
        else:
            df["has_family_support"] = "No"

        # 5. Streaming Bundle (Both TV and Movies)
        if "StreamingTV" in df.columns and "StreamingMovies" in df.columns:
            tv = df["StreamingTV"].astype(str).str.lower() == "yes"
            movies = df["StreamingMovies"].astype(str).str.lower() == "yes"
            df["streaming_bundle"] = np.where(tv & movies, "Yes", "No")
        else:
            df["streaming_bundle"] = "No"

        return df


def explain_engineered_features() -> dict:
    """Return pedagogical rationale for each engineered feature."""
    return {
        "tenure_group": (
            "Categorizes continuous tenure into business lifecycle brackets (0-12m, 13-24m, 25-48m, 49-60m, >60m). "
            "Telecom customers experience highest churn hazard in their first year (onboarding friction / discount expiration). "
            "Discretization captures non-linear survival curve dynamics."
        ),
        "total_services": (
            "Integer sum (0-6) of subscribed add-on services (Online Security, Backup, Device Protection, Tech Support, Streaming TV, Streaming Movies). "
            "Higher service 'stickiness' increases switching costs and significantly reduces churn likelihood."
        ),
        "charges_ratio": (
            "Ratio of TotalCharges divided by (MonthlyCharges * tenure). Detects billing anomalies, past plan upgrades, or discounting adjustments."
        ),
        "charges_per_tenure": (
            "MonthlyCharges divided by tenure. Identifies newer high-expenditure customers who are particularly sensitive to rate hikes."
        ),
        "has_family_support": (
            "Compound binary indicator for customers with both Partner and Dependents. Household accounts show higher stability and lower attrition."
        ),
        "streaming_bundle": (
            "Indicates dual subscription to Streaming TV and Streaming Movies. Heavy content consumers have specific bandwidth demands and distinct churn patterns."
        )
    }


if __name__ == "__main__":
    fe = ChurnFeatureEngineer()
    sample = pd.DataFrame([{
        "tenure": 2,
        "MonthlyCharges": 70.35,
        "TotalCharges": 139.7,
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Partner": "Yes",
        "Dependents": "Yes"
    }])
    transformed = fe.transform(sample)
    print("Transformed columns:", transformed.columns.tolist())
    print("Engineered row:\n", transformed.to_dict(orient="records")[0])

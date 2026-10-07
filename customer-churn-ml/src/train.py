"""
Model Training & Optimization Module for Customer Churn
=======================================================
Trains multiple baseline classification algorithms:
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Gradient Boosting Classifier

Executes GridSearchCV hyperparameter tuning on top candidates.
Packages preprocessors and estimator into a unified deployment Pipeline.
Persists the final production model to joblib.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.data_preprocessing import load_raw_data, prepare_train_test
from src.feature_engineering import ChurnFeatureEngineer
from src.evaluate import evaluate_classifier


def create_preprocessor():
    """
    Constructs a ColumnTransformer that handles both numeric and categorical variables.
    Note: Feature engineering happens first, so transformed columns are handled here.
    """
    numeric_features = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "total_services",
        "charges_ratio",
        "charges_per_tenure",
    ]

    categorical_features = [
        "gender",
        "SeniorCitizen",
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
        "tenure_group",
        "has_family_support",
        "streaming_bundle",
    ]

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop"
    )

    return preprocessor


def train_baseline_models(X_train, y_train, X_test, y_test) -> Tuple[Dict[str, Any], Dict[str, Pipeline]]:
    """
    Trains 4 baseline models inside full preprocessing pipelines to prevent data leakage.
    Returns comparison metrics and trained pipelines dictionary.
    """
    # Instantiate candidates
    algorithms = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=6),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42, learning_rate=0.1),
    }

    results = {}
    pipelines = {}

    for name, clf in algorithms.items():
        pipe = Pipeline(steps=[
            ("feature_engineer", ChurnFeatureEngineer()),
            ("preprocessor", create_preprocessor()),
            ("classifier", clf)
        ])

        print(f"--> Training baseline model: {name}...")
        pipe.fit(X_train, y_train)

        metrics = evaluate_classifier(pipe, X_test, y_test, model_name=name)
        results[name] = metrics
        pipelines[name] = pipe

    return results, pipelines


def tune_best_model(X_train, y_train, X_test, y_test, model_name: str = "Gradient Boosting"):
    """
    Performs GridSearchCV with 5-fold Stratified K-Fold cross validation on the chosen model.
    """
    print(f"\n--> Starting Hyperparameter Tuning for: {model_name}...")

    if model_name == "Gradient Boosting":
        base_clf = GradientBoostingClassifier(random_state=42)
        param_grid = {
            "classifier__n_estimators": [75, 100, 150],
            "classifier__learning_rate": [0.03, 0.05, 0.1],
            "classifier__max_depth": [3, 4, 5],
            "classifier__min_samples_split": [4, 8],
            "classifier__subsample": [0.8, 1.0],
        }
    elif model_name == "Random Forest":
        base_clf = RandomForestClassifier(random_state=42, n_jobs=-1)
        param_grid = {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [8, 12, 16],
            "classifier__min_samples_split": [5, 10],
            "classifier__min_samples_leaf": [2, 4],
        }
    elif model_name == "Logistic Regression":
        base_clf = LogisticRegression(max_iter=1000, random_state=42)
        param_grid = {
            "classifier__C": [0.01, 0.1, 1.0, 5.0],
            "classifier__solver": ["lbfgs", "liblinear"],
        }
    else:
        base_clf = DecisionTreeClassifier(random_state=42)
        param_grid = {
            "classifier__max_depth": [4, 6, 8, 10],
            "classifier__min_samples_split": [5, 10, 20],
            "classifier__criterion": ["gini", "entropy"],
        }

    full_pipeline = Pipeline(steps=[
        ("feature_engineer", ChurnFeatureEngineer()),
        ("preprocessor", create_preprocessor()),
        ("classifier", base_clf)
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=full_pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    best_pipeline = grid_search.best_estimator_
    best_params = {k.replace("classifier__", ""): v for k, v in grid_search.best_params_.items()}
    best_cv_score = float(grid_search.best_score_)

    test_metrics = evaluate_classifier(best_pipeline, X_test, y_test, model_name=f"{model_name} (Tuned)")

    tuning_report = {
        "model_name": model_name,
        "best_params": best_params,
        "best_cv_roc_auc": round(best_cv_score, 4),
        "test_metrics": test_metrics
    }

    return best_pipeline, tuning_report


def extract_feature_importances(pipeline, X_train) -> list:
    """Extract named feature importances or coefficient magnitudes from the fitted pipeline."""
    # Transform sample to get feature names
    fe = pipeline.named_steps["feature_engineer"]
    X_fe = fe.transform(X_train.head(50))
    preprocessor = pipeline.named_steps["preprocessor"]

    # Get onehot feature names
    num_names = [
        "tenure", "MonthlyCharges", "TotalCharges",
        "total_services", "charges_ratio", "charges_per_tenure"
    ]
    try:
        cat_names = preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out().tolist()
        all_features = num_names + cat_names
    except Exception:
        all_features = [f"Feature_{i}" for i in range(40)]

    clf = pipeline.named_steps["classifier"]
    importance_list = []

    if hasattr(clf, "feature_importances_"):
        raw_importances = clf.feature_importances_
        for name, imp in zip(all_features, raw_importances):
            importance_list.append({"feature": name, "importance": round(float(imp), 4)})
    elif hasattr(clf, "coef_"):
        raw_coefs = np.abs(clf.coef_[0])
        for name, coef in zip(all_features, raw_coefs):
            importance_list.append({"feature": name, "importance": round(float(coef), 4)})

    importance_list.sort(key=lambda x: x["importance"], reverse=True)
    return importance_list[:15]


def save_pipeline(pipeline, filepath: str = "customer-churn-ml/models/customer_churn_model.joblib"):
    """Persist the trained end-to-end pipeline to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(pipeline, filepath)
    print(f"--> Saved production pipeline to: {filepath}")

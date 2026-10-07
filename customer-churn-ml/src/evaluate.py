"""
Evaluation Module for Customer Churn Models
===========================================
Calculates full classification metrics:
- Accuracy
- Precision (Positive Predictive Value: minimizing false retention alarms)
- Recall (Sensitivity: catching as many true churners as possible)
- F1-Score (Harmonic mean of Precision and Recall)
- ROC-AUC (Discrimination ability across all thresholds)
- Confusion Matrix (TP, FP, TN, FN)
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)


def evaluate_classifier(model, X_test, y_test, model_name: str = "Model") -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics for a trained estimator or pipeline.
    """
    y_pred = model.predict(X_test)

    # Compute probability if model supports it
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        # Sigmoid normalize
        y_proba = 1 / (1 + np.exp(-scores))
    else:
        y_proba = y_pred.astype(float)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    auc = float(roc_auc_score(y_test, y_proba))
    cm = confusion_matrix(y_test, y_pred).tolist()

    # ROC curve points (sampled for serialization)
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    # Sample down to 25 points for neat JSON transmission
    indices = np.linspace(0, len(fpr) - 1, min(25, len(fpr)), dtype=int)
    roc_points = [
        {"fpr": round(float(fpr[i]), 4), "tpr": round(float(tpr[i]), 4)}
        for i in indices
    ]

    return {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": {
            "tn": int(cm[0][0]),
            "fp": int(cm[0][1]),
            "fn": int(cm[1][0]),
            "tp": int(cm[1][1]),
        },
        "roc_points": roc_points,
    }


def explain_metrics() -> Dict[str, str]:
    """Pedagogical breakdown of classification metrics for Viva / college evaluation."""
    return {
        "Accuracy": (
            "Overall proportion of correct predictions: (TP + TN) / Total. "
            "In an imbalanced dataset (e.g. 73% non-churn, 27% churn), a naive model predicting 'No Churn' for everyone "
            "achieves 73.5% accuracy while being completely useless for retention. Therefore, accuracy alone is insufficient."
        ),
        "Precision": (
            "Proportion of predicted churners who actually churned: TP / (TP + FP). "
            "High precision prevents spending expensive retention budgets (e.g. $100 loyalty credits or handset subsidies) "
            "on satisfied customers who were never going to leave."
        ),
        "Recall": (
            "Proportion of actual churners correctly detected: TP / (TP + FN). "
            "In customer retention, missing a churner (False Negative) means permanently losing customer lifetime value (LTV). "
            "Recall measures how thoroughly the company catches departing accounts."
        ),
        "F1-Score": (
            "Harmonic mean of Precision and Recall: 2 * (P * R) / (P + R). "
            "Balances the tradeoff between budget waste (False Positives) and customer loss (False Negatives). "
            "Crucial metric for selecting models on skewed target distributions."
        ),
        "ROC-AUC": (
            "Area Under the Receiver Operating Characteristic Curve. Evaluates the classifier's ability to rank churners "
            "higher than non-churners across all possible decision thresholds (0.0 to 1.0). An AUC of 0.84+ indicates strong discriminative power."
        ),
        "Confusion Matrix": (
            "2x2 contingency table contrasting Actual vs. Predicted outcomes: True Negatives (retained kept), "
            "False Positives (retained flagged as churn), False Negatives (churners missed), True Positives (churners caught)."
        ),
    }

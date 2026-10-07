"""
Inference & Prediction Module for Customer Churn
================================================
Loads the persisted Joblib pipeline and makes predictions on unseen customer data.
Strictly avoids any retraining during inference.
Provides human-readable risk diagnostics and business retention recommendations.
"""

import os
import sys
import joblib
import pandas as pd
from typing import Dict, Any, Union, List

# Ensure parent directory is in sys.path so 'src.*' imports work from any cwd
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.currency import INR_TO_USD_RATE, convert_inr_to_usd, format_inr

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "customer_churn_model.joblib")


def load_model(filepath: str = MODEL_PATH):
    """Load persisted scikit-learn pipeline from joblib file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Saved model pipeline not found at {filepath}. "
            "Please run `python -m src.train` or `python run_all.py` first to train and persist the model."
        )
    model = joblib.load(filepath)
    return model


def normalize_charges(customer_data: dict) -> tuple[dict, bool, float, float, float, float]:
    """
    Intelligently handles INR (Indian Rupee) and USD amounts.
    The IBM Telco model was trained on USD amounts (MonthlyCharges 18.25 - 118.75).
    Converts INR inputs to USD using the single configurable INR_TO_USD_RATE
    (formula: USD = INR * INR_TO_USD_RATE).
    """
    cleaned = dict(customer_data)

    if "MonthlyCharges_USD" in cleaned and "TotalCharges_USD" in cleaned:
        monthly_usd = round(float(cleaned["MonthlyCharges_USD"]), 2)
        total_usd = round(float(cleaned["TotalCharges_USD"]), 2)
        monthly_inr = float(cleaned.get("MonthlyCharges_INR", cleaned.get("MonthlyCharges", round(monthly_usd / INR_TO_USD_RATE, 2))))
        total_inr = float(cleaned.get("TotalCharges_INR", cleaned.get("TotalCharges", round(total_usd / INR_TO_USD_RATE, 2))))
        is_inr = True
    else:
        raw_monthly = float(cleaned.get("MonthlyCharges", 0.0) or 0.0)
        raw_total = float(cleaned.get("TotalCharges", 0.0) or 0.0)
        is_inr = cleaned.get("currency") == "INR" or cleaned.get("is_inr") is True or raw_monthly > 150.0 or raw_total > 5000.0

        if is_inr:
            monthly_inr = raw_monthly
            total_inr = raw_total
            monthly_usd = convert_inr_to_usd(raw_monthly)
            total_usd = convert_inr_to_usd(raw_total)
        else:
            monthly_usd = round(raw_monthly, 2)
            total_usd = round(raw_total, 2)
            monthly_inr = round(raw_monthly / INR_TO_USD_RATE, 2)
            total_inr = round(raw_total / INR_TO_USD_RATE, 2)

    # Ensure model receives non-negative floats in USD
    cleaned["MonthlyCharges"] = max(0.0, monthly_usd)
    cleaned["TotalCharges"] = max(0.0, total_usd)

    # Handle SeniorCitizen if passed as string "Yes"/"No"
    if "SeniorCitizen" in cleaned:
        sc = cleaned["SeniorCitizen"]
        if str(sc).strip().lower() in ("yes", "1", "true"):
            cleaned["SeniorCitizen"] = 1
        elif str(sc).strip().lower() in ("no", "0", "false"):
            cleaned["SeniorCitizen"] = 0

    return cleaned, is_inr, monthly_inr, total_inr, monthly_usd, total_usd


def generate_explanation(customer_data: dict, proba: float, is_inr: bool = False, orig_monthly: float = 0.0) -> dict:
    """
    Generate human-interpretable risk drivers and actionable retention interventions.
    """
    risk_factors = []
    protective_factors = []
    recommended_actions = []

    tenure = float(customer_data.get("tenure", 0))
    contract = str(customer_data.get("Contract", ""))
    monthly_usd = float(customer_data.get("MonthlyCharges", 0))
    curr_symbol = "₹" if is_inr else "$"
    display_monthly = orig_monthly if is_inr else monthly_usd

    # Evaluate risk contributors
    if contract == "Month-to-month":
        risk_factors.append("Month-to-month commitment provides zero switching barriers.")
        rec_disc = "₹500/month discount" if is_inr else "15% discount"
        recommended_actions.append(f"Offer a {rec_disc} for upgrading to a 1-year or 2-year annual agreement.")
    else:
        protective_factors.append(f"Long-term commitment ({contract}) creates high retention stability.")

    if tenure <= 6:
        risk_factors.append(f"Early lifecycle customer ({tenure:.0f} months tenure); high initial hazard period.")
        recommended_actions.append("Schedule a proactive onboarding concierge call and satisfaction check.")
    elif tenure >= 36:
        protective_factors.append(f"High customer tenure ({tenure:.0f} months) reflects established brand loyalty.")

    if monthly_usd >= 75.0:
        risk_factors.append(f"High monthly billing ({curr_symbol}{display_monthly:,.2f}/mo); customer sensitive to price-to-value.")
        recommended_actions.append("Review billing plan for bundled loyalty discount or service credit.")
    else:
        protective_factors.append(f"Economical monthly bill ({curr_symbol}{display_monthly:,.2f}/mo).")

    internet = str(customer_data.get("InternetService", ""))
    tech_support = str(customer_data.get("TechSupport", ""))
    online_sec = str(customer_data.get("OnlineSecurity", ""))
    payment = str(customer_data.get("PaymentMethod", ""))

    if internet == "Fiber optic" and (tech_support == "No" or online_sec == "No"):
        risk_factors.append("Fiber optic user without Tech Support/Online Security (common source of frustration).")
        recommended_actions.append("Provide complimentary Tech Support + Security suite for 3 months.")

    if "Electronic check" in payment:
        risk_factors.append("Payment via Electronic Check exhibits highest historical churn correlation.")
        rec_pay = "₹250 monthly credit" if is_inr else "$5 monthly credit"
        recommended_actions.append(f"Incentivize enrollment in Credit Card or Bank Transfer Auto-Pay ({rec_pay}).")
    elif "automatic" in payment.lower():
        protective_factors.append("Automatic payment method enabled (reduces billing friction).")

    risk_level = "High" if proba >= 0.50 else "Low"

    return {
        "risk_level": risk_level,
        "key_risk_drivers": risk_factors,
        "retention_anchors": protective_factors,
        "recommended_interventions": recommended_actions or ["The customer currently shows a lower risk of leaving."],
    }


def predict_single_customer(customer_data: dict, model=None) -> Dict[str, Any]:
    """
    Accepts raw customer demographic & account dictionary,
    executes inference through the saved pipeline, and returns predictions.
    """
    if model is None:
        model = load_model()

    # Normalize currency from INR to USD before passing to model
    model_input, is_inr, monthly_inr, total_inr, monthly_usd, total_usd = normalize_charges(customer_data)

    # Convert dictionary into a single-row DataFrame
    df_input = pd.DataFrame([model_input])

    # Strip any extra metadata keys before pipeline transform
    cols_to_drop = [
        c for c in [
            "currency", "is_inr", "MonthlyCharges_USD", "TotalCharges_USD",
            "MonthlyCharges_INR", "TotalCharges_INR", "original_inr", "converted_usd",
            "exchange_rate", "id", "profile_name"
        ] if c in df_input.columns
    ]
    if cols_to_drop:
        df_input = df_input.drop(columns=cols_to_drop)

    # Run pipeline inference (feature engineering + preprocessing + classification)
    pred_class = int(model.predict(df_input)[0])
    probabilities = model.predict_proba(df_input)[0]
    churn_proba = float(probabilities[1])

    is_churn = pred_class == 1 or churn_proba >= 0.50
    churn_label = "Churn" if is_churn else "No Churn"
    human_verdict = (
        "Customer is likely to churn"
        if is_churn
        else "Customer is likely to stay"
    )

    explanation = generate_explanation(model_input, churn_proba, is_inr, monthly_inr)

    rec_text = (
        "This customer may require a retention offer or additional customer support."
        if is_churn
        else "The customer currently shows a lower risk of leaving."
    )

    return {
        "prediction": churn_label,
        "verdict": human_verdict,
        "churn_probability": round(churn_proba, 4),
        "churn_probability_percentage": f"{churn_proba * 100:.1f}%",
        "stay_probability_percentage": f"{(1 - churn_proba) * 100:.1f}%",
        "risk_level": "High" if is_churn else "Low",
        "recommendation": rec_text,
        "is_inr": is_inr,
        "inr_inputs": {
            "MonthlyCharges": monthly_inr,
            "TotalCharges": total_inr,
            "formatted_monthly": format_inr(monthly_inr),
            "formatted_total": format_inr(total_inr),
        },
        "converted_usd": {
            "MonthlyCharges": monthly_usd,
            "TotalCharges": total_usd,
            "formatted_monthly": f"${monthly_usd:.2f}",
            "formatted_total": f"${total_usd:.2f}",
        },
        "exchange_rate": INR_TO_USD_RATE,
        "diagnostics": explanation,
    }


def predict_batch(customers: List[dict], model=None) -> List[Dict[str, Any]]:
    """Predict churn for a batch of customer profiles."""
    if model is None:
        model = load_model()
    return [predict_single_customer(c, model) for c in customers]


# Unseen Realistic Customer Test Profiles (Required by College Specs)
UNSEEN_CUSTOMERS = [
    {
        "id": "UNSEEN-001",
        "profile_name": "High-Risk New Digital Streamer",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.85,
        "TotalCharges": 179.70,
    },
    {
        "id": "UNSEEN-002",
        "profile_name": "Low-Risk Long-Term Family Account",
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 65,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 64.80,
        "TotalCharges": 4212.00,
    },
    {
        "id": "UNSEEN-003",
        "profile_name": "Moderate-Risk Mid-Tenure Solo Professional",
        "gender": "Male",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 18,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 84.40,
        "TotalCharges": 1519.20,
    },
]


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) > 1 and sys.argv[1] == "--json":
        raw_json = sys.argv[2] if len(sys.argv) > 2 else sys.stdin.read()
        try:
            cust_dict = json.loads(raw_json)
            model = load_model()
            res = predict_single_customer(cust_dict, model)
            print(json.dumps(res))
        except Exception as e:
            print(json.dumps({"error": str(e)}))
        sys.exit(0)

    print("==================================================")
    print("RUNNING INFERENCE ON REALISTIC UNSEEN CUSTOMERS")
    print("==================================================")
    try:
        model = load_model()
        for cust in UNSEEN_CUSTOMERS:
            res = predict_single_customer(cust, model)
            print(f"\n--- Customer: {cust['id']} ({cust['profile_name']}) ---")
            print(f"Prediction: {res['prediction']}")
            print(f"Churn Probability: {res['churn_probability_percentage']}")
            print(f"Risk Level: {res['diagnostics']['risk_level']}")
            print(f"Verdict: {res['verdict']}")
            print(f"Key Drivers: {', '.join(res['diagnostics']['key_risk_drivers'])}")
            print(f"Recommended Action: {res['diagnostics']['recommended_interventions'][0]}")
    except Exception as e:
        print(f"Inference error: {e}")

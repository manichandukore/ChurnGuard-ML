"""
Currency conversion configuration and utilities for Customer Churn Prediction.
=============================================================================
Keeps the ML model compatible with the original IBM Telco dataset (USD)
while allowing user interface and inputs in Indian Rupees (INR).

Conversion Formula:
USD = INR * INR_TO_USD_RATE
"""

import os

# Configurable INR to USD Exchange Rate Setting
# Default rate converts ₹8,700 -> ~$90.00 USD (1 USD ≈ 96.67 INR)
# as specified in the test case requirements.
# Can be configured via the INR_TO_USD_RATE environment variable.
FALLBACK_INR_TO_USD_RATE = 0.010345
INR_TO_USD_RATE = float(os.environ.get("INR_TO_USD_RATE", FALLBACK_INR_TO_USD_RATE))


def convert_inr_to_usd(inr_amount: float) -> float:
    """
    Utility function to convert INR amount to USD using the single configurable rate.
    Formula: USD = INR * INR_TO_USD_RATE
    """
    if inr_amount is None:
        return 0.0
    val = float(inr_amount)
    if val < 0:
        val = 0.0
    return round(val * INR_TO_USD_RATE, 2)


def format_inr(amount: float) -> str:
    """Format positive numeric value in Indian number formatting with ₹ symbol."""
    try:
        val = int(round(float(amount)))
        # Format using Indian numbering system (xx,xx,xxx)
        s = str(abs(val))
        if len(s) <= 3:
            formatted = s
        else:
            last3 = s[-3:]
            remaining = s[:-3]
            groups = []
            while len(remaining) > 2:
                groups.insert(0, remaining[-2:])
                remaining = remaining[:-2]
            if remaining:
                groups.insert(0, remaining)
            formatted = ",".join(groups) + "," + last3
        return f"₹{formatted}"
    except Exception:
        return f"₹{amount}"

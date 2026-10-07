/**
 * Currency conversion utilities for Customer Churn Prediction.
 * Configurable INR to USD Exchange Rate Setting:
 * Formula: USD = INR * INR_TO_USD_RATE
 *
 * Default: 0.010345 (1 USD ≈ 96.67 INR)
 * Example: ₹8,700 -> ~$90.00 USD, ₹17,400 -> ~$180.00 USD
 */

export const INR_TO_USD_RATE = 0.010345;

/**
 * Converts an INR amount to USD using the single configurable exchange rate.
 * USD = INR * INR_TO_USD_RATE
 */
export function convert_inr_to_usd(inr_amount: number): number {
  if (typeof inr_amount !== "number" || isNaN(inr_amount) || inr_amount < 0) {
    return 0;
  }
  return Number((inr_amount * INR_TO_USD_RATE).toFixed(2));
}

/**
 * Formats a numeric amount using Indian number formatting (e.g. ₹8,700, ₹17,400, ₹1,25,000).
 */
export function formatINR(amount: number): string {
  if (typeof amount !== "number" || isNaN(amount)) {
    return "₹0";
  }
  const rounded = Math.round(amount);
  return `₹${rounded.toLocaleString("en-IN")}`;
}

/**
 * Formats a USD amount (e.g. $90.00, $180.00).
 */
export function formatUSD(amount: number): string {
  if (typeof amount !== "number" || isNaN(amount)) {
    return "$0.00";
  }
  return `$${amount.toFixed(2)}`;
}

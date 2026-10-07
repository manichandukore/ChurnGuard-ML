import React, { useState } from "react";
import {
  Shield,
  TrendingUp,
  Info,
  Home,
  User,
  CreditCard,
  Clock,
  Target,
  AlertTriangle,
  CheckCircle2,
  Brain,
  Zap,
  ArrowRight,
  Wifi,
  Sparkles,
  Loader2,
  Lock,
  Star,
  Package,
  ShoppingBag,
  BarChart3,
  PieChart,
  ShieldCheck,
  RotateCcw,
} from "lucide-react";

import {
  INR_TO_USD_RATE,
  convert_inr_to_usd,
  formatINR,
  formatUSD,
} from "./utils/currency";

export interface FormState {
  gender: "Male" | "Female";
  seniorCitizen: "No" | "Yes";
  partner: "No" | "Yes";
  dependents: "No" | "Yes";
  tenure: number | string;
  phoneService: "No" | "Yes";
  multipleLines: "No" | "Yes" | "No phone service";
  internetService: "DSL" | "Fiber optic" | "No";
  onlineSecurity: "No" | "Yes" | "No internet service";
  onlineBackup: "No" | "Yes" | "No internet service";
  deviceProtection: "No" | "Yes" | "No internet service";
  techSupport: "No" | "Yes" | "No internet service";
  streamingTV: "No" | "Yes" | "No internet service";
  streamingMovies: "No" | "Yes" | "No internet service";
  contract: "Month-to-month" | "One year" | "Two year";
  paperlessBilling: "No" | "Yes";
  paymentMethod:
    | "Electronic check"
    | "Mailed check"
    | "Bank transfer (automatic)"
    | "Credit card (automatic)";
  monthlyCharges: number | string;
  totalCharges: number | string;
}

export interface PredictionData {
  prediction: "Churn" | "No Churn";
  churn_probability: number;
  churn_probability_percentage: string;
  stay_probability_percentage: string;
  risk_level: "High" | "Low";
  verdict: string;
  recommendation: string;
  factors: Array<{
    icon: string;
    label: string;
  }>;
  inr_inputs?: {
    MonthlyCharges: number;
    TotalCharges: number;
    formatted_monthly: string;
    formatted_total: string;
  };
  converted_usd?: {
    MonthlyCharges: number;
    TotalCharges: number;
    formatted_monthly: string;
    formatted_total: string;
  };
}

// Initial form state matching the specification test case (₹8,700 Monthly, ₹17,400 Total, 3 tenure)
const INITIAL_FORM: FormState = {
  gender: "Male",
  seniorCitizen: "No",
  partner: "Yes",
  dependents: "Yes",
  tenure: 3,
  phoneService: "Yes",
  multipleLines: "No",
  internetService: "Fiber optic",
  onlineSecurity: "Yes",
  onlineBackup: "No",
  deviceProtection: "Yes",
  techSupport: "No",
  streamingTV: "Yes",
  streamingMovies: "Yes",
  contract: "Month-to-month",
  paperlessBilling: "Yes",
  paymentMethod: "Electronic check",
  monthlyCharges: 8700,
  totalCharges: 17400,
};

// Initial prediction result evaluated from the saved model on the test case
const INITIAL_PREDICTION: PredictionData = {
  prediction: "Churn",
  churn_probability: 0.6792,
  churn_probability_percentage: "67.9%",
  stay_probability_percentage: "32.1%",
  risk_level: "High",
  verdict: "Customer is likely to churn.",
  recommendation:
    "This customer may require a retention offer or additional customer support.",
  factors: [
    { icon: "contract", label: "Month-to-month Contract" },
    { icon: "charges", label: "High Monthly Charges (₹8,700)" },
    { icon: "internet", label: "Fiber Optic Service" },
    { icon: "tenure", label: "Short Tenure (3 months)" },
  ],
  inr_inputs: {
    MonthlyCharges: 8700,
    TotalCharges: 17400,
    formatted_monthly: "₹8,700",
    formatted_total: "₹17,400",
  },
  converted_usd: {
    MonthlyCharges: 90.0,
    TotalCharges: 180.0,
    formatted_monthly: "$90.00",
    formatted_total: "$180.00",
  },
};

// Helpers to prevent leading zero issues like "090" when typing
function cleanNumberInput(val: string): string {
  if (!val) return "";
  // Strip any characters other than digits and decimal point
  let cleaned = val.replace(/[^0-9.]/g, "");
  const parts = cleaned.split(".");
  if (parts.length > 2) {
    cleaned = parts[0] + "." + parts.slice(1).join("");
  }
  // Strip leading zeroes if followed by another digit (e.g., "090" -> "90", "05" -> "5")
  // but keep "0", "0.", "0.xx"
  if (/^0\d+/.test(cleaned)) {
    cleaned = cleaned.replace(/^0+/, "") || "0";
  }
  return cleaned;
}

function cleanTenureInput(val: string): string {
  if (!val) return "";
  let cleaned = val.replace(/[^0-9]/g, "");
  // Strip leading zeroes (e.g. "03" -> "3", "090" -> "72")
  if (cleaned.length > 1 && cleaned.startsWith("0")) {
    cleaned = cleaned.replace(/^0+/, "") || "0";
  }
  if (cleaned !== "") {
    const num = parseInt(cleaned, 10);
    if (num > 72) return "72";
  }
  return cleaned;
}

export default function App() {
  const [currentPage, setCurrentPage] = useState<"home" | "predict" | "about">(
    "home"
  );
  const [currency, setCurrency] = useState<"USD" | "INR">("INR");
  const [formData, setFormData] = useState<FormState>(INITIAL_FORM);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [predictionResult, setPredictionResult] =
    useState<PredictionData>(INITIAL_PREDICTION);

  const currencySymbol = currency === "USD" ? "$" : "₹";

  // Handle direct manual input change - never overwrite TotalCharges automatically
  const handleInputChange = (field: keyof FormState, value: any) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
  };

  const handleTenureChange = (val: string) => {
    handleInputChange("tenure", cleanTenureInput(val));
  };

  const handleMonthlyChargesChange = (val: string) => {
    handleInputChange("monthlyCharges", cleanNumberInput(val));
  };

  const handleTotalChargesChange = (val: string) => {
    handleInputChange("totalCharges", cleanNumberInput(val));
  };

  const handleResetForm = () => {
    setFormData(INITIAL_FORM);
    setCurrency("INR");
  };

  // Toggle between USD and INR with accurate conversion
  const handleCurrencyToggle = (newCurrency: "USD" | "INR") => {
    if (newCurrency === currency) return;
    setCurrency(newCurrency);
    setFormData((prev) => {
      const mStr = String(prev.monthlyCharges ?? "").trim();
      const tStr = String(prev.totalCharges ?? "").trim();
      const m = mStr === "" ? "" : Number(mStr);
      const t = tStr === "" ? "" : Number(tStr);

      if (newCurrency === "INR") {
        // Convert USD to INR
        const inrM = m === "" || isNaN(m as number) ? "" : String(Math.round((m as number) / INR_TO_USD_RATE));
        const inrT = t === "" || isNaN(t as number) ? "" : String(Math.round((t as number) / INR_TO_USD_RATE));
        return {
          ...prev,
          monthlyCharges: inrM,
          totalCharges: inrT,
        };
      } else {
        // Convert INR to USD
        const usdM = m === "" || isNaN(m as number) ? "" : String(Number(((m as number) * INR_TO_USD_RATE).toFixed(2)));
        const usdT = t === "" || isNaN(t as number) ? "" : String(Number(((t as number) * INR_TO_USD_RATE).toFixed(2)));
        return {
          ...prev,
          monthlyCharges: usdM,
          totalCharges: usdT,
        };
      }
    });
  };

  // Submit prediction to the API
  const handlePredict = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    setIsLoading(true);

    try {
      const rawMonthly = Number(formData.monthlyCharges) || 0;
      const rawTotal = Number(formData.totalCharges) || 0;

      // Backend supports INR or USD
      const payload = {
        gender: formData.gender,
        SeniorCitizen: formData.seniorCitizen === "Yes" ? 1 : 0,
        Partner: formData.partner,
        Dependents: formData.dependents,
        tenure: Number(formData.tenure),
        PhoneService: formData.phoneService,
        MultipleLines: formData.multipleLines,
        InternetService: formData.internetService,
        OnlineSecurity: formData.onlineSecurity,
        OnlineBackup: formData.onlineBackup,
        DeviceProtection: formData.deviceProtection,
        TechSupport: formData.techSupport,
        StreamingTV: formData.streamingTV,
        StreamingMovies: formData.streamingMovies,
        Contract: formData.contract,
        PaperlessBilling: formData.paperlessBilling,
        PaymentMethod: formData.paymentMethod,
        MonthlyCharges: currency === "INR" ? rawMonthly : rawMonthly,
        TotalCharges: currency === "INR" ? rawTotal : rawTotal,
        currency: currency,
        is_inr: currency === "INR",
      };

      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        throw new Error(`Server returned HTTP ${res.status}`);
      }

      const data = await res.json();

      // Extract dynamic factors
      const dynamicFactors: Array<{ icon: string; label: string }> = [];
      if (formData.contract === "Month-to-month") {
        dynamicFactors.push({ icon: "contract", label: "Month-to-month Contract" });
      } else {
        dynamicFactors.push({ icon: "contract", label: `${formData.contract} Contract` });
      }

      const usdMonthly = currency === "INR" ? convert_inr_to_usd(rawMonthly) : rawMonthly;
      const displayCharges = currency === "INR" ? formatINR(rawMonthly) : formatUSD(rawMonthly);
      if (usdMonthly > 70) {
        dynamicFactors.push({ icon: "charges", label: `High Monthly Charges (${displayCharges})` });
      } else {
        dynamicFactors.push({ icon: "charges", label: `Standard Monthly Charges (${displayCharges})` });
      }

      if (formData.internetService === "Fiber optic") {
        dynamicFactors.push({ icon: "internet", label: "Fiber Optic Service" });
      } else if (formData.internetService === "DSL") {
        dynamicFactors.push({ icon: "internet", label: "DSL Internet Service" });
      }

      const tenureNum = Number(formData.tenure) || 0;
      if (tenureNum <= 6) {
        dynamicFactors.push({ icon: "tenure", label: `Short Tenure (${tenureNum} months)` });
      } else if (tenureNum >= 36) {
        dynamicFactors.push({ icon: "tenure", label: `Long Tenure (${tenureNum} months)` });
      }

      setPredictionResult({
        prediction: data.prediction || (data.churn_probability >= 0.5 ? "Churn" : "No Churn"),
        churn_probability: data.churn_probability,
        churn_probability_percentage: data.churn_probability_percentage,
        stay_probability_percentage: data.stay_probability_percentage,
        risk_level: data.risk_level,
        verdict:
          data.prediction === "Churn"
            ? "This customer is likely to churn."
            : "This customer is likely to stay.",
        recommendation:
          data.recommendation ||
          (data.prediction === "Churn"
            ? "Consider offering a retention plan, personalized offers, or additional customer support to reduce churn risk."
            : "The customer currently shows a lower risk of leaving."),
        factors: dynamicFactors.length > 0 ? dynamicFactors : INITIAL_PREDICTION.factors,
        inr_inputs: data.inr_inputs,
        converted_usd: data.converted_usd,
      });
    } catch (err) {
      console.error("Prediction error:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const isChurn = predictionResult.prediction === "Churn";

  return (
    <div className="min-h-screen bg-[#f8fafc] text-slate-800 flex flex-col font-sans">
      {/* ======================================================== */}
      {/* 1. TOP HEADER (Exact ChurnGuard Design from Screenshot) */}
      {/* ======================================================== */}
      <header className="bg-[#0b132b] text-white sticky top-0 z-50 shadow-md">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          {/* Logo & Subtitle */}
          <div
            onClick={() => setCurrentPage("home")}
            className="flex items-center gap-3 cursor-pointer group"
          >
            <div className="w-10 h-10 rounded-xl bg-blue-600 flex items-center justify-center shadow-md shadow-blue-500/30">
              <Shield className="w-5 h-5 text-white fill-white/20" />
            </div>
            <div>
              <div className="font-bold text-lg tracking-tight text-white">
                ChurnGuard
              </div>
              <p className="text-[11px] text-slate-400 -mt-0.5">
                Customer Churn Prediction
              </p>
            </div>
          </div>

          {/* Navigation Buttons + Currency Selector */}
          <div className="flex items-center gap-2 sm:gap-3">
            {/* Currency Pill Switcher */}
            <div className="flex items-center bg-slate-800/80 rounded-lg p-0.5 border border-slate-700/80 text-xs">
              <button
                type="button"
                onClick={() => handleCurrencyToggle("INR")}
                className={`px-2.5 py-1 rounded-md font-medium transition-colors ${
                  currency === "INR"
                    ? "bg-blue-600 text-white shadow-sm"
                    : "text-slate-400 hover:text-white"
                }`}
              >
                ₹ INR
              </button>
              <button
                type="button"
                onClick={() => handleCurrencyToggle("USD")}
                className={`px-2.5 py-1 rounded-md font-medium transition-colors ${
                  currency === "USD"
                    ? "bg-blue-600 text-white shadow-sm"
                    : "text-slate-400 hover:text-white"
                }`}
              >
                $ USD
              </button>
            </div>

            {/* 3 Main Tabs: Home, Predict Churn, About */}
            <nav className="flex items-center gap-1.5 sm:gap-2">
              <button
                onClick={() => setCurrentPage("home")}
                className={`flex items-center gap-2 px-3.5 py-2 rounded-lg text-sm font-medium transition-colors ${
                  currentPage === "home"
                    ? "bg-blue-600 text-white shadow-sm shadow-blue-600/30"
                    : "text-slate-300 hover:text-white hover:bg-slate-800/60"
                }`}
              >
                <Home className="w-4 h-4" />
                <span>Home</span>
              </button>

              <button
                onClick={() => setCurrentPage("predict")}
                className={`flex items-center gap-2 px-3.5 py-2 rounded-lg text-sm font-medium transition-colors ${
                  currentPage === "predict"
                    ? "bg-blue-600 text-white shadow-sm shadow-blue-600/30"
                    : "text-slate-300 hover:text-white hover:bg-slate-800/60"
                }`}
              >
                <TrendingUp className="w-4 h-4" />
                <span>Predict Churn</span>
              </button>

              <button
                onClick={() => setCurrentPage("about")}
                className={`flex items-center gap-2 px-3.5 py-2 rounded-lg text-sm font-medium transition-colors ${
                  currentPage === "about"
                    ? "bg-blue-600 text-white shadow-sm shadow-blue-600/30"
                    : "text-slate-300 hover:text-white hover:bg-slate-800/60"
                }`}
              >
                <Info className="w-4 h-4" />
                <span>About</span>
              </button>
            </nav>
          </div>
        </div>
      </header>

      {/* ======================================================== */}
      {/* 2. BODY CONTENT (HOME / PREDICT / ABOUT) */}
      {/* ======================================================== */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-10">
        {/* ======================================================== */}
        {/* VIEW 1: HOME PAGE (Exact Left Screenshot) */}
        {/* ======================================================== */}
        {currentPage === "home" && (
          <div className="max-w-4xl mx-auto py-4 sm:py-8 space-y-12">
            {/* Main Hero Header */}
            <div className="text-center space-y-3">
              <h1 className="text-4xl sm:text-6xl font-black text-slate-900 tracking-tight">
                ChurnGuard
              </h1>
              <h2 className="text-2xl sm:text-4xl font-bold bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 bg-clip-text text-transparent">
                Customer Churn Prediction
              </h2>
              <p className="text-slate-600 text-base sm:text-lg max-w-xl mx-auto pt-1 leading-relaxed">
                Predict whether a customer is likely to leave the service using our
                trained Machine Learning model.
              </p>
            </div>

            {/* Center Illustrated Shield with floating charts */}
            <div className="relative py-8 flex items-center justify-center">
              {/* Soft radial glow */}
              <div className="absolute w-80 h-80 rounded-full bg-blue-100/70 blur-3xl pointer-events-none" />
              <div className="absolute w-64 h-64 rounded-full bg-indigo-100/60 blur-2xl pointer-events-none" />

              {/* Floating Left Chart Badge */}
              <div className="absolute left-1/4 -translate-x-12 sm:-translate-x-16 top-1/2 -translate-y-8 w-16 h-16 rounded-full bg-white/90 border border-slate-100 shadow-xl shadow-slate-200/50 flex items-center justify-center">
                <BarChart3 className="w-8 h-8 text-indigo-400" />
              </div>

              {/* Center Main Shield Icon */}
              <div className="relative z-10 w-44 h-48 sm:w-52 sm:h-56 rounded-3xl bg-gradient-to-b from-blue-500 via-indigo-600 to-blue-700 flex items-center justify-center shadow-2xl shadow-blue-500/30 transform hover:scale-105 transition-transform duration-300">
                <div className="w-20 h-20 sm:w-24 sm:h-24 rounded-full bg-white flex items-center justify-center shadow-inner">
                  <User className="w-12 h-12 sm:w-14 sm:h-14 text-blue-600" />
                </div>
              </div>

              {/* Floating Right Chart Badge */}
              <div className="absolute right-1/4 translate-x-12 sm:translate-x-16 top-1/2 -translate-y-8 w-16 h-16 rounded-full bg-white/90 border border-slate-100 shadow-xl shadow-slate-200/50 flex items-center justify-center">
                <PieChart className="w-8 h-8 text-emerald-400" />
              </div>
            </div>

            {/* Big Gradient "Predict Customer Churn" Action Button */}
            <div className="text-center pt-2">
              <button
                type="button"
                onClick={() => setCurrentPage("predict")}
                className="inline-flex items-center gap-3 px-8 py-4 rounded-2xl bg-gradient-to-r from-blue-600 via-indigo-600 to-blue-600 hover:from-blue-700 hover:to-indigo-700 text-white font-bold text-lg shadow-xl shadow-blue-500/25 transition-all hover:scale-105 active:scale-95 group"
              >
                <span>Predict Customer Churn</span>
                <div className="w-7 h-7 rounded-full bg-white text-blue-600 flex items-center justify-center font-black group-hover:translate-x-0.5 transition-transform">
                  →
                </div>
              </button>
            </div>

            {/* "Why Choose ChurnGuard?" Section */}
            <div className="pt-10 space-y-6">
              <h3 className="text-2xl font-bold text-slate-900 text-center tracking-tight">
                Why Choose ChurnGuard?
              </h3>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {/* 1. Fast Prediction */}
                <div className="bg-white rounded-2xl border border-slate-200/80 shadow-sm p-6 text-center space-y-3 hover:shadow-md transition-shadow">
                  <div className="w-14 h-14 rounded-full bg-blue-50 text-blue-600 flex items-center justify-center mx-auto">
                    <Zap className="w-7 h-7" />
                  </div>
                  <h4 className="text-lg font-bold text-slate-900">Fast Prediction</h4>
                  <p className="text-xs text-slate-500 leading-relaxed">
                    Get instant churn predictions in seconds.
                  </p>
                </div>

                {/* 2. ML-Powered */}
                <div className="bg-white rounded-2xl border border-slate-200/80 shadow-sm p-6 text-center space-y-3 hover:shadow-md transition-shadow">
                  <div className="w-14 h-14 rounded-full bg-emerald-50 text-emerald-600 flex items-center justify-center mx-auto">
                    <Brain className="w-7 h-7" />
                  </div>
                  <h4 className="text-lg font-bold text-slate-900">ML-Powered</h4>
                  <p className="text-xs text-slate-500 leading-relaxed">
                    Trained on real customer data using Machine Learning.
                  </p>
                </div>

                {/* 3. No Retraining */}
                <div className="bg-white rounded-2xl border border-slate-200/80 shadow-sm p-6 text-center space-y-3 hover:shadow-md transition-shadow">
                  <div className="w-14 h-14 rounded-full bg-purple-50 text-purple-600 flex items-center justify-center mx-auto">
                    <ShieldCheck className="w-7 h-7" />
                  </div>
                  <h4 className="text-lg font-bold text-slate-900">No Retraining</h4>
                  <p className="text-xs text-slate-500 leading-relaxed">
                    Uses our optimized trained model for accurate results.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ======================================================== */}
        {/* VIEW 2: PREDICT CHURN PAGE (Exact Right Screenshot) */}
        {/* ======================================================== */}
        {currentPage === "predict" && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
            {/* Left Column: Form Card (7 cols) */}
            <div className="lg:col-span-7 bg-white rounded-2xl border border-slate-200/80 shadow-sm p-6 sm:p-8 space-y-8">
              {/* Form Title & Icon with Prominent Currency Selector */}
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-100">
                <div className="flex items-start gap-3.5">
                  <div className="w-10 h-10 rounded-xl bg-blue-50 text-blue-600 flex items-center justify-center shrink-0 mt-0.5">
                    <User className="w-5 h-5" />
                  </div>
                  <div>
                    <h2 className="text-xl font-bold text-slate-900 tracking-tight">
                      Predict Customer Churn
                    </h2>
                    <p className="text-xs text-slate-500 mt-0.5">
                      Enter customer details below to predict the likelihood of churn.
                    </p>
                  </div>
                </div>

                {/* Currency Selector and Reset */}
                <div className="flex items-center gap-1.5 self-start sm:self-auto bg-slate-100 p-1 rounded-xl border border-slate-200">
                  <span className="text-[11px] font-semibold text-slate-500 px-1.5">Currency:</span>
                  <button
                    type="button"
                    onClick={() => handleCurrencyToggle("INR")}
                    className={`px-3 py-1 rounded-lg text-xs font-bold transition-all ${
                      currency === "INR"
                        ? "bg-white text-blue-700 shadow-sm border border-slate-200/80"
                        : "text-slate-600 hover:text-slate-900"
                    }`}
                  >
                    ₹ INR
                  </button>
                  <button
                    type="button"
                    onClick={() => handleCurrencyToggle("USD")}
                    className={`px-3 py-1 rounded-lg text-xs font-bold transition-all ${
                      currency === "USD"
                        ? "bg-white text-blue-700 shadow-sm border border-slate-200/80"
                        : "text-slate-600 hover:text-slate-900"
                    }`}
                  >
                    $ USD
                  </button>
                  <div className="h-4 w-px bg-slate-200 mx-0.5" />
                  <button
                    type="button"
                    onClick={handleResetForm}
                    title="Reset to default values"
                    className="flex items-center gap-1 px-2 py-1 rounded-lg text-[11px] font-medium text-slate-500 hover:text-slate-900 hover:bg-white transition-all"
                  >
                    <RotateCcw className="w-3 h-3" />
                    <span>Reset</span>
                  </button>
                </div>
              </div>

              <form onSubmit={handlePredict} className="space-y-7">
                {/* 1. Customer Information */}
                <div className="space-y-3.5">
                  <h3 className="text-sm font-bold text-blue-600 tracking-tight">
                    1. Customer Information
                  </h3>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                    {/* Gender */}
                    <div>
                      <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                        Gender
                      </label>
                      <select
                        value={formData.gender}
                        onChange={(e) =>
                          handleInputChange("gender", e.target.value as "Male" | "Female")
                        }
                        className="w-full bg-slate-50/80 border border-slate-200 rounded-lg px-3 py-2 text-xs font-medium text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white"
                      >
                        <option value="Male">Male</option>
                        <option value="Female">Female</option>
                      </select>
                    </div>

                    {/* Senior Citizen */}
                    <div>
                      <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                        Senior Citizen
                      </label>
                      <select
                        value={formData.seniorCitizen}
                        onChange={(e) =>
                          handleInputChange("seniorCitizen", e.target.value as "No" | "Yes")
                        }
                        className="w-full bg-slate-50/80 border border-slate-200 rounded-lg px-3 py-2 text-xs font-medium text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white"
                      >
                        <option value="No">No</option>
                        <option value="Yes">Yes</option>
                      </select>
                    </div>

                    {/* Partner */}
                    <div>
                      <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                        Partner
                      </label>
                      <select
                        value={formData.partner}
                        onChange={(e) =>
                          handleInputChange("partner", e.target.value as "No" | "Yes")
                        }
                        className="w-full bg-slate-50/80 border border-slate-200 rounded-lg px-3 py-2 text-xs font-medium text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white"
                      >
                        <option value="Yes">Yes</option>
                        <option value="No">No</option>
                      </select>
                    </div>

                    {/* Dependents */}
                    <div>
                      <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                        Dependents
                      </label>
                      <select
                        value={formData.dependents}
                        onChange={(e) =>
                          handleInputChange("dependents", e.target.value as "No" | "Yes")
                        }
                        className="w-full bg-slate-50/80 border border-slate-200 rounded-lg px-3 py-2 text-xs font-medium text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white"
                      >
                        <option value="Yes">Yes</option>
                        <option value="No">No</option>
                      </select>
                    </div>
                  </div>
                </div>

                {/* 2. Account & Services */}
                <div className="space-y-3.5">
                  <h3 className="text-sm font-bold text-blue-600 tracking-tight">
                    2. Account & Services
                  </h3>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                    {/* Tenure (months) with small badge */}
                    <div className="relative">
                      <label className="block text-xs font-semibold text-slate-700 mb-1.5 flex items-center gap-1">
                        <span>Tenure (months)</span>
                        <Info className="w-3 h-3 text-slate-400" />
                      </label>
                      <div className="relative">
                        <input
                          type="text"
                          inputMode="numeric"
                          value={formData.tenure}
                          onChange={(e) => handleTenureChange(e.target.value)}
                          onFocus={(e) => e.target.select()}
                          className="w-full bg-slate-50/80 border border-slate-200 rounded-lg px-3 py-2 text-xs font-medium text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white pr-16"
                          placeholder="0 - 72"
                        />
                        <div className="absolute right-2 top-1/2 -translate-y-1/2 text-[10px] text-slate-500 font-medium flex items-center gap-0.5 pointer-events-none">
                          <Clock className="w-2.5 h-2.5" />
                          <span>{formData.tenure === "" ? "0" : formData.tenure}m</span>
                        </div>
                      </div>
                    </div>

                    {/* Phone Service */}
                    <div>
                      <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                        Phone Service
                      </label>
                      <select
                        value={formData.phoneService}
                        onChange={(e) =>
                          handleInputChange("phoneService", e.target.value as "No" | "Yes")
                        }
                        className="w-full bg-slate-50/80 border border-slate-200 rounded-lg px-3 py-2 text-xs font-medium text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white"
                      >
                        <option value="Yes">Yes</option>
                        <option value="No">No</option>
                      </select>
                    </div>

                    {/* Multiple Lines */}
                    <div>
                      <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                        Multiple Lines
                      </label>
                      <select
                        value={formData.multipleLines}
                        onChange={(e) =>
                          handleInputChange("multipleLines", e.target.value)
                        }
                        className="w-full bg-slate-50/80 border border-slate-200 rounded-lg px-3 py-2 text-xs font-medium text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white"
                      >
                        <option value="No">No</option>
                        <option value="Yes">Yes</option>
                        <option value="No phone service">No phone service</option>
                      </select>
                    </div>

                    {/* Internet Service */}
                    <div>
                      <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                        Internet Service
                      </label>
                      <select
                        value={formData.internetService}
                        onChange={(e) =>
                          handleInputChange("internetService", e.target.value)
                        }
                        className="w-full bg-slate-50/80 border border-slate-200 rounded-lg px-3 py-2 text-xs font-medium text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white"
                      >
                        <option value="Fiber optic">Fiber optic</option>
                        <option value="DSL">DSL</option>
                        <option value="No">No</option>
                      </select>
                    </div>

                    {/* Online Security */}
                    <div>
                      <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                        Online Security
                      </label>
                      <select
                        value={formData.onlineSecurity}
                        onChange={(e) =>
                          handleInputChange("onlineSecurity", e.target.value)
                        }
                        className="w-full bg-slate-50/80 border border-slate-200 rounded-lg px-3 py-2 text-xs font-medium text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white"
                      >
                        <option value="Yes">Yes</option>
                        <option value="No">No</option>
                        <option value="No internet service">No internet service</option>
                      </select>
                    </div>

                    {/* Online Backup */}
                    <div>
                      <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                        Online Backup
                      </label>
                      <select
                        value={formData.onlineBackup}
                        onChange={(e) =>
                          handleInputChange("onlineBackup", e.target.value)
                        }
                        className="w-full bg-slate-50/80 border border-slate-200 rounded-lg px-3 py-2 text-xs font-medium text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white"
                      >
                        <option value="No">No</option>
                        <option value="Yes">Yes</option>
                        <option value="No internet service">No internet service</option>
                      </select>
                    </div>

                    {/* Device Protection */}
                    <div>
                      <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                        Device Protection
                      </label>
                      <select
                        value={formData.deviceProtection}
                        onChange={(e) =>
                          handleInputChange("deviceProtection", e.target.value)
                        }
                        className="w-full bg-slate-50/80 border border-slate-200 rounded-lg px-3 py-2 text-xs font-medium text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white"
                      >
                        <option value="Yes">Yes</option>
                        <option value="No">No</option>
                        <option value="No internet service">No internet service</option>
                      </select>
                    </div>

                    {/* Tech Support */}
                    <div>
                      <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                        Tech Support
                      </label>
                      <select
                        value={formData.techSupport}
                        onChange={(e) =>
                          handleInputChange("techSupport", e.target.value)
                        }
                        className="w-full bg-slate-50/80 border border-slate-200 rounded-lg px-3 py-2 text-xs font-medium text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white"
                      >
                        <option value="No">No</option>
                        <option value="Yes">Yes</option>
                        <option value="No internet service">No internet service</option>
                      </select>
                    </div>

                    {/* Streaming TV */}
                    <div>
                      <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                        Streaming TV
                      </label>
                      <select
                        value={formData.streamingTV}
                        onChange={(e) =>
                          handleInputChange("streamingTV", e.target.value)
                        }
                        className="w-full bg-slate-50/80 border border-slate-200 rounded-lg px-3 py-2 text-xs font-medium text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white"
                      >
                        <option value="Yes">Yes</option>
                        <option value="No">No</option>
                        <option value="No internet service">No internet service</option>
                      </select>
                    </div>

                    {/* Streaming Movies */}
                    <div>
                      <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                        Streaming Movies
                      </label>
                      <select
                        value={formData.streamingMovies}
                        onChange={(e) =>
                          handleInputChange("streamingMovies", e.target.value)
                        }
                        className="w-full bg-slate-50/80 border border-slate-200 rounded-lg px-3 py-2 text-xs font-medium text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white"
                      >
                        <option value="Yes">Yes</option>
                        <option value="No">No</option>
                        <option value="No internet service">No internet service</option>
                      </select>
                    </div>
                  </div>
                </div>

                {/* 3. Contract & Billing */}
                <div className="space-y-4">
                  <h3 className="text-sm font-bold text-blue-600 tracking-tight">
                    3. Contract & Billing
                  </h3>

                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                    {/* Contract */}
                    <div>
                      <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                        Contract
                      </label>
                      <select
                        value={formData.contract}
                        onChange={(e) =>
                          handleInputChange("contract", e.target.value)
                        }
                        className="w-full bg-slate-50/80 border border-slate-200 rounded-lg px-3 py-2 text-xs font-medium text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white"
                      >
                        <option value="Month-to-month">Month-to-month</option>
                        <option value="One year">One year</option>
                        <option value="Two year">Two year</option>
                      </select>
                    </div>

                    {/* Paperless Billing */}
                    <div>
                      <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                        Paperless Billing
                      </label>
                      <select
                        value={formData.paperlessBilling}
                        onChange={(e) =>
                          handleInputChange("paperlessBilling", e.target.value as "No" | "Yes")
                        }
                        className="w-full bg-slate-50/80 border border-slate-200 rounded-lg px-3 py-2 text-xs font-medium text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white"
                      >
                        <option value="Yes">Yes</option>
                        <option value="No">No</option>
                      </select>
                    </div>

                    {/* Payment Method */}
                    <div>
                      <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                        Payment Method
                      </label>
                      <select
                        value={formData.paymentMethod}
                        onChange={(e) =>
                          handleInputChange("paymentMethod", e.target.value)
                        }
                        className="w-full bg-slate-50/80 border border-slate-200 rounded-lg px-2.5 py-2 text-xs font-medium text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white"
                      >
                        <option value="Electronic check">Electronic check</option>
                        <option value="Mailed check">Mailed check</option>
                        <option value="Bank transfer (automatic)">
                          Bank transfer (automatic)
                        </option>
                        <option value="Credit card (automatic)">
                          Credit card (automatic)
                        </option>
                      </select>
                    </div>
                  </div>

                  {/* Monthly Charges & Total Charges inputs (Requirements 1, 2, 5, 6) */}
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-1">
                    {/* Monthly Charges */}
                    <div>
                      <label className="block text-xs font-semibold text-slate-700 mb-1">
                        Monthly Charges ({currencySymbol})
                      </label>
                      <p className="text-[11px] text-slate-500 mb-1.5">
                        {currency === "INR"
                          ? "Enter the customer's monthly charges in INR."
                          : "Enter the customer's monthly charges in USD."}
                      </p>
                      <input
                        type="text"
                        inputMode="decimal"
                        value={formData.monthlyCharges}
                        onChange={(e) => handleMonthlyChargesChange(e.target.value)}
                        onFocus={(e) => e.target.select()}
                        className="w-full bg-slate-50/80 border border-slate-200 rounded-lg px-3 py-2 text-xs font-medium text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white"
                        placeholder={currency === "INR" ? "8700" : "90.00"}
                      />
                      {currency === "INR" && (
                        <div className="text-[11px] text-blue-600 font-medium mt-1 flex items-center justify-between">
                          <span>
                            Converted value: ${convert_inr_to_usd(Number(formData.monthlyCharges) || 0).toFixed(2)} USD
                          </span>
                          <span className="text-[10px] text-slate-400">Rate: {INR_TO_USD_RATE}</span>
                        </div>
                      )}
                    </div>

                    {/* Total Charges (Manual Entry - No automatic calculation) */}
                    <div>
                      <label className="block text-xs font-semibold text-slate-700 mb-1">
                        Total Charges ({currencySymbol})
                      </label>
                      <p className="text-[11px] text-slate-500 mb-1.5">
                        {currency === "INR"
                          ? "Enter the customer's total accumulated charges in INR."
                          : "Enter the customer's total accumulated charges in USD."}
                      </p>
                      <input
                        type="text"
                        inputMode="decimal"
                        value={formData.totalCharges}
                        onChange={(e) => handleTotalChargesChange(e.target.value)}
                        onFocus={(e) => e.target.select()}
                        className="w-full bg-slate-50/80 border border-slate-200 rounded-lg px-3 py-2 text-xs font-medium text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white"
                        placeholder={currency === "INR" ? "17400" : "180.00"}
                      />
                      {currency === "INR" && (
                        <div className="text-[11px] text-blue-600 font-medium mt-1 flex items-center justify-between">
                          <span>
                            Converted value: ${convert_inr_to_usd(Number(formData.totalCharges) || 0).toFixed(2)} USD
                          </span>
                          <span className="text-[10px] text-slate-400">Rate: {INR_TO_USD_RATE}</span>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* INR -> USD Conversion Information Banner */}
                  {currency === "INR" ? (
                    <div className="bg-blue-50/70 border border-blue-100 rounded-xl p-3.5 text-xs text-blue-900 space-y-1">
                      <div className="font-semibold text-blue-800 flex items-center gap-1.5">
                        <Sparkles className="w-3.5 h-3.5 text-blue-600" />
                        <span>Automatic INR → USD Conversion for Trained Model</span>
                      </div>
                      <p className="text-blue-700 text-[11px] leading-relaxed">
                        The IBM Telco ML model is trained on original USD charges. Your ₹ INR inputs are converted before prediction (USD = INR × {INR_TO_USD_RATE}).
                      </p>
                    </div>
                  ) : (
                    <div className="bg-slate-50 border border-slate-200/80 rounded-xl p-3 text-xs text-slate-600">
                      <span>Standard USD dataset values sent directly to ML pipeline.</span>
                    </div>
                  )}

                  {/* Gradient "Predict Churn" Button */}
                  <button
                    type="submit"
                    disabled={isLoading}
                    className="w-full py-3.5 rounded-xl bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 hover:from-blue-700 hover:to-indigo-700 text-white font-bold text-sm shadow-lg shadow-blue-500/25 flex items-center justify-center gap-2 transition-all hover:scale-[1.01] active:scale-[0.99] disabled:opacity-60"
                  >
                    {isLoading ? (
                      <>
                        <Loader2 className="w-4 h-4 animate-spin text-white" />
                        <span>Predicting Customer Churn...</span>
                      </>
                    ) : (
                      <>
                        <Brain className="w-5 h-5 text-white" />
                        <span>Predict Churn</span>
                      </>
                    )}
                  </button>

                  {/* Security message */}
                  <div className="bg-blue-50/50 border border-blue-100/80 rounded-lg p-2.5 text-xs text-blue-700 flex items-center gap-2">
                    <Lock className="w-3.5 h-3.5 text-blue-600 shrink-0" />
                    <span>Your data is secure. We do not store any of your information.</span>
                  </div>
                </div>
              </form>
            </div>

            {/* Right Column: Prediction Result Card (5 cols) */}
            <div className="lg:col-span-5 bg-white rounded-2xl border border-slate-200/80 shadow-sm p-6 sm:p-7 space-y-6 sticky top-24">
              {/* Header */}
              <div className="flex items-start gap-3 pb-2 border-b border-slate-100">
                <div className="w-9 h-9 rounded-xl bg-indigo-50 text-indigo-600 flex items-center justify-center shrink-0">
                  <BarChart3 className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="font-bold text-base text-slate-900 tracking-tight">
                    Prediction Result
                  </h3>
                  <p className="text-xs text-slate-500">Based on the provided information</p>
                </div>
              </div>

              {/* Big Risk Card with Circular Progress Meter */}
              <div
                className={`rounded-2xl p-6 text-center space-y-4 border ${
                  isChurn
                    ? "bg-red-50/60 border-red-100"
                    : "bg-emerald-50/60 border-emerald-100"
                }`}
              >
                {/* Warning / Success icon */}
                <div>
                  {isChurn ? (
                    <div className="w-12 h-12 rounded-full bg-red-100 text-red-600 flex items-center justify-center mx-auto">
                      <AlertTriangle className="w-6 h-6" />
                    </div>
                  ) : (
                    <div className="w-12 h-12 rounded-full bg-emerald-100 text-emerald-600 flex items-center justify-center mx-auto">
                      <CheckCircle2 className="w-6 h-6" />
                    </div>
                  )}
                </div>

                {/* Big Text Alert */}
                <div
                  className={`text-lg font-black tracking-wider uppercase ${
                    isChurn ? "text-red-600" : "text-emerald-600"
                  }`}
                >
                  {isChurn ? "HIGH CHURN RISK" : "LOW CHURN RISK"}
                </div>

                {/* Circular Gauge */}
                <div className="relative w-36 h-36 mx-auto flex items-center justify-center">
                  <svg className="w-full h-full -rotate-90" viewBox="0 0 100 100">
                    <circle
                      cx="50"
                      cy="50"
                      r="42"
                      stroke="#f1f5f9"
                      strokeWidth="8"
                      fill="none"
                    />
                    <circle
                      cx="50"
                      cy="50"
                      r="42"
                      stroke={isChurn ? "#ef4444" : "#10b981"}
                      strokeWidth="8"
                      strokeDasharray={`${
                        (predictionResult.churn_probability || 0.826) * 263.89
                      } 263.89`}
                      strokeLinecap="round"
                      fill="none"
                      className="transition-all duration-1000 ease-out"
                    />
                  </svg>
                  <div className="absolute flex flex-col items-center justify-center text-center">
                    <span className="text-2xl font-black text-slate-900">
                      {predictionResult.churn_probability_percentage || "82.6%"}
                    </span>
                    <span className="text-[10px] text-slate-500 font-semibold tracking-wider uppercase">
                      Churn Probability
                    </span>
                  </div>
                </div>

                {/* Verdict Sentence */}
                <p className="text-sm font-semibold text-slate-800">
                  {predictionResult.verdict || "This customer is likely to churn."}
                </p>

                {/* Risk Badge */}
                <div>
                  <span
                    className={`inline-block text-xs font-semibold px-4 py-1 rounded-full ${
                      isChurn
                        ? "bg-red-100 text-red-600"
                        : "bg-emerald-100 text-emerald-600"
                    }`}
                  >
                    {isChurn ? "High Risk Customer" : "Low Risk Customer"}
                  </span>
                </div>
              </div>

              {/* Evaluated Customer Charges (Requirement 7) */}
              <div className="bg-slate-50 border border-slate-200/80 rounded-xl p-3.5 space-y-2 text-xs">
                <div className="flex items-center justify-between font-semibold text-slate-800">
                  <span className="flex items-center gap-1.5">
                    <CreditCard className="w-4 h-4 text-slate-600" />
                    <span>Evaluated Customer Charges</span>
                  </span>
                  <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-blue-100 text-blue-700">
                    {currency === "INR" ? "₹ INR Input" : "$ USD Input"}
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-2 pt-1 text-slate-600">
                  <div className="bg-white p-2.5 rounded-lg border border-slate-200/70">
                    <span className="text-[10px] text-slate-400 block font-medium">Monthly Charges</span>
                    <span className="font-bold text-slate-800 text-sm">
                      {currency === "INR"
                        ? formatINR(Number(predictionResult.inr_inputs?.MonthlyCharges ?? formData.monthlyCharges) || 0)
                        : formatUSD(Number(formData.monthlyCharges) || 0)}
                    </span>
                    {currency === "INR" && (
                      <span className="text-[10px] text-blue-600 block mt-0.5 font-medium">
                        Model: ${convert_inr_to_usd(Number(predictionResult.inr_inputs?.MonthlyCharges ?? formData.monthlyCharges) || 0).toFixed(2)} USD
                      </span>
                    )}
                  </div>
                  <div className="bg-white p-2.5 rounded-lg border border-slate-200/70">
                    <span className="text-[10px] text-slate-400 block font-medium">Total Charges</span>
                    <span className="font-bold text-slate-800 text-sm">
                      {currency === "INR"
                        ? formatINR(Number(predictionResult.inr_inputs?.TotalCharges ?? formData.totalCharges) || 0)
                        : formatUSD(Number(formData.totalCharges) || 0)}
                    </span>
                    {currency === "INR" && (
                      <span className="text-[10px] text-blue-600 block mt-0.5 font-medium">
                        Model: ${convert_inr_to_usd(Number(predictionResult.inr_inputs?.TotalCharges ?? formData.totalCharges) || 0).toFixed(2)} USD
                      </span>
                    )}
                  </div>
                </div>
              </div>

              {/* Recommendation Section */}
              <div className="space-y-2">
                <div className="flex items-center gap-1.5 text-xs font-bold text-blue-600 tracking-tight">
                  <Target className="w-4 h-4" />
                  <span>Recommendation</span>
                </div>
                <div className="bg-blue-50/50 border border-blue-100/80 rounded-xl p-3.5 text-xs text-slate-700 leading-relaxed">
                  {predictionResult.recommendation}
                </div>
              </div>

              {/* Top Factors Contributing */}
              <div className="space-y-2">
                <div className="flex items-center gap-1.5 text-xs font-bold text-amber-500 tracking-tight">
                  <Star className="w-4 h-4 fill-amber-400" />
                  <span>Top Factors Contributing</span>
                </div>
                <div className="space-y-1.5">
                  {predictionResult.factors.map((factor, idx) => (
                    <div
                      key={idx}
                      className="bg-slate-50 border border-slate-200/70 rounded-xl p-2.5 text-xs text-slate-700 flex items-center gap-2.5 font-medium"
                    >
                      {factor.icon === "contract" && (
                        <Package className="w-3.5 h-3.5 text-purple-500 shrink-0" />
                      )}
                      {factor.icon === "charges" && (
                        <ShoppingBag className="w-3.5 h-3.5 text-pink-500 shrink-0" />
                      )}
                      {factor.icon === "internet" && (
                        <Wifi className="w-3.5 h-3.5 text-blue-500 shrink-0" />
                      )}
                      {factor.icon === "tenure" && (
                        <Target className="w-3.5 h-3.5 text-cyan-500 shrink-0" />
                      )}
                      <span>{factor.label}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Note callout box */}
              <div className="bg-emerald-50/60 border border-emerald-100/80 rounded-xl p-3.5 text-xs text-emerald-900 leading-relaxed">
                <strong className="font-bold">Note:</strong> This prediction is generated
                using a trained Machine Learning model on historical customer data. It is
                not 100% guaranteed.
              </div>
            </div>
          </div>
        )}

        {/* ======================================================== */}
        {/* VIEW 3: ABOUT PAGE */}
        {/* ======================================================== */}
        {currentPage === "about" && (
          <div className="max-w-3xl mx-auto space-y-8 py-4">
            <div className="border-b border-slate-200 pb-4">
              <h2 className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">
                About ChurnGuard
              </h2>
              <p className="text-sm text-slate-500 mt-1">
                Machine Learning architecture and dataset background.
              </p>
            </div>

            <div className="bg-white rounded-2xl border border-slate-200/80 shadow-sm p-6 sm:p-8 space-y-6">
              <h3 className="text-lg font-bold text-slate-900 flex items-center gap-2">
                <Brain className="w-5 h-5 text-blue-600" />
                <span>IBM Telco Customer Churn ML Pipeline</span>
              </h3>
              <p className="text-sm text-slate-600 leading-relaxed">
                ChurnGuard predicts whether a telecom customer is likely to unsubscribe
                based on customer demographic, contract, and service engagement patterns.
                The system was trained on 7,043 authentic telecom accounts from the IBM
                Telco dataset using Scikit-Learn pipelines serialized with Joblib.
              </p>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-2">
                <div className="bg-slate-50 p-4 rounded-xl border border-slate-200/80">
                  <div className="text-xs font-semibold text-slate-500 uppercase">
                    Dataset Features
                  </div>
                  <div className="text-lg font-bold text-slate-900 mt-1">21 Parameters</div>
                  <div className="text-xs text-slate-500 mt-1">
                    Contract type, tenure, billing, and value-added services.
                  </div>
                </div>

                <div className="bg-slate-50 p-4 rounded-xl border border-slate-200/80">
                  <div className="text-xs font-semibold text-slate-500 uppercase">
                    Multi-Currency Support
                  </div>
                  <div className="text-lg font-bold text-blue-600 mt-1">USD & INR (₹)</div>
                  <div className="text-xs text-slate-500 mt-1">
                    Internal conversion layer preserves trained USD model accuracy.
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* ======================================================== */}
      {/* 3. FOOTER (Exact ChurnGuard ML Design from Screenshot) */}
      {/* ======================================================== */}
      <footer className="bg-[#0b132b] text-slate-400 py-6 text-xs mt-auto border-t border-slate-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-3">
          <div className="flex items-center gap-2 text-slate-300">
            <Lock className="w-3.5 h-3.5 text-slate-400" />
            <span>Secure • Reliable • Accurate</span>
          </div>
          <div>
            <span>© 2024 ChurnGuard. All rights reserved.</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

import express from "express";
import { execFile } from "child_process";
import path from "path";
import fs from "fs";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT ? parseInt(process.env.PORT, 10) : 3000;

app.use(express.json());

// Load precomputed experiment results
const experimentResultsPath = path.join(__dirname, "customer-churn-ml", "data", "experiment_results.json");
let experimentResults: any = null;
if (fs.existsSync(experimentResultsPath)) {
  try {
    experimentResults = JSON.parse(fs.readFileSync(experimentResultsPath, "utf-8"));
  } catch (err) {
    console.error("Failed to parse experiment_results.json:", err);
  }
}

// Serve public static visualizations
app.use("/visualizations", express.static(path.join(__dirname, "public", "visualizations")));
app.use("/public", express.static(path.join(__dirname, "public")));

// API Endpoints
app.get("/api/health", (req, res) => {
  const modelExists = fs.existsSync(path.join(__dirname, "customer-churn-ml", "models", "customer_churn_model.joblib"));
  res.json({
    status: "healthy",
    model_loaded: modelExists,
    model_path: "customer-churn-ml/models/customer_churn_model.joblib",
    engine: "Scikit-Learn Pipeline via Joblib (Python 3.10)",
    experiment_results_available: experimentResults !== null,
  });
});

app.get("/api/experiments", (req, res) => {
  if (experimentResults) {
    res.json(experimentResults);
  } else if (fs.existsSync(experimentResultsPath)) {
    const data = JSON.parse(fs.readFileSync(experimentResultsPath, "utf-8"));
    res.json(data);
  } else {
    res.status(404).json({ error: "Experiment results not found. Run run_all.py first." });
  }
});

app.get("/api/unseen", (req, res) => {
  if (experimentResults && experimentResults.unseen_predictions) {
    res.json(experimentResults.unseen_predictions);
  } else {
    res.json([]);
  }
});

// Configurable INR to USD Exchange Rate Setting
// Default rate converts ₹8,700 -> ~$90.00 USD (1 USD ≈ 96.67 INR) as requested in the specification.
export const INR_TO_USD_RATE = Number(process.env.INR_TO_USD_RATE) || 0.010345;

export function convert_inr_to_usd(inr_amount: number): number {
  if (typeof inr_amount !== "number" || isNaN(inr_amount) || inr_amount < 0) {
    return 0.0;
  }
  return Number((inr_amount * INR_TO_USD_RATE).toFixed(2));
}

app.get("/api/exchange-rate", (req, res) => {
  res.json({
    rate: INR_TO_USD_RATE,
    currency_from: "INR",
    currency_to: "USD",
    formula: "USD = INR * INR_TO_USD_RATE",
    sample_monthly_inr: 8700,
    sample_monthly_usd: convert_inr_to_usd(8700),
    sample_total_inr: 17400,
    sample_total_usd: convert_inr_to_usd(17400),
  });
});

// Live inference endpoint invoking Joblib model directly with backend INR -> USD conversion
app.post("/api/predict", (req, res) => {
  const customerData = req.body;
  if (!customerData || typeof customerData !== "object") {
    return res.status(400).json({ error: "Invalid JSON customer payload provided." });
  }

  // 1. Validation: Accept only positive numeric values for charges and valid tenure
  const rawMonthly = Number(customerData.MonthlyCharges);
  const rawTotal = Number(customerData.TotalCharges);
  const tenure = Number(customerData.tenure);

  if (isNaN(rawMonthly) || rawMonthly < 0) {
    return res.status(400).json({
      error: "Validation Error: Monthly Charges must be a positive numeric value.",
    });
  }

  if (isNaN(rawTotal) || rawTotal < 0) {
    return res.status(400).json({
      error: "Validation Error: Total Charges must be a positive numeric value.",
    });
  }

  if (isNaN(tenure) || tenure < 0 || tenure > 72) {
    return res.status(400).json({
      error: "Validation Error: Tenure must be between 0 and 72 months.",
    });
  }

  // 2. Currency Detection and Automatic INR -> USD Conversion
  const isINR =
    customerData.currency === "INR" ||
    customerData.is_inr === true ||
    rawMonthly > 150;

  let inrMonthly: number;
  let inrTotal: number;
  let usdMonthly: number;
  let usdTotal: number;

  if (isINR) {
    inrMonthly = rawMonthly;
    inrTotal = rawTotal;
    usdMonthly = convert_inr_to_usd(rawMonthly);
    usdTotal = convert_inr_to_usd(rawTotal);

    console.log(
      `[Backend INR -> USD Conversion] Received INR inputs: ` +
      `MonthlyCharges = ₹${Math.round(inrMonthly).toLocaleString("en-IN")} | ` +
      `TotalCharges = ₹${Math.round(inrTotal).toLocaleString("en-IN")}`
    );
    console.log(
      `[Backend INR -> USD Conversion] Converted to USD: ` +
      `MonthlyCharges = $${usdMonthly.toFixed(2)} | ` +
      `TotalCharges = $${usdTotal.toFixed(2)} (Rate: ${INR_TO_USD_RATE})`
    );
  } else {
    // Input is already in USD (e.g. 80.50 Monthly, 241.50 Total)
    usdMonthly = Number(rawMonthly.toFixed(2));
    usdTotal = Number(rawTotal.toFixed(2));
    inrMonthly = Math.round(usdMonthly / INR_TO_USD_RATE);
    inrTotal = Math.round(usdTotal / INR_TO_USD_RATE);

    console.log(
      `[Backend USD Input] MonthlyCharges = $${usdMonthly.toFixed(2)} | ` +
      `TotalCharges = $${usdTotal.toFixed(2)}`
    );
  }

  // 3. Map SeniorCitizen if provided as Yes/No
  let seniorCitizenVal = customerData.SeniorCitizen;
  if (typeof seniorCitizenVal === "string") {
    seniorCitizenVal = seniorCitizenVal.toLowerCase() === "yes" ? 1 : 0;
  }

  // 4. Send ONLY the calibrated USD values to the trained ML model
  const modelPayload = {
    ...customerData,
    SeniorCitizen: seniorCitizenVal,
    MonthlyCharges: usdMonthly,
    TotalCharges: usdTotal,
    MonthlyCharges_USD: usdMonthly,
    TotalCharges_USD: usdTotal,
    MonthlyCharges_INR: inrMonthly,
    TotalCharges_INR: inrTotal,
    currency: isINR ? "INR" : "USD",
    is_inr: isINR,
  };

  const scriptPath = path.join(__dirname, "customer-churn-ml", "src", "predict.py");
  const payloadStr = JSON.stringify(modelPayload);
  const pythonExecutable = process.env.PYTHON_EXECUTABLE || (process.platform === "win32" ? "python" : "python3");

  execFile(pythonExecutable, [scriptPath, "--json", payloadStr], { maxBuffer: 1024 * 1024 }, (error, stdout, stderr) => {
    if (error) {
      console.error("Python inference error:", error, stderr);
      return res.status(500).json({
        error: "Inference script execution failed",
        details: stderr || error.message,
      });
    }

    try {
      const parsedResult = JSON.parse(stdout.trim());
      if (parsedResult.error) {
        return res.status(500).json({ error: parsedResult.error });
      }

      // Explicitly attach conversion metadata to development response for verification
      parsedResult.inr_inputs = {
        MonthlyCharges: inrMonthly,
        TotalCharges: inrTotal,
        formatted_monthly: `₹${Math.round(inrMonthly).toLocaleString("en-IN")}`,
        formatted_total: `₹${Math.round(inrTotal).toLocaleString("en-IN")}`,
      };
      parsedResult.converted_usd = {
        MonthlyCharges: usdMonthly,
        TotalCharges: usdTotal,
        formatted_monthly: `$${usdMonthly.toFixed(2)}`,
        formatted_total: `$${usdTotal.toFixed(2)}`,
      };
      parsedResult.exchange_rate = INR_TO_USD_RATE;
      res.json(parsedResult);
    } catch (parseErr) {
      console.error("Failed to parse Python output:", stdout);
      res.status(500).json({ error: "Failed to parse model output", raw: stdout });
    }
  });
});

async function startServer() {
  const isProduction = process.env.NODE_ENV === "production";

  if (!isProduction) {
    const { createServer: createViteServer } = await import("vite");
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(__dirname, "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`[Express] Churn Prediction Server listening on http://0.0.0.0:${PORT}`);
  });
}

startServer();

# Customer Churn Prediction Using Machine Learning with Model Optimization and Deployment-Ready Pipeline

A complete, production-grade Machine Learning system designed to predict customer churn in telecommunications and subscription enterprises. Built with rigorous end-to-end data science methodology: data hygiene, exploratory data analysis (EDA), data-leakage prevention, domain feature engineering, multi-algorithm benchmarking, 5-fold cross-validated hyperparameter optimization, scikit-learn pipeline persistence via Joblib, RESTful FastAPI microservice, and an interactive intelligence dashboard.

---

## 1. Project Title
**"Customer Churn Prediction Using Machine Learning with Model Optimization and Deployment-Ready Pipeline"**

---

## 2. Problem Statement
Customer attrition (churn) directly erodes recurring subscription revenues and increases customer acquisition costs (CAC). In telecom and SaaS businesses, acquiring a new subscriber is typically 5 to 7 times more expensive than retaining an existing customer. The objective of this project is to build an accurate, calibrated classification system that predicts whether a subscriber is likely to cancel service, identifies their specific risk drivers (e.g., contract type, pricing friction, tech support gaps), and provides automated retention interventions before attrition occurs.

---

## 3. Real-World Motivation
Telecom markets suffer from fierce competition, commoditized connectivity, and low switching friction for monthly contract holders. Proactively targeting high-risk subscribers with tailored retention offers (such as term renewal incentives, loyalty discounts, or complimentary device protection) preserves customer lifetime value (LTV) and protects recurring revenue streams.

---

## 4. Dataset Information
* **Dataset Name:** IBM Telco Customer Churn Dataset
* **Dataset Source:** IBM Business Analytics / Public ML Repository (Kaggle)
* **Dataset Size:** 7,043 rows, 21 columns
* **Target Variable:** `Churn` (Binary: `Yes` = 1, `No` = 0)
* **Target Distribution:**
  * **No Churn (Retained):** 5,174 customers (73.46%)
  * **Churn (Attrited):** 1,869 customers (26.54%)
* **Why this represents a real-world problem:** Real customer datasets are characterized by class imbalance (attrition ~26%), heterogeneous data types (continuous billing, categorical contracts, demographic flags), and real data cleanliness flaws (such as blank strings in cumulative charges for newly onboarded customers).

---

## 5. Feature Catalog & Descriptions

| Feature Name | Type | Description & Domain Relevance |
| :--- | :--- | :--- |
| `customerID` | String | Unique identifier (dropped prior to training to prevent overfitting) |
| `gender` | Categorical | Customer gender (Male / Female) |
| `SeniorCitizen` | Binary | Whether customer is 65+ years old (1 = Yes, 0 = No) |
| `Partner` | Categorical | Whether customer has a domestic partner (Yes / No) |
| `Dependents` | Categorical | Whether customer has dependents/children (Yes / No) |
| `tenure` | Numeric (int) | Number of continuous months customer has stayed with company |
| `PhoneService` | Categorical | Whether customer has landline phone service (Yes / No) |
| `MultipleLines` | Categorical | Multiple telephone lines (Yes / No / No phone service) |
| `InternetService` | Categorical | Internet connection technology (DSL / Fiber optic / No) |
| `OnlineSecurity` | Categorical | Add-on cyber security filter (Yes / No / No internet service) |
| `OnlineBackup` | Categorical | Cloud backup service (Yes / No / No internet service) |
| `DeviceProtection`| Categorical | Hardware warranty & insurance (Yes / No / No internet service) |
| `TechSupport` | Categorical | Dedicated 24/7 technical assistance (Yes / No / No internet service) |
| `StreamingTV` | Categorical | IPTV streaming subscription (Yes / No / No internet service) |
| `StreamingMovies`| Categorical | Movie package subscription (Yes / No / No internet service) |
| `Contract` | Categorical | Agreement terms: Month-to-month, One year, Two year |
| `PaperlessBilling`| Categorical | Digital e-billing statements (Yes / No) |
| `PaymentMethod` | Categorical | Electronic check, Mailed check, Bank transfer (auto), Credit card (auto) |
| `MonthlyCharges` | Numeric (float)| Current monthly invoice billing amount ($ USD) |
| `TotalCharges` | Numeric (float)| Cumulative total billed over customer lifetime ($ USD) |
| **`Churn`** | **Binary (Target)** | **Whether subscriber terminated service (Yes / No)** |

---

## 6. Technology Stack
* **Language:** Python 3.10+
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-learn (`Pipeline`, `ColumnTransformer`, `GridSearchCV`)
* **Model Serialization:** Joblib
* **Data Visualization:** Matplotlib, Seaborn
* **API Microservice:** FastAPI, Uvicorn, Pydantic v2
* **Notebooks:** Jupyter Notebook / IPython
* **Dashboard Frontend:** React 19, TypeScript, Tailwind CSS, Lucide Icons

---

## 7. Data Quality Check & Preprocessing
Real-world inspection conducted on the raw data revealed:
1. **Empty String Detection:** Exactly 11 rows in `TotalCharges` contained whitespace `' '`. These represent brand-new accounts where `tenure == 0` (no billing cycle completed yet). These were converted to numeric and cleanly imputed with `0.0`.
2. **Identifier Removal:** `customerID` was removed to prevent high-cardinality label memorization.
3. **Data Leakage Prevention:** The dataset was split into **Training (80%, 5,634 rows)** and **Test (20%, 1,409 rows)** using stratified sampling *before* computing scaling statistics or imputations.
4. **Encoding & Scaling:**
   - Numerical attributes (`tenure`, `MonthlyCharges`, `TotalCharges`, engineered metrics) scaled via `StandardScaler()`.
   - Categorical attributes encoded via `OneHotEncoder(drop='first', handle_unknown='ignore')`.

---

## 8. Domain Feature Engineering
Six domain-specific features were engineered:
1. `tenure_group`: Lifecycle discretization into brackets (`0-12m`, `13-24m`, `25-48m`, `49-60m`, `>60m`).
2. `total_services`: Integer count (0 to 6) of optional add-ons (`OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`). Customers with higher service breadth experience steeper switching costs.
3. `charges_ratio`: Ratio of `TotalCharges` divided by (`MonthlyCharges * tenure`), indicating billing anomalies or historical discounts.
4. `charges_per_tenure`: Progression of spending rate per month of tenure.
5. `has_family_support`: Compound flag if customer has both `Partner` and `Dependents` (family accounts display lower churn).
6. `streaming_bundle`: Compound flag if customer subscribes to both `StreamingTV` and `StreamingMovies`.

---

## 9. Baseline Model Comparison (Actual Experimental Results)

All models were evaluated on the identical holdout test set (1,409 customers) using stratified metrics:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.8062** | **0.6700** | **0.5321** | **0.5931** | **0.8466** |
| **Gradient Boosting** | 0.7977 | 0.6540 | 0.5053 | 0.5701 | 0.8437 |
| **Random Forest** | 0.7871 | 0.6259 | 0.4920 | 0.5509 | 0.8233 |
| **Decision Tree** | 0.7921 | 0.6441 | 0.4840 | 0.5527 | 0.8177 |

### Why these metrics matter for imbalanced churn:
* **Accuracy (80.6%):** A naive "No Churn" predictor yields 73.5% accuracy. True model skill must be validated through discriminative metrics.
* **Precision (67.0%):** Guarantees retention promotions are not squandered on loyal, non-churning customers.
* **Recall (53.2%):** Catches true departing customers; critical for preserving customer lifetime value.
* **ROC-AUC (0.8466):** Demonstrates strong threshold-independent rank discrimination.

---

## 10. Automated Model Selection & Hyperparameter Tuning
The pipeline autonomously selected **Logistic Regression** as the best baseline model due to its highest ROC-AUC (0.8466) and balanced F1-score (0.5931), with Gradient Boosting performing closely.

### GridSearchCV 5-Fold Stratified Cross-Validation:
* **Search Space:** `C: [0.01, 0.1, 1.0, 5.0]`, `solver: ['lbfgs', 'liblinear']`
* **Best Hyperparameters Found:** `{'C': 0.1, 'solver': 'lbfgs'}`
* **Best 5-Fold CV ROC-AUC:** **0.8482**
* **Final Holdout Test ROC-AUC:** **0.8467**

---

## 11. Integrated Production Pipeline & Model Persistence
The complete preprocessing pipeline, feature engineering transformer, and optimized classifier were packaged into a single scikit-learn `Pipeline` object:
```
Raw Customer JSON / Dict
        ↓
ChurnFeatureEngineer()
        ↓
ColumnTransformer [StandardScaler + OneHotEncoder]
        ↓
Optimized Logistic Regression (C=0.1, solver='lbfgs')
        ↓
Prediction Class & Probability (predict_proba)
```
Saved to disk via Joblib:
```python
joblib.dump(final_pipeline, "models/customer_churn_model.joblib")
```

---

## 12. Inference on Realistic Unseen Customers (Without Retraining)

Predictions generated strictly from the loaded Joblib artifact:

### Customer 1: `UNSEEN-001` (High-Risk New Digital Streamer)
* **Profile:** 2 months tenure, Month-to-month, Fiber optic, No Tech Support, Electronic check, $89.85/mo.
* **Model Prediction:** **Churn**
* **Churn Probability:** **85.2%**
* **Risk Drivers:** Month-to-month contract, low tenure, lack of technical support, electronic check payment.
* **Retention Strategy:** Offer $15/mo discount for converting to a 1-year agreement + 3 months free tech support.

### Customer 2: `UNSEEN-002` (Low-Risk Long-Term Family Account)
* **Profile:** 65 months tenure, Two-year contract, DSL, Full security suite, Bank auto-pay, $64.80/mo.
* **Model Prediction:** **No Churn**
* **Churn Probability:** **1.0%**
* **Risk Drivers:** Long loyalty tenure, two-year contract commitment, multi-product bundle.
* **Retention Strategy:** Standard annual VIP appreciation thank-you; no discount needed.

### Customer 3: `UNSEEN-003` (Moderate-Risk Mid-Tenure Solo Professional)
* **Profile:** 18 months tenure, One-year contract, Fiber optic, Credit card auto-pay, $84.40/mo.
* **Model Prediction:** **No Churn** (Moderate Risk)
* **Churn Probability:** **35.3%**
* **Risk Drivers:** Higher monthly billing, fiber optic without active tech support.
* **Retention Strategy:** Proactive check-in on fiber performance; offer complimentary device protection.

---

## 13. FastAPI Microservice Usage

Run the FastAPI backend:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Endpoints:
1. `GET /` — API metadata and available endpoints
2. `GET /health` — Health check verifying model is loaded in memory
3. `POST /predict` — Real-time inference

### Example Request (`POST /predict`):
```json
{
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
  "TotalCharges": 179.70
}
```

### Example Response:
```json
{
  "prediction": "Churn",
  "churn_probability": 0.852,
  "verdict": "Customer is likely to churn",
  "diagnostics": {
    "risk_level": "High",
    "key_risk_drivers": [
      "Month-to-month commitment provides zero switching barriers.",
      "Early lifecycle customer (2.0 months tenure); high initial hazard period.",
      "High monthly billing ($89.85/mo); customer sensitive to price-to-value.",
      "Payment via Electronic Check exhibits highest historical churn correlation."
    ],
    "retention_anchors": [],
    "recommended_interventions": [
      "Offer a 15% discount for upgrading to a 1-year or 2-year annual agreement."
    ]
  }
}
```

---

## 14. Real-World Business Workflow

```
Customer Data (CRM / Billing System)
               ↓
Real-Time ML Pipeline (Joblib Artifact)
               ↓
Churn Probability Scoring (0.0 to 1.0)
               ↓
Risk Tier Segmentation:
  • High Risk (≥60%): Proactive Retention Team Call + 15% Annual Upgrade Incentive
  • Medium Risk (35-59%): Automated Email Concierge + Add-on Trial (Tech Support)
  • Low Risk (<35%): Normal Service & Cross-sell
               ↓
Measure Retention ROI & Reduce Customer Acquisition Expense
```

---

## 15. How to Run the Project

### Clone & Install:
```bash
git clone https://github.com/tajamulkhann/Machine-Learning-Projects.git # Reference
cd customer-churn-ml
pip install -r requirements.txt
```

### Execute Complete Pipeline (EDA, Training, Tuning, Visualizations):
```bash
python run_all.py
```

### Run Unseen Predictions Directly:
```bash
python -m src.predict
```

### Launch FastAPI Microservice:
```bash
uvicorn api.main:app --port 8000
```

---

## 16. Future Scope
1. **Explainable AI (SHAP & LIME):** Integrate TreeSHAP/KernelSHAP force plots for individual customer risk explanations.
2. **Survival Analysis:** Implement Cox Proportional Hazards or Kaplan-Meier estimators to predict the exact month a customer is most likely to churn.
3. **Automated Retraining (MLOps):** Integrate MLflow for model registry and Apache Airflow for monthly pipeline retraining on fresh CRM dumps.

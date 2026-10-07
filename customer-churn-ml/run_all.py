"""
Master Pipeline Execution & Experiment Runner
==============================================
Runs the entire customer churn project step-by-step:
1. Data Acquisition & Data Quality Check
2. Exploratory Data Analysis (generates and saves all visualizations)
3. Preprocessing & Leakage-Free Train/Test Split
4. Feature Engineering
5. Baseline Training (4 models)
6. Evaluation & Comparison Table
7. Best Model Selection & Hyperparameter Tuning (GridSearchCV)
8. Final Pipeline Assembly & Model Persistence (Joblib)
9. Unseen Customer Prediction (Loading Joblib without retraining)
10. Export verified metrics to JSON for the frontend dashboard
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure local imports work cleanly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

from src.data_preprocessing import load_raw_data, clean_raw_dataframe, prepare_train_test, get_feature_types
from src.feature_engineering import ChurnFeatureEngineer, explain_engineered_features
from src.train import train_baseline_models, tune_best_model, extract_feature_importances, save_pipeline
from src.evaluate import evaluate_classifier, explain_metrics
from src.predict import load_model, predict_single_customer, UNSEEN_CUSTOMERS


VIZ_DIR = os.path.join(CURRENT_DIR, "visualizations")
os.makedirs(VIZ_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"


def run_eda_and_save_visualizations(df_raw: pd.DataFrame):
    """Generate all required EDA visualizations with titles, axis labels, legends and save to disk."""
    print("\n[EDA] Generating and saving charts to visualizations/ ...")
    df = clean_raw_dataframe(df_raw)

    # 1. Target Class Distribution
    fig, ax = plt.subplots(figsize=(7, 5))
    churn_counts = df["Churn"].value_counts()
    colors = ["#2563eb", "#dc2626"]
    bars = ax.bar(churn_counts.index, churn_counts.values, color=colors, width=0.5, edgecolor="black", linewidth=1.2)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:,} ({h/len(df)*100:.1f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title("Customer Churn Target Distribution (IBM Telco Dataset)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Churn Status", fontsize=11)
    ax.set_ylabel("Number of Customers", fontsize=11)
    ax.set_ylim(0, max(churn_counts.values) * 1.15)
    plt.tight_layout()
    fig.savefig(os.path.join(VIZ_DIR, "01_target_distribution.png"), dpi=200)
    plt.close(fig)

    # 2. Churn vs Contract Type
    fig, ax = plt.subplots(figsize=(8, 5))
    contract_df = df.groupby(["Contract", "Churn"]).size().unstack(fill_value=0)
    contract_pct = contract_df.div(contract_df.sum(axis=1), axis=0) * 100
    contract_pct.plot(kind="bar", stacked=True, color=["#3b82f6", "#ef4444"], ax=ax, edgecolor="black")
    ax.set_title("Churn Rate by Contract Type (Month-to-Month vs Annual)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Contract Type", fontsize=11)
    ax.set_ylabel("Percentage (%)", fontsize=11)
    ax.legend(["No Churn (Retained)", "Churn (Attrited)"], loc="upper right")
    plt.xticks(rotation=0)
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        if height > 5:
            x, y = p.get_xy()
            ax.text(x + width/2, y + height/2, f"{height:.1f}%", ha="center", va="center", color="white", fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(VIZ_DIR, "02_churn_by_contract.png"), dpi=200)
    plt.close(fig)

    # 3. Churn vs Tenure Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x="tenure", hue="Churn", multiple="stack", palette={"No": "#3b82f6", "Yes": "#ef4444"},
                 bins=30, ax=ax, edgecolor="black")
    ax.set_title("Customer Tenure Distribution by Churn Status", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Tenure (Months with Company)", fontsize=11)
    ax.set_ylabel("Customer Count", fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(VIZ_DIR, "03_churn_by_tenure.png"), dpi=200)
    plt.close(fig)

    # 4. Churn vs Monthly Charges
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(data=df[df["Churn"] == "No"]["MonthlyCharges"], label="Retained (No)", color="#2563eb", fill=True, alpha=0.35, ax=ax)
    sns.kdeplot(data=df[df["Churn"] == "Yes"]["MonthlyCharges"], label="Churned (Yes)", color="#dc2626", fill=True, alpha=0.35, ax=ax)
    ax.set_title("Density of Monthly Charges by Churn Outcome", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Monthly Charges ($ USD)", fontsize=11)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(os.path.join(VIZ_DIR, "04_churn_by_monthly_charges.png"), dpi=200)
    plt.close(fig)

    # 5. Churn vs Internet Service Type
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x="InternetService", hue="Churn", palette={"No": "#3b82f6", "Yes": "#ef4444"}, ax=ax, edgecolor="black")
    ax.set_title("Customer Attrition by Internet Service Technology", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Internet Service Provider Type", fontsize=11)
    ax.set_ylabel("Customer Count", fontsize=11)
    ax.legend(title="Churn Status")
    plt.tight_layout()
    fig.savefig(os.path.join(VIZ_DIR, "05_churn_by_internet_service.png"), dpi=200)
    plt.close(fig)

    # 6. Churn vs Payment Method
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.countplot(data=df, y="PaymentMethod", hue="Churn", palette={"No": "#3b82f6", "Yes": "#ef4444"}, ax=ax, edgecolor="black")
    ax.set_title("Churn Rate Across Customer Payment Methods", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Customer Count", fontsize=11)
    ax.set_ylabel("Payment Channel", fontsize=11)
    ax.legend(title="Churn Status", loc="lower right")
    plt.tight_layout()
    fig.savefig(os.path.join(VIZ_DIR, "06_churn_by_payment_method.png"), dpi=200)
    plt.close(fig)

    # 7. Numerical Correlation Heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, cmap="Blues", fmt=".3f", linewidths=1, ax=ax, cbar_kws={"label": "Pearson Correlation"})
    ax.set_title("Correlation Matrix of Continuous Financial & Tenure Features", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    fig.savefig(os.path.join(VIZ_DIR, "07_numerical_correlation_heatmap.png"), dpi=200)
    plt.close(fig)
    print("✓ All 7 EDA plots generated successfully.")


def plot_model_comparisons(baseline_metrics: dict, tuned_metrics: dict):
    """Plot comparative evaluation metrics, ROC curves, and confusion matrices."""
    print("\n[EVAL] Plotting model comparison charts...")

    all_models = {**baseline_metrics, "Gradient Boosting (Tuned)": tuned_metrics["test_metrics"]}

    # Metric Comparison Grouped Bar Chart
    fig, ax = plt.subplots(figsize=(11, 6))
    metrics_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    display_names = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    model_names = list(all_models.keys())
    x = np.arange(len(display_names))
    width = 0.15

    palette = ["#64748b", "#f59e0b", "#3b82f6", "#10b981", "#8b5cf6"]
    for idx, (m_name, m_data) in enumerate(all_models.items()):
        values = [m_data[m] for m in metrics_names]
        ax.bar(x + (idx * width) - (len(all_models)*width/2) + width/2, values, width, label=m_name, color=palette[idx % len(palette)], edgecolor="black")

    ax.set_title("Comprehensive Machine Learning Model Comparison on Test Data", fontsize=14, fontweight="bold", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=11, fontweight="bold")
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("Score", fontsize=12)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(VIZ_DIR, "08_model_metrics_comparison.png"), dpi=200)
    plt.close(fig)

    # ROC Curves Comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Random Classifier (AUC = 0.50)")
    for idx, (m_name, m_data) in enumerate(all_models.items()):
        pts = m_data.get("roc_points", [])
        if pts:
            fpr = [p["fpr"] for p in pts]
            tpr = [p["tpr"] for p in pts]
            ax.plot(fpr, tpr, label=f"{m_name} (AUC = {m_data['roc_auc']:.3f})", linewidth=2)
    ax.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(os.path.join(VIZ_DIR, "09_models_roc_curves.png"), dpi=200)
    plt.close(fig)

    # Confusion Matrices Grid
    fig, axes = plt.subplots(1, len(all_models), figsize=(16, 3.8))
    for ax, (m_name, m_data) in zip(axes, all_models.items()):
        cm = m_data["confusion_matrix"]
        mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        sns.heatmap(mat, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                    xticklabels=["Pred: No", "Pred: Yes"], yticklabels=["Act: No", "Act: Yes"])
        ax.set_title(m_name, fontsize=10, fontweight="bold")
    plt.suptitle("Confusion Matrices on Test Set (N=1,409)", fontsize=13, fontweight="bold", y=1.05)
    plt.tight_layout()
    fig.savefig(os.path.join(VIZ_DIR, "10_confusion_matrices.png"), dpi=200)
    plt.close(fig)
    print("✓ Model comparison and evaluation plots generated.")


def main():
    print("=" * 70)
    print("CUSTOMER CHURN PREDICTION USING MACHINE LEARNING")
    print("College Project Pipeline Execution & Verification")
    print("=" * 70)

    # 1. Data Acquisition
    data_path = os.path.join(CURRENT_DIR, "data", "dataset.csv")
    df_raw = load_raw_data(data_path)
    print(f"\n[Step 1: Data Acquisition]")
    print(f"Loaded dataset: IBM Telco Customer Churn")
    print(f"Shape: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
    num_cols, cat_cols = get_feature_types(df_raw)
    print(f"Numerical features ({len(num_cols)}): {num_cols}")
    print(f"Categorical features ({len(cat_cols)}): {cat_cols}")

    # 2. Data Quality Checks
    print(f"\n[Step 2: Data Quality Checks]")
    missing_counts = df_raw.isnull().sum()
    empty_charges = (df_raw["TotalCharges"].astype(str).str.strip() == "").sum()
    duplicate_count = df_raw.duplicated().sum()
    print(f"- Standard missing values (NaN): {missing_counts.sum()}")
    print(f"- Empty string values in TotalCharges: {empty_charges} (fixed by imputing with 0.0)")
    print(f"- Duplicate records: {duplicate_count}")
    print(f"- Class balance: {df_raw['Churn'].value_counts(normalize=True).to_dict()}")

    # 3. EDA
    run_eda_and_save_visualizations(df_raw)

    # 4. Train-Test Split (Prevent Data Leakage)
    print(f"\n[Step 4: Train-Test Split]")
    X_train, X_test, y_train, y_test = prepare_train_test(df_raw, test_size=0.2, random_state=42)
    print(f"Training set: {X_train.shape[0]} samples (Churn: {y_train.mean()*100:.1f}%)")
    print(f"Test set:     {X_test.shape[0]} samples (Churn: {y_test.mean()*100:.1f}%)")
    print("Train/test split applied with stratification prior to calculating transformers.")

    # 5. Baseline Model Training
    print(f"\n[Step 5: Baseline Model Training (4 Algorithms)]")
    baseline_results, baseline_pipelines = train_baseline_models(X_train, y_train, X_test, y_test)

    print("\n--- BASELINE MODEL COMPARISON TABLE ---")
    print(f"{'Model':<24} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8} | {'F1':<8} | {'ROC-AUC':<8}")
    print("-" * 75)
    for name, m in baseline_results.items():
        print(f"{name:<24} | {m['accuracy']:<8.4f} | {m['precision']:<9.4f} | {m['recall']:<8.4f} | {m['f1']:<8.4f} | {m['roc_auc']:<8.4f}")

    # 6. Automatic Best Model Selection
    # Select based on highest ROC-AUC on holdout test set
    best_baseline_name = max(baseline_results.keys(), key=lambda k: baseline_results[k]["roc_auc"])
    print(f"\n[Step 6: Best Baseline Selection]")
    print(f"Automated selection: '{best_baseline_name}' achieved highest baseline ROC-AUC of {baseline_results[best_baseline_name]['roc_auc']:.4f}")

    # 7. Hyperparameter Tuning
    print(f"\n[Step 7: Hyperparameter Tuning via GridSearchCV]")
    tuned_pipeline, tuning_report = tune_best_model(
        X_train, y_train, X_test, y_test, model_name=best_baseline_name
    )
    print(f"Best Hyperparameters: {tuning_report['best_params']}")
    print(f"Best 5-Fold CV ROC-AUC: {tuning_report['best_cv_roc_auc']:.4f}")
    print(f"Tuned Test ROC-AUC: {tuning_report['test_metrics']['roc_auc']:.4f}")
    print(f"Tuned Test F1-Score: {tuning_report['test_metrics']['f1']:.4f}")

    # Compare Baseline vs Tuned
    b_auc = baseline_results[best_baseline_name]["roc_auc"]
    t_auc = tuning_report["test_metrics"]["roc_auc"]
    print(f"Optimization Delta: ROC-AUC {b_auc:.4f} -> {t_auc:.4f} ({(t_auc - b_auc)*100:+.2f}%)")

    # Plot evaluation visualizations
    plot_model_comparisons(baseline_results, tuning_report)

    # 8. Extract Feature Importance
    feature_importances = extract_feature_importances(tuned_pipeline, X_train)
    fig, ax = plt.subplots(figsize=(8, 6))
    f_names = [f["feature"] for f in feature_importances[:12]]
    f_vals = [f["importance"] for f in feature_importances[:12]]
    sns.barplot(x=f_vals, y=f_names, palette="viridis", ax=ax, edgecolor="black")
    ax.set_title(f"Top Predictive Features for Churn ({best_baseline_name})", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Relative Importance / Influence", fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(VIZ_DIR, "11_feature_importances.png"), dpi=200)
    plt.close(fig)

    # 9. Model Persistence
    model_save_path = os.path.join(CURRENT_DIR, "models", "customer_churn_model.joblib")
    save_pipeline(tuned_pipeline, model_save_path)

    # 10. Unseen Data Prediction Verification (Without Retraining)
    print(f"\n[Step 10: Unseen Customer Testing via Joblib Model]")
    loaded_model = load_model(model_save_path)
    unseen_predictions = []
    for cust in UNSEEN_CUSTOMERS:
        res = predict_single_customer(cust, model=loaded_model)
        unseen_predictions.append({
            "customer_id": cust["id"],
            "profile_name": cust["profile_name"],
            "prediction": res["prediction"],
            "churn_probability": res["churn_probability"],
            "churn_probability_percentage": res["churn_probability_percentage"],
            "verdict": res["verdict"],
            "diagnostics": res["diagnostics"],
            "customer_input": cust,
        })
        print(f"-> {cust['id']} ({cust['profile_name']}): {res['prediction']} ({res['churn_probability_percentage']})")

    # 11. Export Complete Experiment Results to JSON
    experiment_data = {
        "dataset_info": {
            "name": "IBM Telco Customer Churn Dataset",
            "source": "IBM Business Analytics / Kaggle Repository",
            "rows": int(df_raw.shape[0]),
            "columns": int(df_raw.shape[1]),
            "train_rows": int(X_train.shape[0]),
            "test_rows": int(X_test.shape[0]),
            "target_variable": "Churn",
            "class_distribution": {
                "No": int((df_raw["Churn"] == "No").sum()),
                "Yes": int((df_raw["Churn"] == "Yes").sum()),
                "churn_rate_pct": round(float((df_raw["Churn"] == "Yes").mean() * 100), 2),
            },
            "numerical_features": num_cols,
            "categorical_features": cat_cols,
            "data_quality": {
                "missing_values": int(missing_counts.sum()),
                "empty_total_charges_rows": int(empty_charges),
                "duplicates": int(duplicate_count),
                "resolution": "Coerced TotalCharges whitespace to float with 0.0 imputation for 0-tenure accounts; dropped customerID; one-hot encoded categories."
            }
        },
        "feature_engineering": explain_engineered_features(),
        "baseline_models": baseline_results,
        "best_baseline_model": best_baseline_name,
        "hyperparameter_tuning": tuning_report,
        "feature_importances": feature_importances,
        "metrics_explanations": explain_metrics(),
        "unseen_predictions": unseen_predictions,
    }

    # Save to data directory
    json_path = os.path.join(CURRENT_DIR, "data", "experiment_results.json")
    with open(json_path, "w") as f:
        json.dump(experiment_data, f, indent=2)
    print(f"\n✓ Experiment results saved to: {json_path}")

    # Also mirror into src/data/experiment_results.json for frontend web dashboard
    frontend_data_dir = os.path.join(CURRENT_DIR, "..", "src", "data")
    os.makedirs(frontend_data_dir, exist_ok=True)
    with open(os.path.join(frontend_data_dir, "experiment_results.json"), "w") as f:
        json.dump(experiment_data, f, indent=2)

    print("\n" + "=" * 70)
    print("SUCCESS: PIPELINE FULLY EXECUTED WITH REAL NUMBERS!")
    print("=" * 70)


if __name__ == "__main__":
    main()

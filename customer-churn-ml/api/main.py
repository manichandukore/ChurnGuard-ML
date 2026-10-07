"""
FastAPI Microservice for Customer Churn Prediction
==================================================
Production-style API serving real-time predictions from the persisted Joblib pipeline.
Features:
- Pydantic v2 schema validation for incoming customer attributes
- Model cached on application startup (zero retraining on request)
- Structured error handling & Swagger/OpenAPI docs
- Detailed diagnostics, probability scoring, and retention interventions
"""

import os
import sys
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predict import load_model, predict_single_customer, UNSEEN_CUSTOMERS


# Global model cache
pipeline_cache: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model pipeline into memory once on application startup."""
    try:
        print("[FastAPI] Loading persisted Joblib model pipeline...")
        pipeline_cache["model"] = load_model()
        print("[FastAPI] Pipeline loaded successfully and ready for inference.")
    except Exception as exc:
        print(f"[FastAPI] WARNING: Could not pre-load model: {exc}")
        pipeline_cache["model"] = None
    yield
    print("[FastAPI] Shutting down churn prediction service.")


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Deployment-ready REST API powered by optimized Machine Learning pipeline",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for dashboard access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CustomerFeatures(BaseModel):
    """Pydantic model validating incoming raw customer attributes."""

    gender: str = Field(default="Female", description="Gender ('Male' or 'Female')")
    SeniorCitizen: int = Field(default=0, ge=0, le=1, description="1 if senior citizen, 0 otherwise")
    Partner: str = Field(default="No", description="'Yes' or 'No'")
    Dependents: str = Field(default="No", description="'Yes' or 'No'")
    tenure: float = Field(default=12.0, ge=0.0, le=100.0, description="Months customer has stayed with company")
    PhoneService: str = Field(default="Yes", description="'Yes' or 'No'")
    MultipleLines: str = Field(default="No", description="'Yes', 'No', or 'No phone service'")
    InternetService: str = Field(default="DSL", description="'DSL', 'Fiber optic', or 'No'")
    OnlineSecurity: str = Field(default="No", description="'Yes', 'No', or 'No internet service'")
    OnlineBackup: str = Field(default="No", description="'Yes', 'No', or 'No internet service'")
    DeviceProtection: str = Field(default="No", description="'Yes', 'No', or 'No internet service'")
    TechSupport: str = Field(default="No", description="'Yes', 'No', or 'No internet service'")
    StreamingTV: str = Field(default="No", description="'Yes', 'No', or 'No internet service'")
    StreamingMovies: str = Field(default="No", description="'Yes', 'No', or 'No internet service'")
    Contract: str = Field(default="Month-to-month", description="'Month-to-month', 'One year', 'Two year'")
    PaperlessBilling: str = Field(default="Yes", description="'Yes' or 'No'")
    PaymentMethod: str = Field(
        default="Electronic check",
        description="'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'",
    )
    MonthlyCharges: float = Field(default=55.0, ge=0.0, description="Monthly subscription charges in USD")
    TotalCharges: Optional[float] = Field(default=660.0, ge=0.0, description="Cumulative charges billed to date")


class PredictionResponse(BaseModel):
    """Response schema matching college specification with rich extensions."""

    prediction: str = Field(..., description="'Churn' or 'No Churn'")
    churn_probability: float = Field(..., description="Calculated churn probability between 0.0 and 1.0")
    verdict: str
    diagnostics: Dict[str, Any]


@app.get("/", tags=["General"])
def read_root():
    """Root endpoint detailing project metadata and available endpoints."""
    return {
        "project": "Customer Churn Prediction Using Machine Learning",
        "status": "Online",
        "documentation": "/docs",
        "endpoints": {
            "health": "GET /health",
            "predict": "POST /predict",
            "examples": "GET /examples",
        },
    }


@app.get("/health", tags=["General"])
def health_check():
    """Health status and model residency verification."""
    is_loaded = pipeline_cache.get("model") is not None
    return {
        "status": "healthy" if is_loaded else "degraded",
        "model_loaded": is_loaded,
        "engine": "Scikit-Learn Pipeline via Joblib",
    }


@app.get("/examples", tags=["Inference"])
def get_examples():
    """Retrieve preconfigured unseen customer profiles for one-click testing."""
    return {"unseen_customers": UNSEEN_CUSTOMERS}


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_churn(customer: CustomerFeatures):
    """
    Main prediction endpoint. Accepts customer attributes and computes churn risk
    WITHOUT retraining the model.
    """
    model = pipeline_cache.get("model")
    if model is None:
        try:
            model = load_model()
            pipeline_cache["model"] = model
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model could not be loaded: {str(exc)}",
            )

    customer_dict = customer.model_dump()
    result = predict_single_customer(customer_dict, model=model)

    return PredictionResponse(
        prediction=result["prediction"],
        churn_probability=result["churn_probability"],
        verdict=result["verdict"],
        diagnostics=result["diagnostics"],
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

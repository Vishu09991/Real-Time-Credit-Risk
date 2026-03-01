from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import logging
from datetime import datetime
import shap
import pandas as pd

# ======================================================
# App Initialization
# ======================================================

app = FastAPI(
    title="Real-Time Credit Risk API",
    description="Production-style credit risk scoring engine with explainability and logging",
    version="2.0"
)

# ======================================================
# Load Model
# ======================================================

xgb_model = joblib.load("models/xgboost_model.pkl")

# Optimized threshold (update if needed)
DEFAULT_THRESHOLD = 0.45

# ======================================================
# Logging Configuration
# ======================================================

logging.basicConfig(
    filename="prediction_logs.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ======================================================
# SHAP Explainer Initialization
# ======================================================

data = pd.read_csv("data/credit_data.csv")
X_background = data.drop("default", axis=1)

explainer = shap.TreeExplainer(xgb_model)

# ======================================================
# Input Schema
# ======================================================

class Applicant(BaseModel):
    age: int
    income: float
    loan_amount: float
    credit_score: int
    employment_years: int

# ======================================================
# Health Endpoint
# ======================================================

@app.get("/health")
def health_check():
    return {
        "status": "API is running",
        "model_loaded": True,
        "threshold": DEFAULT_THRESHOLD,
        "timestamp": datetime.utcnow()
    }

# ======================================================
# Prediction Endpoint
# ======================================================

@app.post("/predict")
def predict(applicant: Applicant):

    feature_names = [
        "age",
        "income",
        "loan_amount",
        "credit_score",
        "employment_years"
    ]

    features = np.array([[
        applicant.age,
        applicant.income,
        applicant.loan_amount,
        applicant.credit_score,
        applicant.employment_years
    ]])

    # Get probability
    probability = xgb_model.predict_proba(features)[0][1]

    # Apply threshold
    prediction = int(probability >= DEFAULT_THRESHOLD)

    # Business decision
    if prediction == 1:
        risk_level = "High Risk"
        decision = "Reject Loan"
    else:
        risk_level = "Low Risk"
        decision = "Approve Loan"

    # ======================================================
    # SHAP Explanation
    # ======================================================

    shap_values = explainer.shap_values(features)[0]

    feature_contributions = dict(zip(feature_names, shap_values))

    sorted_features = sorted(
        feature_contributions.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    top_risk_factors = [f[0] for f in sorted_features[:3]]

    # ======================================================
    # Structured Logging
    # ======================================================

    logging.info(
        f"Applicant={applicant.dict()} | "
        f"Probability={probability:.4f} | "
        f"Prediction={prediction} | "
        f"Decision={decision} | "
        f"TopFactors={top_risk_factors}"
    )

    # ======================================================
    # Response
    # ======================================================

    return {
        "default_probability": round(float(probability), 4),
        "threshold_used": DEFAULT_THRESHOLD,
        "prediction": prediction,
        "risk_level": risk_level,
        "decision": decision,
        "top_risk_factors": top_risk_factors,
        "timestamp": datetime.utcnow()
    }
    
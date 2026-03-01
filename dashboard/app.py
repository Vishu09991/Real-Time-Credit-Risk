import streamlit as st
import numpy as np
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")

st.title("🏦 Real-Time Credit Risk Monitoring System")

# Load model
model = joblib.load("models/xgboost_model.pkl")

# Sidebar input
st.sidebar.header("Applicant Details")

age = st.sidebar.slider("Age", 21, 65, 35)
income = st.sidebar.number_input("Income", 20000, 200000, 50000)
loan_amount = st.sidebar.number_input("Loan Amount", 5000, 100000, 20000)
credit_score = st.sidebar.slider("Credit Score", 300, 850, 700)
employment_years = st.sidebar.slider("Employment Years", 0, 30, 5)

if st.sidebar.button("Predict Risk"):

    features = np.array([[age, income, loan_amount, credit_score, employment_years]])
    probability = model.predict_proba(features)[0][1]

    if probability < 0.30:
        risk = "Low"
    elif probability <= 0.60:
        risk = "Medium"
    else:
        risk = "High"

    st.subheader("📊 Prediction Result")
    st.metric("Default Probability", f"{probability:.2%}")
    st.metric("Risk Level", risk)

    # SHAP Explanation
    st.subheader("🔍 Feature Importance (SHAP)")

    data = pd.read_csv("data/credit_data.csv")
    X = data.drop("default", axis=1)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X, show=False)
    st.pyplot(fig)
# 🏦 Real-Time Credit Risk Prediction System

## 📌 Overview
A production-style credit risk scoring engine that predicts loan default probability using Logistic Regression and XGBoost. The system exposes a FastAPI REST API and a Streamlit dashboard with SHAP-based explainability.

---

## 🚀 Features
- Logistic Regression (baseline model)
- XGBoost (advanced model)
- ROC-AUC evaluation
- SHAP explainability
- FastAPI real-time prediction API
- Streamlit monitoring dashboard

---

## 📂 Project Structure

real_time_credit_risk/
│
├── data/
├── models/
├── utils/
├── api/
├── dashboard/
├── requirements.txt
└── README.md

---

## ⚙️ Installation

```bash
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
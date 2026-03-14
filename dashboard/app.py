import streamlit as st
import numpy as np
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import time

st.set_page_config(page_title="Credit Risk Intelligence", page_icon="🏦", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS for Premium UI ---
st.markdown("""
<style>
    /* Global Background and Typography */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Top Header Styling */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        text-align: center;
        text-shadow: 0px 4px 15px rgba(0,0,0,0.5);
    }
    .sub-header {
        font-size: 1.1rem;
        color: #94a3b8;
        text-align: center;
        margin-bottom: 3rem;
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        margin-bottom: 1.5rem;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        border-color: rgba(255,255,255,0.15);
    }
    
    /* Sidebar Improvements */
    [data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #1e293b;
    }
    [data-testid="stSidebar"] .css-17lntkn { /* target inner container if accessible */
        padding: 2rem 1rem;
    }
    .sidebar-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #334155;
    }

    /* Metric Styling */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        background: -webkit-linear-gradient(45deg, #f8fafc, #cbd5e1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
    }
    
    /* Status Badges */
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1.2rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .status-low { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(52, 211, 153, 0.3); }
    .status-medium { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(251, 191, 36, 0.3); }
    .status-high { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(248, 113, 113, 0.3); }
    
    /* Button Styling */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #3b82f6 0%, #6366f1 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 0rem;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 14px 0 rgba(99, 102, 241, 0.39);
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(90deg, #2563eb 0%, #4f46e5 100%);
        box-shadow: 0 6px 20px 0 rgba(99, 102, 241, 0.5);
        transform: translateY(-1px);
    }

    hr { border-color: #334155; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# Main Header
st.markdown('<div class="main-header">🏦 Real-Time Credit Risk Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-powered risk assessment with advanced interpretability</div>', unsafe_allow_html=True)

# Cache model and data loading
@st.cache_resource
def load_model():
    return joblib.load("models/xgboost_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("data/credit_data.csv")

try:
    model = load_model()
    data = load_data()
    X_background = data.drop("default", axis=1)
except Exception as e:
    st.error(f"Error loading assets. Please make sure the model and data files exist. Details: {e}")
    st.stop()

# --- Sidebar Inputs ---
with st.sidebar:
    st.markdown('<div class="sidebar-title">Applicant Profile 📋</div>', unsafe_allow_html=True)
    
    age = st.slider("🧑 Age", 18, 80, 35, help="Applicant's age in years")
    income = st.number_input("💰 Annual Income ($)", 10000, 500000, 50000, step=5000)
    loan_amount = st.number_input("💳 Requested Loan Amount ($)", 1000, 200000, 20000, step=1000)
    credit_score = st.slider("📈 Credit Score", 300, 850, 700, help="FICO Score (300-850)")
    employment_years = st.slider("💼 Employment Years", 0, 40, 5)
    
    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("🚀 Analyze Risk Profile")
    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption("powered by XGBoost & SHAP")

# --- Default View / Prediction Execution ---
if not predict_clicked:
    st.markdown("""
    <div class="glass-card" style="text-align: center; padding: 4rem 2rem;">
        <h2 style="color: #94a3b8; font-weight: 400;">Awaiting Applicant Data...</h2>
        <p style="color: #64748b;">Adjust the applicant profile in the sidebar and click <b>Analyze Risk Profile</b> to generate scoring.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    # Spinner for dramatic effect and real-time feel
    with st.spinner("Analyzing profile against risk models..."):
        time.sleep(0.6) # small artificial delay for UX processing feel
        
        # Prediction
        features = np.array([[age, income, loan_amount, credit_score, employment_years]])
        probability = model.predict_proba(features)[0][1]
        
        # Logic thresholds
        if probability < 0.30:
            risk = "Low Risk"
            risk_class = "status-low"
            gauge_color = "#34d399"
            recommendation = "✅ Approval Recommended. Applicant shows strong financial stability."
        elif probability <= 0.60:
            risk = "Medium Risk"
            risk_class = "status-medium"
            gauge_color = "#fbbf24"
            recommendation = "⚠️ Manual Review Suggested. Proceed with standard verification."
        else:
            risk = "High Risk"
            risk_class = "status-high"
            gauge_color = "#f87171"
            recommendation = "❌ Decline Advised. High probability of default detected."

    # --- Dashboard Layout ---
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🎯 Risk Assessment")
        
        # Plotly Gauge Chart for Probability
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = probability * 100,
            number = {'suffix': "%", 'font': {'size': 48, 'color': 'white'}},
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Default Probability", 'font': {'size': 18, 'color': '#94a3b8'}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': gauge_color},
                'bgcolor': "rgba(255,255,255,0.05)",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 30], 'color': 'rgba(52, 211, 153, 0.1)'},
                    {'range': [30, 60], 'color': 'rgba(251, 191, 36, 0.1)'},
                    {'range': [60, 100], 'color': 'rgba(248, 113, 113, 0.1)'}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': probability * 100
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "white"},
            height=250,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Status Badge & Decision
        st.markdown(f'<div style="text-align: center;"><div class="{risk_class} status-badge">{risk}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<p style="text-align: center; margin-top: 1rem; color: #cbd5e1;">{recommendation}</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📈 Financial Profile Summary")
        
        # Quick metrics in a 2x2 grid inside the card
        m1, m2 = st.columns(2)
        m1.metric("Debt-to-Income", f"{(loan_amount/income)*100:.1f}%")
        m2.metric("Credit Score", credit_score)
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Feature Importance insights text
        st.markdown("<b>Why did the AI make this decision?</b>", unsafe_allow_html=True)
        st.markdown("<p style='color: #94a3b8; font-size: 0.9rem;'>The SHAP analysis below reveals the driving factors behind this applicant's risk score. Bars extending to the right increase default risk, while bars to the left decrease it.</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # SHAP Explanation Full Width Card
    st.markdown('<div class="glass-card" style="margin-top: 0.5rem;">', unsafe_allow_html=True)
    st.subheader("🔮 Interpretability: Feature Attribution (SHAP)")
    
    with st.spinner("Computing Shapley values..."):
        explainer = shap.TreeExplainer(model)
        # Standardize feature names matching training data
        feature_names = ['age', 'income', 'loan_amount', 'credit_score', 'employment_years']
        X_applicant = pd.DataFrame(features, columns=feature_names)
        
        shap_values = explainer.shap_values(X_applicant)
        
        # Style matplotlib for dark theme
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor('#1e293b') # Match glass card (ish)
        ax.set_facecolor('#1e293b')
        
        # Generate summary bar plot for the SINGLE prediction
        shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0], feature_names=feature_names, show=False)
        
        # Tweak plot visuals
        plt.tight_layout()
        st.pyplot(fig)
        
    st.markdown('</div>', unsafe_allow_html=True)
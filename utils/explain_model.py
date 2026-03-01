import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

# 1️⃣ Load data
data = pd.read_csv("data/credit_data.csv")
X = data.drop("default", axis=1)

# 2️⃣ Load trained XGBoost model
model = joblib.load("models/xgboost_model.pkl")

# 3️⃣ Create SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# 4️⃣ Global feature importance
print("Generating SHAP summary plot...")
shap.summary_plot(shap_values, X)

# 5️⃣ Explain single prediction
sample = X.iloc[[0]]
sample_shap = explainer.shap_values(sample)

print("\nExplaining first applicant:")
shap.force_plot(explainer.expected_value, sample_shap, sample, matplotlib=True)

plt.show()
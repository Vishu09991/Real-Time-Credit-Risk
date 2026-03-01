import pandas as pd
import numpy as np

# Reproducibility
np.random.seed(42)

n = 3000

# Generate synthetic applicant features
data = pd.DataFrame({
    "age": np.random.randint(21, 65, n),
    "income": np.random.randint(25000, 150000, n),
    "loan_amount": np.random.randint(5000, 60000, n),
    "credit_score": np.random.randint(300, 850, n),
    "employment_years": np.random.randint(0, 30, n)
})

# -----------------------------
# Create Realistic Risk Factors
# -----------------------------

# Normalize components
credit_risk = (700 - data["credit_score"]) / 400
loan_ratio = data["loan_amount"] / data["income"]
employment_risk = 1 / (data["employment_years"] + 1)

# Weighted risk score (strong but not deterministic)
risk_score = (
    credit_risk * 0.6 +
    loan_ratio * 0.8 +
    employment_risk * 0.4
)

# Add controlled noise (real-world uncertainty)
noise = np.random.normal(0, 0.1, n)
risk_score = risk_score + noise

# Convert to probability using sigmoid function
probability = 1 / (1 + np.exp(-5 * (risk_score - 0.5)))

# Generate default outcome probabilistically
data["default"] = np.random.binomial(1, probability)

# Save dataset
data.to_csv("data/credit_data.csv", index=False)

print("Realistic dataset generated successfully.")
print("Default rate:", data["default"].mean())
import pandas as pd
import numpy as np

np.random.seed(42)
n = 2000

data = pd.DataFrame({
    "age": np.random.randint(21, 65, n),
    "income": np.random.randint(20000, 150000, n),
    "loan_amount": np.random.randint(5000, 50000, n),
    "credit_score": np.random.randint(300, 850, n),
    "employment_years": np.random.randint(0, 30, n)
})

data["default"] = (
    (data["credit_score"] < 600) &
    (data["loan_amount"] > 30000)
).astype(int)

data.to_csv("data/credit_data.csv", index=False)

print("Dataset generated successfully.")
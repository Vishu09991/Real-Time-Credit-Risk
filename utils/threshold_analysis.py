import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

# Load dataset
data = pd.read_csv("data/credit_data.csv")
X = data.drop("default", axis=1)
y = data["default"]

# Load model
model = joblib.load("models/xgboost_model.pkl")

# Get probabilities
y_proba = model.predict_proba(X)[:, 1]

thresholds = np.arange(0.1, 0.9, 0.05)

precision_list = []
recall_list = []
f1_list = []

for t in thresholds:
    y_pred = (y_proba >= t).astype(int)
    precision_list.append(precision_score(y, y_pred))
    recall_list.append(recall_score(y, y_pred))
    f1_list.append(f1_score(y, y_pred))

# Plot
plt.figure(figsize=(8, 5))
plt.plot(thresholds, precision_list, label="Precision")
plt.plot(thresholds, recall_list, label="Recall")
plt.plot(thresholds, f1_list, label="F1 Score")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Threshold Tuning Analysis")
plt.legend()
plt.show()

# Print best threshold based on F1
best_index = np.argmax(f1_list)
print("\nBest Threshold (F1 optimized):", thresholds[best_index])
print("Precision:", precision_list[best_index])
print("Recall:", recall_list[best_index])
print("F1 Score:", f1_list[best_index])
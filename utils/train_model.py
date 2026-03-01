import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from xgboost import XGBClassifier
import joblib

# 1️⃣ Load dataset
data = pd.read_csv("data/credit_data.csv")

# 2️⃣ Separate features and target
X = data.drop("default", axis=1)
y = data["default"]

# 3️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4️⃣ Scale features (only for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# 🔹 Logistic Regression
# -------------------------------
log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train)

y_pred_log = log_model.predict(X_test_scaled)
y_proba_log = log_model.predict_proba(X_test_scaled)[:, 1]

print("🔹 Logistic Regression Results")
print(f"Accuracy: {accuracy_score(y_test, y_pred_log):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba_log):.4f}")
print(classification_report(y_test, y_pred_log))

# -------------------------------
# 🔹 XGBoost Model
# -------------------------------
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric="logloss"
)

xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)
y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

print("\n🔹 XGBoost Results")
print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba_xgb):.4f}")
print(classification_report(y_test, y_pred_xgb))

# 5️⃣ Save models
joblib.dump(log_model, "models/logistic_model.pkl")
joblib.dump(xgb_model, "models/xgboost_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\nModels saved successfully.")
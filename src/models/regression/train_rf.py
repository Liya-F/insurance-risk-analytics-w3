# src/models/regression/train_rf.py

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import os

# Load claims-only dataset
claims_data = pd.read_csv("../../../data/claims_only.csv")

# Define features and target
X = claims_data.drop(columns=["TotalClaims"])
y = claims_data["TotalClaims"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = rf_model.predict(X_test)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Random Forest Model Evaluation:")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

# Save the model
output_path = "../models/rf_regression_claims_model.joblib"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
joblib.dump(rf_model, output_path)
print(f"Model saved to: {output_path}")

# Random Forest Model Evaluation:
# RMSE: 6289.8085
# R²: 0.9754
# Model saved to: ../models/rf_regression_claims_model.joblib
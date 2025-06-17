import pandas as pd
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import os

# Load the claims-only dataset
claims_df = pd.read_csv("../../../data/claims_only.csv")

# Prepare features and target
X = claims_df.drop(columns=["TotalClaims"])
y = claims_df["TotalClaims"]

# Train-test split (80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the XGBoost model
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)
xgb_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = xgb_model.predict(X_test)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print("XGBoost Model Evaluation:")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

# Save the model
model_path = "../models/xgb_regression_claims_model.joblib"
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(xgb_model, model_path)
print(f"Model saved to: {model_path}")

# XGBoost Model Evaluation:
# RMSE: 6417.3070
# R²: 0.9744
# Model saved to: ../models/xgb_regression_claims_model.joblib
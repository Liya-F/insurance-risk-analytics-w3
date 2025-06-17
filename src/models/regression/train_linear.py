import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import numpy as np

def train_linear_regression(input_path: str, model_output_path: str):
    # Load preprocessed data (use your claims_only.csv or appropriate dataset)
    df = pd.read_csv(input_path)

    # Filter to only rows where HasClaim == True (or TotalClaims > 0)
    df = df[df['TotalClaims'] > 0].copy()

    # Define target and features
    target = 'TotalClaims'
    
    # Drop target and any irrelevant columns
    # You can customize this list based on your data
    drop_cols = ['TotalClaims', 'TransactionMonth']  
    
    X = df.drop(columns=drop_cols)
    y = df[target]

    # Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Evaluate
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"Linear Regression Model Evaluation:")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")

    # Save the model
    joblib.dump(model, model_output_path)
    print(f"Model saved to: {model_output_path}")

if __name__ == "__main__":
    input_path = "../../../data/claims_only.csv" 
    model_output_path = "../models/linear_regression_claims_model.joblib"
    train_linear_regression(input_path, model_output_path)

# Linear Regression Model Evaluation:
# RMSE: 33395.1153
# R²: 0.3066
# Model saved to: ../models/linear_regression_claims_model.joblib
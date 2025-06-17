import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Load dataset
data = pd.read_csv('../../../data/classification.csv')  

# Separate features and target
X = data.drop(columns=['HasClaim'])
y = data['HasClaim']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Train Logistic Regression
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Logistic Regression Model Evaluation:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

# Save model
joblib.dump(model, '../models/logistic_regression_claims_model.joblib')
print("Model saved to: ../models/logistic_regression_claims_model.joblib")

# Logistic Regression Model Evaluation:
# Accuracy: 1.0000
# Precision: 1.0000
# Recall: 0.9982
# F1-score: 0.9991
# Model saved to: ../models/logistic_regression_claims_model.joblib
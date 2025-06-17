import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import joblib

# Load data
data = pd.read_csv('../../../data/classification.csv')

# Define features and target
X = data.drop(columns=['HasClaim'])
y = data['HasClaim']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Initialize XGBoost classifier
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Train model
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("XGBoost Classification Model Evaluation:")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-score: {f1:.4f}")

# Save the model
joblib.dump(model, '../models/xgb_classification_claims_model.joblib')
print("Model saved to: ../models/xgb_classification_claims_model.joblib")

# XGBoost Classification Model Evaluation:
# Accuracy: 0.9972
# Precision: 0.5714
# Recall: 0.0048
# F1-score: 0.0095
# Model saved to: ../models/xgb_classification_claims_model.joblib
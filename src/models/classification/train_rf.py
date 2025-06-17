import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Load data
data = pd.read_csv('../../../data/classification.csv')

# Features and target
X = data.drop(columns=['HasClaim'])
y = data['HasClaim']

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize and train RF classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Random Forest Model Evaluation:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

# Save the model
joblib.dump(rf, '../models/rf_classification_claims_model.joblib')
print("Model saved to: ../models/rf_classification_claims_model.joblib")

# Random Forest Model Evaluation:
# Accuracy: 1.0000
# Precision: 1.0000
# Recall: 1.0000
# F1-score: 1.0000
# Model saved to: ../models/rf_classification_claims_model.joblib
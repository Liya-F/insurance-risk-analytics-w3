import os

# Project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Data paths
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "MachineLearningRating_v3.txt")
CLEAN_DATA_PATH = os.path.join(BASE_DIR, "data", "cleaned_full.csv")

# Model output paths
MODEL_DIR = os.path.join(BASE_DIR, "src", "models")

CLASSIFICATION_MODELS = {
    "logistic": os.path.join(MODEL_DIR, "logistic_regression_claims_model.joblib"),
    "rf": os.path.join(MODEL_DIR, "rf_classification_claims_model.joblib"),
    "xgb": os.path.join(MODEL_DIR, "xgb_classification_claims_model.joblib"),
}

REGRESSION_MODELS = {
    "linear": os.path.join(MODEL_DIR, "linear_regression_claims_model.joblib"),
    "rf": os.path.join(MODEL_DIR, "rf_regression_claims_model.joblib"),
    "xgb": os.path.join(MODEL_DIR, "xgb_regression_claims_model.joblib"),
}

# Columns
TARGET_CLASSIFICATION = "HasClaim"
TARGET_REGRESSION = "TotalClaims"

# Random seed for reproducibility
RANDOM_STATE = 42

# Train/test split size
TEST_SIZE = 0.2

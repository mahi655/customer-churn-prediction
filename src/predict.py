from pathlib import Path

import joblib
import pandas as pd



# Project root directory

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Model paths

PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "preprocessor.joblib"
MODEL_PATH = PROJECT_ROOT / "models" / "xgb_churn_model.joblib"
THRESHOLD_PATH = PROJECT_ROOT / "models" / "churn_threshold.joblib"


# Load trained components

preprocessor = joblib.load(PREPROCESSOR_PATH)
model = joblib.load(MODEL_PATH)
threshold = joblib.load(THRESHOLD_PATH)


# Prediction function

def predict_churn(customer_data: dict) -> dict:
    """
    Predict whether a customer is likely to churn.

    Parameters
    ----------
    customer_data : dict
        Customer feature values.

    Returns
    -------
    dict
        Churn prediction and churn probability.
    """

    # Convert input dictionary to DataFrame
    customer_df = pd.DataFrame([customer_data])

    # Convert signup_date to datetime
    customer_df["signup_date"] = pd.to_datetime(
        customer_df["signup_date"]
    )

    # Create date-based features
    customer_df["signup_year"] = customer_df["signup_date"].dt.year
    customer_df["signup_month"] = customer_df["signup_date"].dt.month

    # Remove original date column
    customer_df = customer_df.drop(
        columns=["signup_date"]
    )

    # Apply the already-fitted preprocessing pipeline
    customer_processed = preprocessor.transform(
        customer_df
    )

    # Get probability of churn
    churn_probability = model.predict_proba(
        customer_processed
    )[:, 1][0]

    # Apply selected classification threshold
    prediction = int(
        churn_probability >= threshold
    )

    return {
        "churn_prediction": prediction,
        "churn_probability": float(churn_probability)
    }
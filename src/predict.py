from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "preprocessor.joblib"
MODEL_PATH = PROJECT_ROOT / "models" / "xgb_churn_model.joblib"
THRESHOLD_PATH = PROJECT_ROOT / "models" / "churn_threshold.joblib"


@lru_cache(maxsize=1)
def load_model_artifacts():
    """
    Load the preprocessing pipeline, model, and threshold once.
    """

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)
    threshold = joblib.load(THRESHOLD_PATH)

    return preprocessor, model, threshold


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

    preprocessor, model, threshold = load_model_artifacts()

    customer_df = pd.DataFrame([customer_data])

    customer_df["signup_date"] = pd.to_datetime(
        customer_df["signup_date"]
    )

    customer_df["signup_year"] = customer_df["signup_date"].dt.year
    customer_df["signup_month"] = customer_df["signup_date"].dt.month

    customer_df = customer_df.drop(
        columns=["signup_date"]
    )

    customer_processed = preprocessor.transform(
        customer_df
    )

    churn_probability = model.predict_proba(
        customer_processed
    )[:, 1][0]

    prediction = int(
        churn_probability >= threshold
    )

    return {
        "churn_prediction": prediction,
        "churn_probability": float(churn_probability)
    }
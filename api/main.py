from fastapi import FastAPI
from pydantic import BaseModel

from src.predict import predict_churn


app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn.",
    version="1.0.0"
)


class CustomerData(BaseModel):
    age: int
    gender: str
    annual_income: float
    education: str
    marital_status: str
    dependents: int
    tenure: int
    contract: str
    payment_method: str
    paperless_billing: str
    senior_citizen: int
    monthlycharges: float
    totalcharges: float
    num_services: int
    has_phone_service: int
    has_internet_service: int
    has_online_security: int
    has_online_backup: int
    has_device_protection: int
    has_tech_support: int
    has_streaming_tv: int
    has_streaming_movies: int
    customer_satisfaction: int
    num_complaints: int
    num_service_calls: int
    late_payments: int
    avg_monthly_gb: float
    days_since_last_interaction: int
    credit_score: float
    signup_date: str


@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running"
    }


@app.post("/predict")
def predict(customer: CustomerData):

    customer_data = customer.model_dump()

    result = predict_churn(customer_data)

    return result
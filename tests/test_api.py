from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_predict_endpoint():
    customer_data = {
        "age": 35,
        "gender": "Male",
        "annual_income": 55000,
        "education": "Bachelor's",
        "marital_status": "Single",
        "dependents": 0,
        "tenure": 8,
        "contract": "month_to_month",
        "payment_method": "credit_card",
        "paperless_billing": "Yes",
        "senior_citizen": 0,
        "monthlycharges": 85.5,
        "totalcharges": 684.0,
        "num_services": 2,
        "has_phone_service": 1,
        "has_internet_service": 1,
        "has_online_security": 0,
        "has_online_backup": 0,
        "has_device_protection": 0,
        "has_tech_support": 0,
        "has_streaming_tv": 1,
        "has_streaming_movies": 1,
        "customer_satisfaction": 3,
        "num_complaints": 3,
        "num_service_calls": 5,
        "late_payments": 2,
        "avg_monthly_gb": 150.0,
        "days_since_last_interaction": 30,
        "credit_score": 650,
        "signup_date": "2025-05-15"
    }

    response = client.post("/predict", json=customer_data)

    assert response.status_code == 200

    result = response.json()

    assert "churn_prediction" in result
    assert "churn_probability" in result

    assert result["churn_prediction"] in [0, 1]
    assert 0 <= result["churn_probability"] <= 1
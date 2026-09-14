# Customer Churn Prediction & Retention System

An end-to-end machine learning system that predicts whether a customer is likely to churn and provides churn predictions through a REST API built with FastAPI and Docker.

---

## Project Overview

Customer churn prediction helps businesses identify customers who are at higher risk of leaving their service.

This project demonstrates a complete machine learning workflow, from raw customer data analysis to model development and API deployment.

The project includes:

- Data cleaning and validation
- Exploratory data analysis (EDA)
- Feature engineering
- Train/test splitting
- Missing-value handling
- Categorical encoding
- Feature scaling
- Multiple machine learning models
- Stratified cross-validation
- Hyperparameter tuning
- Classification threshold tuning
- Model evaluation
- FastAPI deployment
- Docker containerization
- Automated testing
- GitHub Actions CI/CD
- Docker image publishing through GitHub Container Registry

---

## Dataset

The project uses the **Customer Churn Prediction Dataset 1M** from Kaggle.

Dataset size:

- 1,000,000 customer records
- 32 original features

The dataset contains:

- Customer demographics
- Contract and payment information
- Service usage
- Billing information
- Customer satisfaction
- Complaints and service calls
- Payment behavior
- Churn status

The dataset is not included in this repository because of its size.

---

## Machine Learning Workflow

```text
Raw Customer Data
        ↓
Data Understanding & Validation
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Train / Test Split
        ↓
Feature Engineering
        ↓
Missing Value Imputation
        ↓
Categorical Encoding
        ↓
Feature Scaling
        ↓
Model Training
        ↓
Stratified Cross-Validation
        ↓
Hyperparameter Tuning
        ↓
Classification Threshold Tuning
        ↓
Final Model
        ↓
FastAPI
        ↓
Docker
        ↓
GitHub Actions CI/CD
        ↓
GitHub Container Registry
```

---

## Exploratory Data Analysis

Exploratory data analysis was performed to understand customer behavior and identify patterns associated with churn.

Key observations included:

- Customers on month-to-month contracts had substantially higher churn rates than customers on longer-term contracts.
- Lower customer satisfaction was associated with higher churn.
- Customers with more complaints showed higher churn rates.
- Customers with more service calls showed higher churn rates.
- Customers with more late payments showed higher churn rates.
- Customers with more services generally showed lower churn rates.
- Tenure showed a mild downward relationship with churn.
- Several demographic features showed relatively weak differences in churn rates.

These findings helped guide feature analysis and model development.

---

## Data Preprocessing

The preprocessing pipeline was implemented using Scikit-learn.

### Numerical Features

Numerical features were processed using:

- Median imputation for missing values
- Standard scaling

### Categorical Features

Categorical features were processed using:

- Most-frequent-value imputation
- One-hot encoding
- Unknown-category handling

The preprocessing steps were implemented using a Scikit-learn `ColumnTransformer` and `Pipeline`.

The preprocessing pipeline was fitted only on the training data and then applied to the test data to prevent data leakage.

The trained preprocessing pipeline is saved as:

```text
models/preprocessor.joblib
```

### Feature Engineering

The `signup_date` feature was converted into:

- `signup_year`
- `signup_month`

The original date column was then removed before model training.

The `customer_id` column was removed because it is an identifier and does not provide useful predictive information.

---

## Model Development

Three classification models were trained and evaluated:

- Logistic Regression
- Random Forest
- XGBoost

The models were evaluated using the same preprocessing pipeline and a stratified train/test split.

Because the churn target is imbalanced, accuracy alone was not used to select the final model.

The primary evaluation metric was **F1-score**, with precision and recall also considered.

### Cross-Validation

Stratified 5-fold cross-validation was used to evaluate model performance while maintaining the class distribution across folds.

### Hyperparameter Tuning

XGBoost hyperparameters were optimized using `RandomizedSearchCV` with stratified 5-fold cross-validation.

The selected XGBoost configuration was:

```text
n_estimators      = 100
max_depth         = 7
learning_rate     = 0.05
subsample         = 0.7
colsample_bytree  = 1.0
min_child_weight  = 5
```

---

## Model Evaluation

### Logistic Regression

- Accuracy: **62.77%**
- Precision: **15.86%**
- Recall: **63.95%**
- F1-score: **25.42%**

### Random Forest

- Accuracy: **89.86%**
- Precision: **35.94%**
- Recall: **2.85%**
- F1-score: **5.28%**

Although Random Forest achieved high accuracy, its recall for churned customers was very low. This demonstrates why accuracy alone is not sufficient for this imbalanced classification problem.

### XGBoost Baseline

- Accuracy: **63.24%**
- Precision: **15.92%**
- Recall: **63.20%**
- F1-score: **25.44%**

### Tuned XGBoost

- Accuracy: **63.06%**
- Precision: **15.89%**
- Recall: **63.41%**
- F1-score: **25.41%**

---

## Classification Threshold Tuning

The default classification threshold of `0.50` was not treated as automatically optimal.

A validation set was used to evaluate different classification thresholds and identify a better precision-recall balance.

The selected threshold was:

```text
0.57
```

Using the tuned threshold improved the final test-set F1-score compared with the default classification threshold.

The selected threshold is saved as:

```text
models/churn_threshold.joblib
```

---

## Final Model

The final production model is a tuned **XGBoost classifier** using a classification threshold of **0.57**.

Final test-set performance:

- Accuracy: **75.05%**
- Precision: **18.97%**
- Recall: **46.30%**
- F1-score: **26.91%**

The trained model is saved as:

```text
models/xgb_churn_model.joblib
```

The model, preprocessing pipeline, and classification threshold are loaded by the prediction service at runtime.

---

## API

The trained model is exposed through a REST API built with **FastAPI**.

### API Endpoints

```text
GET  /
POST /predict
```

The `/predict` endpoint accepts customer information and returns:

- Churn prediction
- Churn probability

### Start the API Locally

Install the project dependencies:

```bash
pip install -r requirements.txt
```

Start the FastAPI application:

```bash
uvicorn api.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

---

## Prediction Example

### Request

```json
{
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
```

### Response

```json
{
  "churn_prediction": 1,
  "churn_probability": 0.9213
}
```

A `churn_prediction` of `1` indicates that the customer is classified as likely to churn.

A `churn_prediction` of `0` indicates that the customer is classified as not likely to churn.

The `churn_probability` represents the model's estimated probability of churn.

---

## Docker

The FastAPI application and trained model are packaged into a Docker image.

### Build the Docker Image

```bash
docker build -t customer-churn-api .
```

### Run the Container

```bash
docker run -p 8000:8000 customer-churn-api
```

The API documentation can then be accessed at:

```text
http://127.0.0.1:8000/docs
```

---

## CI/CD

GitHub Actions is used to automate testing and Docker image publishing.

The CI workflow performs the following steps:

```text
Code Push / Pull Request
        ↓
Install Dependencies
        ↓
Run Pytest
        ↓
Build Docker Image
        ↓
If Push to main
        ↓
Login to GitHub Container Registry
        ↓
Tag Docker Image
        ↓
Push Docker Image
```

The Docker image is published to GitHub Container Registry:

```text
ghcr.io/mahi655/customer-churn-prediction:latest
```

The workflow publishes the Docker image automatically when changes are pushed to the `main` branch.

---

## Testing

API tests are implemented using:

- Pytest
- FastAPI TestClient

Run the tests with:

```bash
python -m pytest
```

The tests verify:

- API endpoint availability
- Successful prediction response
- Presence of prediction fields
- Valid prediction values
- Valid churn probability range

---

## Project Structure

```text
customer-churn-prediction/
│
├── api/
│   └── main.py
│
├── data/
│   └── customer_churn_1M.csv
│
├── models/
│   ├── preprocessor.joblib
│   ├── xgb_churn_model.joblib
│   └── churn_threshold.joblib
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   └── 02_data_preprocessing.ipynb
│
├── src/
│   └── predict.py
│
├── tests/
│   └── test_api.py
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md
```

> The dataset CSV files are excluded from Git tracking because of their size.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- FastAPI
- Pytest
- Docker
- Git
- GitHub
- GitHub Actions
- GitHub Container Registry

---

## Key Skills Demonstrated

This project demonstrates practical experience in:

- Data cleaning and validation
- Exploratory data analysis
- Feature engineering
- Imbalanced classification
- Model comparison
- Cross-validation
- Hyperparameter tuning
- Classification threshold optimization
- Model persistence
- REST API development
- Automated testing
- Docker containerization
- CI/CD automation
- GitHub Container Registry

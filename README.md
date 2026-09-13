# Customer Churn Prediction & Retention System

A machine learning system that predicts whether a customer is likely to churn and exposes the trained model through a REST API using FastAPI and Docker.

## Project Overview

Customer churn prediction helps businesses identify customers who are likely to leave their service. This project builds an end-to-end machine learning pipeline covering data analysis, preprocessing, model training, evaluation, threshold tuning, and deployment.

## Dataset

The project uses the **Customer Churn Prediction Dataset 1M** from Kaggle.

Dataset size:
- 1,000,000 customer records
- 32 original features

The dataset contains customer demographics, tenure, contract information, billing information, service usage, satisfaction, complaints, payment behavior, and churn status.

The dataset is not included in this repository because of its size.

## Machine Learning Workflow

```text
Raw Data
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Train/Test Split
   ↓
Feature Engineering
   ↓
Preprocessing
   ↓
Model Training
   ↓
Cross-Validation
   ↓
Hyperparameter Tuning
   ↓
Threshold Tuning
   ↓
Final Model
   ↓
FastAPI
   ↓
Docker

## Model Performance

Multiple machine learning models were trained and evaluated using the same preprocessing pipeline.

The primary evaluation metric was **F1-score** because the target variable is imbalanced, with approximately 90% non-churned customers and 10% churned customers.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.628 | 0.159 | 0.639 | 0.254 |
| Random Forest | 0.899 | 0.359 | 0.029 | 0.053 |
| XGBoost Baseline | 0.632 | 0.159 | 0.632 | 0.254 |
| XGBoost Tuned | 0.631 | 0.159 | 0.634 | 0.254 |
| XGBoost + Threshold Tuning | 0.752 | 0.190 | 0.459 | **0.269** |

### Final Model

The final model is a tuned **XGBoost classifier** with a decision threshold of **0.57**.

Final test-set performance:

- Accuracy: **75.05%**
- Precision: **18.97%**
- Recall: **46.30%**
- F1-score: **26.91%**

Threshold tuning was used to improve the balance between precision and recall for churn detection instead of relying on the default classification threshold of 0.50.
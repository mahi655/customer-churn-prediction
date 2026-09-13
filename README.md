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
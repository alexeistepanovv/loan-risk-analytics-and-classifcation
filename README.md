# Loan Approval Prediction System

This project provides a **loan approval prediction system** based on a variety of machine learning techniques, designed to classify applicants based on their likelihood of loan approval. The model is deployed via a **Flask web application** with a survey interface for user-friendly input and real-time predictions.

## 📑 Project Overview

The objective is to predict loan approval probabilities using applicant details. The project involves data preprocessing, feature engineering, and training a machine learning model to achieve high prediction accuracy.

## Features

- **Data Preprocessing**: Cleans and transforms application data.
- **Modeling**: Trains multiple models and selects the best-performing one (CatBoost).
- **Deployment**: Provides a Flask web app for real-time loan approval predictions.

## 📂 Project Structure
loan_approve/
├── data/                     # Contains labeled and engineered datasets
├── models/                   # Stores the trained CatBoost model
├── notebooks/                # Jupyter notebooks for EDA, feature engineering, and modeling
├── src/                      # Scripts and common functions for data processing and modeling
└── webapp/                   # Flask app with HTML form template for data input

## 🔑 Key Components
Exploratory Data Analysis (eda_vintage_survival.ipynb): Analyzes and visualizes dataset characteristics. Includes vintage analysis for labeling and survival analysis for predictor analytics.
Feature Engineering (feature_engineering.ipynb): Engineers relevant features for modeling.
Model Selection (modeling.ipynb): Trains, evaluates, and selects the best model.
Flask Web App (webapp/app.py): Deploys the model for real-time loan predictions.

## 🏆 Model Performance
The CatBoost model was chosen for deployment, achieving the best accuracy among evaluated models. Further metrics, including precision, recall, and AUC, are detailed in the modeling.ipynb notebook.

## 🚀 Future Improvements
Enhance model accuracy with further feature engineering and hyperparameter tuning.
Extend form validations and error handling in the survey interface.
Containerize the Flask app using Docker for easier deployment and scaling.


# telecom-usage-forecasting-churn-prediction-
Telecom usage forecasting and churn prediction system using machine learning to analyze customer behavior, predict next-month data usage, and identify high-risk churn customers for targeted retention strategies.

# Telecom Network – Customer Usage Forecasting & Churn Risk Classification

## Overview
This project builds a data-driven system to forecast next-month customer
data usage and classify churn risk for a telecom company using machine
learning models. It helps identify high-risk customers and optimize mobile
plan recommendations based on usage behavior.

## Objectives
- Predict next-month data usage using historical behavior
- Identify customers likely to churn
- Analyze key behavioral factors influencing churn
- Provide actionable insights for retention strategies

## Models Used
- Linear Regression (Usage Forecasting)
- Support Vector Machine (SVM – Churn Classification)
- XGBoost (Advanced Churn Classification)

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn

## Project Structure

telecom-usage-forecasting-churn-prediction/
│
├── README.md                     # Project overview, instructions, and results
├── requirements.txt              # Python libraries required
├── .gitignore                    # Files to ignore in Git
│
├── data/
│   └── sample_data.csv           # Small sample dataset for demonstration
│
├── notebooks/
│   ├── exploratory_data_analysis.ipynb   # Understand data, plots, insights
│   ├── usage_forecasting.ipynb           # Linear regression experiments
│   └── churn_classification.ipynb        # SVM & XGBoost experiments
│
├── src/
│   ├── data_preprocessing.py     # Clean & encode data
│   ├── feature_engineering.py    # Add derived behavioral features
│   ├── train_forecasting_model.py # Train Linear Regression for usage
│   ├── train_churn_model.py      # Train SVM & XGBoost for churn
│   └── evaluate_models.py        # Evaluate models & generate plots
│
├── models/
│   ├── svm_churn_model.pkl       # Saved SVM model
│   └── xgboost_churn_model.pkl   # Saved XGBoost model
│
├── results/
│   ├── usage_forecast_plot.png       # Visualization of forecast results
│   ├── confusion_matrix.png          # Churn classification evaluation
│   └── churn_feature_importance.png  # Key factors influencing churn
│
└── docs/
    └── system_architecture.png       # Workflow diagram showing pipeline

## Results
- Accurate next-month data usage forecasting using Linear Regression
- High-accuracy churn prediction using SVM and XGBoost
- Feature importance analysis highlights key behavioral factors
- Visualization supports data-driven retention strategies

## Installation
Clone the repository and install dependencies:

```bash
git clone <your-repo-url>
cd telecom-usage-forecasting-churn-prediction
pip install -r requirements.txt



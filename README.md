# Customer Churn Prediction

A complete machine learning project that predicts whether a customer is likely to leave a service based on historical customer behavior.

## Description

Customer Churn Prediction is a classification-based machine learning project that analyzes customer profile, billing, contract, and service usage data to estimate whether a customer is likely to stop using a service. The project includes data preprocessing, model training, evaluation reports, feature-importance analysis, a saved prediction model, a command-line predictor, and a Streamlit app for interactive churn prediction.

## Project Goals

- Predict customer churn using classification models.
- Analyze customer behavior and identify important churn factors.
- Save a reusable trained model pipeline.
- Provide simple command-line and Streamlit prediction interfaces.

## Folder Structure

```text
Codec Project 1/
  data/
    customer_churn_sample.csv
  models/
    .gitkeep
  reports/
    .gitkeep
  src/
    app.py
    predict.py
    train_model.py
  README.md
  requirements.txt
```

## Dataset

The included sample dataset is synthetic and small enough for demonstration. You can replace `data/customer_churn_sample.csv` with a real churn dataset using the same column names.

Target column:

- `Churn`: `Yes` or `No`

Feature columns:

- `gender`
- `SeniorCitizen`
- `Partner`
- `Dependents`
- `tenure`
- `PhoneService`
- `InternetService`
- `Contract`
- `PaperlessBilling`
- `PaymentMethod`
- `MonthlyCharges`
- `TotalCharges`

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train the Model

```bash
python src/train_model.py
```

This creates:

- `models/churn_model.joblib`
- `reports/metrics.txt`
- `reports/feature_importance.csv`

## Predict One Customer

```bash
python src/predict.py --gender Female --senior 0 --partner Yes --dependents No --tenure 12 --phone Yes --internet Fiber --contract Month-to-month --paperless Yes --payment "Electronic check" --monthly 80.5 --total 966.0
```

## Run Web App

```bash
streamlit run src/app.py
```

## Classification Techniques Used

The training pipeline uses:

- One-hot encoding for categorical values
- Standard scaling for numeric values
- Logistic Regression classifier
- Random Forest classifier
- Cross-validation model comparison
- Feature importance analysis from the final Random Forest model

## Key Churn Factors Commonly Seen

In churn datasets, the strongest factors are often:

- Month-to-month contracts
- Higher monthly charges
- Lower tenure
- Electronic check payment method
- Fiber internet plans with high support or billing friction
- Paperless billing behavior patterns

Use `reports/feature_importance.csv` after training to inspect factors from this project data.

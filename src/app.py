"""Streamlit app for interactive customer churn prediction."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.title("Customer Churn Prediction")
st.caption("Estimate whether a customer may leave the service.")

if not MODEL_PATH.exists():
    st.warning("Model not found. Run `python src/train_model.py` first.")
    st.stop()

model = joblib.load(MODEL_PATH)

with st.form("customer_form"):
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda value: "Yes" if value else "No")
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure in months", min_value=0, max_value=72, value=12)
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber", "No"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
    monthly = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=80.0, step=1.0)
    total = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=960.0, step=10.0)
    submitted = st.form_submit_button("Predict")

if submitted:
    customer = pd.DataFrame(
        [
            {
                "gender": gender,
                "SeniorCitizen": senior,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone,
                "InternetService": internet,
                "Contract": contract,
                "PaperlessBilling": paperless,
                "PaymentMethod": payment,
                "MonthlyCharges": monthly,
                "TotalCharges": total,
            }
        ]
    )
    probability = model.predict_proba(customer)[0, 1]
    prediction = "Likely to churn" if probability >= 0.5 else "Not likely to churn"

    st.metric("Prediction", prediction)
    st.metric("Churn Probability", f"{probability:.2%}")

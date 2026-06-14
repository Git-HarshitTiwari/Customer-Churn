"""Predict churn for a single customer from command-line arguments."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict customer churn probability.")
    parser.add_argument("--gender", required=True, choices=["Female", "Male"])
    parser.add_argument("--senior", required=True, type=int, choices=[0, 1])
    parser.add_argument("--partner", required=True, choices=["Yes", "No"])
    parser.add_argument("--dependents", required=True, choices=["Yes", "No"])
    parser.add_argument("--tenure", required=True, type=int)
    parser.add_argument("--phone", required=True, choices=["Yes", "No"])
    parser.add_argument("--internet", required=True, choices=["DSL", "Fiber", "No"])
    parser.add_argument("--contract", required=True, choices=["Month-to-month", "One year", "Two year"])
    parser.add_argument("--paperless", required=True, choices=["Yes", "No"])
    parser.add_argument(
        "--payment",
        required=True,
        choices=["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
    )
    parser.add_argument("--monthly", required=True, type=float)
    parser.add_argument("--total", required=True, type=float)
    return parser.parse_args()


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run `python src/train_model.py` first.")

    args = parse_args()
    model = joblib.load(MODEL_PATH)
    customer = pd.DataFrame(
        [
            {
                "gender": args.gender,
                "SeniorCitizen": args.senior,
                "Partner": args.partner,
                "Dependents": args.dependents,
                "tenure": args.tenure,
                "PhoneService": args.phone,
                "InternetService": args.internet,
                "Contract": args.contract,
                "PaperlessBilling": args.paperless,
                "PaymentMethod": args.payment,
                "MonthlyCharges": args.monthly,
                "TotalCharges": args.total,
            }
        ]
    )

    churn_probability = model.predict_proba(customer)[0, 1]
    churn_prediction = "Yes" if churn_probability >= 0.5 else "No"

    print(f"Predicted churn: {churn_prediction}")
    print(f"Churn probability: {churn_probability:.2%}")


if __name__ == "__main__":
    main()

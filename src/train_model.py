"""Train classification models for customer churn prediction."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "customer_churn_sample.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.txt"
FEATURE_IMPORTANCE_PATH = PROJECT_ROOT / "reports" / "feature_importance.csv"
TARGET = "Churn"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    return df


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    categorical_features = features.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric_features = [column for column in features.columns if column not in categorical_features]

    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )


def compare_models(preprocessor: ColumnTransformer, x: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=6,
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced",
        ),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores: dict[str, float] = {}

    for name, model in models.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        cv_scores = cross_val_score(pipeline, x, y, cv=cv, scoring="roc_auc")
        scores[name] = float(cv_scores.mean())

    return scores


def train() -> None:
    df = load_data()
    x = df.drop(columns=[TARGET])
    y = df[TARGET].map({"No": 0, "Yes": 1})

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    preprocessor = build_preprocessor(x_train)
    model_scores = compare_models(preprocessor, x_train, y_train)

    final_model = RandomForestClassifier(
        n_estimators=250,
        max_depth=6,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced",
    )
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", final_model)])
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)
    report = classification_report(y_test, predictions, target_names=["No Churn", "Churn"])
    matrix = confusion_matrix(y_test, predictions)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = pipeline.named_steps["model"].feature_importances_
    feature_importance = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    metrics_text = [
        "Customer Churn Prediction - Model Report",
        "========================================",
        "",
        "Cross-validation ROC AUC:",
        *(f"- {name}: {score:.3f}" for name, score in model_scores.items()),
        "",
        f"Final model: Random Forest",
        f"Test accuracy: {accuracy:.3f}",
        f"Test ROC AUC: {roc_auc:.3f}",
        "",
        "Confusion matrix [[TN, FP], [FN, TP]]:",
        str(matrix),
        "",
        "Classification report:",
        report,
        "Top churn factors:",
        feature_importance.head(10).to_string(index=False),
    ]
    METRICS_PATH.write_text("\n".join(metrics_text), encoding="utf-8")

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Feature importance saved to: {FEATURE_IMPORTANCE_PATH}")
    print(f"Test ROC AUC: {roc_auc:.3f}")


if __name__ == "__main__":
    train()



import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# ── Paths ───────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved_model")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "iris_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")


def load_data():
    
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="species")
    target_names = list(iris.target_names)
    feature_names = list(iris.feature_names)
    return X, y, target_names, feature_names


def train_and_evaluate():
    """Train several classifiers, pick the best, and persist it."""
    X, y, target_names, feature_names = load_data()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define candidate models
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42
        ),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(
            max_iter=200, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=42
        ),
    }

    results = {}
    best_accuracy = 0
    best_model_name = None
    best_model = None

    print("=" * 60)
    print("  IRIS FLOWER CLASSIFICATION - MODEL COMPARISON")
    print("=" * 60)

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)

        cm = confusion_matrix(y_test, y_pred).tolist()
        report = classification_report(
            y_test, y_pred, target_names=target_names, output_dict=True
        )

        results[name] = {
            "accuracy": round(accuracy * 100, 2),
            "cv_mean": round(cv_scores.mean() * 100, 2),
            "cv_std": round(cv_scores.std() * 100, 2),
            "confusion_matrix": cm,
            "classification_report": report,
        }

        print(f"\n{'-' * 40}")
        print(f"  Model: {name}")
        print(f"  Test Accuracy:  {accuracy * 100:.2f}%")
        print(f"  CV Accuracy:    {cv_scores.mean() * 100:.2f}% "
              f"(+/- {cv_scores.std() * 100:.2f}%)")
        print(f"{'-' * 40}")
        print(classification_report(y_test, y_pred, target_names=target_names))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name
            best_model = model

    # Persist the winner
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    metrics = {
        "best_model": best_model_name,
        "best_accuracy": round(best_accuracy * 100, 2),
        "all_results": results,
        "feature_names": feature_names,
        "target_names": target_names,
        "dataset_info": {
            "total_samples": len(X),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "num_features": len(feature_names),
            "num_classes": len(target_names),
        },
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 60)
    print(f"  [BEST] Best model: {best_model_name} ({best_accuracy * 100:.2f}%)")
    print(f"  [SAVE] Saved to:   {MODEL_PATH}")
    print("=" * 60)

    return metrics


def predict(features: list[float]):
    """Load the saved model and predict the species for given features."""
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)

    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0]

    probabilities = {}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features_scaled)[0]
        probabilities = {
            name: round(float(p) * 100, 2)
            for name, p in zip(metrics["target_names"], probs)
        }

    return {
        "predicted_class": int(prediction),
        "predicted_species": metrics["target_names"][prediction],
        "probabilities": probabilities,
        "model_used": metrics["best_model"],
    }


# ── Run if executed directly ────────────────────────────────────────
if __name__ == "__main__":
    train_and_evaluate()

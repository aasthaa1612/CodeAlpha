"""
Iris Flower Classification – Flask REST API
============================================
Endpoints:
  POST /api/predict        → Predict species from flower measurements
  GET  /api/metrics        → Return model evaluation metrics
  GET  /api/dataset        → Return the full Iris dataset (for charts)
  GET  /api/dataset/stats  → Summary statistics of the dataset
"""

import os
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

from model import train_and_evaluate, predict, MODEL_PATH, METRICS_PATH, load_data

# Point Flask to the frontend folder for serving HTML/CSS/JS
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)  # Allow requests from the frontend


# ── Ensure the model is trained on startup ──────────────────────────
def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("\n[*] No saved model found -- training now...")
        train_and_evaluate()
    else:
        print("\n[OK] Saved model found -- ready to serve predictions!")


# ── Routes ──────────────────────────────────────────────────────────

@app.route("/")
def serve_frontend():
    """Serve the frontend index.html at the root URL."""
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Predict the Iris species from four numeric measurements.

    JSON body:
      {
        "sepal_length": float,
        "sepal_width": float,
        "petal_length": float,
        "petal_width": float
      }
    """
    data = request.get_json()

    required = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    missing = [k for k in required if k not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    try:
        features = [float(data[k]) for k in required]
    except (ValueError, TypeError):
        return jsonify({"error": "All measurements must be numeric."}), 400

    result = predict(features)
    return jsonify(result)


@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    """Return full model evaluation metrics."""
    if not os.path.exists(METRICS_PATH):
        return jsonify({"error": "Model not trained yet."}), 404

    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)
    return jsonify(metrics)


@app.route("/api/dataset", methods=["GET"])
def api_dataset():
    """Return the complete Iris dataset as JSON (for frontend charts)."""
    iris = load_iris()
    records = []
    for i in range(len(iris.data)):
        row = {
            "sepal_length": round(float(iris.data[i][0]), 1),
            "sepal_width": round(float(iris.data[i][1]), 1),
            "petal_length": round(float(iris.data[i][2]), 1),
            "petal_width": round(float(iris.data[i][3]), 1),
            "species": iris.target_names[iris.target[i]],
            "species_id": int(iris.target[i]),
        }
        records.append(row)
    return jsonify(records)


@app.route("/api/dataset/stats", methods=["GET"])
def api_dataset_stats():
    """Return summary statistics per species."""
    X, y, target_names, feature_names = load_data()
    X["species"] = y.map({i: n for i, n in enumerate(target_names)})

    stats = {}
    for species in target_names:
        subset = X[X["species"] == species].drop(columns=["species"])
        stats[species] = {
            "mean": subset.mean().round(2).to_dict(),
            "std": subset.std().round(2).to_dict(),
            "min": subset.min().round(2).to_dict(),
            "max": subset.max().round(2).to_dict(),
            "count": int(len(subset)),
        }

    return jsonify(stats)


@app.route("/api/retrain", methods=["POST"])
def api_retrain():
    """Force-retrain the model and return updated metrics."""
    metrics = train_and_evaluate()
    return jsonify(metrics)


# ── Entrypoint ──────────────────────────────────────────────────────
if __name__ == "__main__":
    ensure_model()
    app.run(debug=True, port=5000)

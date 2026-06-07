# 🌸 Iris Flower Classification

A full-stack machine learning project that classifies Iris flower species using petal and sepal measurements. Built with **Scikit-learn**, **Flask**, and a modern **HTML/CSS/JS** frontend.

## 📋 Task

- Use measurements of **Iris flowers** (setosa, versicolor, virginica) as input data.
- Train a **machine learning model** to classify the species based on measurements.
- Use **Scikit-learn** for dataset access and model building.
- Evaluate the model's **accuracy and performance** using test data.
- Understand **basic classification concepts** in machine learning.

## 🗂️ Project Structure

```
iris_classification/
├── backend/
│   ├── app.py              # Flask REST API server
│   ├── model.py            # ML training, evaluation & prediction
│   └── saved_model/        # Auto-generated model files
│       ├── iris_model.pkl
│       ├── scaler.pkl
│       └── metrics.json
├── frontend/
│   ├── index.html           # Main web page
│   ├── style.css            # Design system & styles
│   └── script.js            # API integration & interactivity
├── requirements.txt         # Python dependencies
└── README.md
```

## 🚀 Getting Started

### 1. Install Python Dependencies

```bash
cd iris_classification
pip install -r requirements.txt
```

### 2. Train the Model & Start the Backend

```bash
cd backend
python app.py
```

This will:
- Automatically train 5 ML models (Random Forest, SVM, KNN, Logistic Regression, Gradient Boosting)
- Select the best model based on accuracy
- Start the Flask API on `http://127.0.0.1:5000`

### 3. Open the Frontend

Open `frontend/index.html` in your browser (or use Live Server in VS Code).

## 🔌 API Endpoints

| Method | Endpoint            | Description                        |
|--------|--------------------|------------------------------------|
| POST   | `/api/predict`     | Predict species from measurements  |
| GET    | `/api/metrics`     | Get model evaluation metrics       |
| GET    | `/api/dataset`     | Get the full Iris dataset (JSON)   |
| GET    | `/api/dataset/stats` | Get per-species summary stats    |
| POST   | `/api/retrain`     | Force retrain the model            |

### Example Prediction Request

```json
POST /api/predict
{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}
```

### Response

```json
{
    "predicted_class": 0,
    "predicted_species": "setosa",
    "probabilities": {
        "setosa": 97.0,
        "versicolor": 2.0,
        "virginica": 1.0
    },
    "model_used": "Random Forest"
}
```

## 🤖 Models Trained

1. **Random Forest** – Ensemble of decision trees
2. **SVM (RBF)** – Support Vector Machine with radial basis function kernel
3. **K-Nearest Neighbors** – Distance-based classification
4. **Logistic Regression** – Linear classification model
5. **Gradient Boosting** – Sequential ensemble learning

## 📊 Evaluation Metrics

- **Accuracy** – Overall prediction correctness
- **Precision** – Per-class prediction quality
- **Recall** – Per-class detection rate
- **F1-Score** – Harmonic mean of precision and recall
- **Cross-Validation** – 5-fold CV for robust evaluation
- **Confusion Matrix** – Detailed error analysis

## ⚙️ Tech Stack

- **Backend:** Python 3, Flask, Scikit-learn, Pandas, NumPy
- **Frontend:** HTML5, CSS3, JavaScript (ES6+)
- **ML:** StandardScaler, Cross-Validation, Multiple Classifiers

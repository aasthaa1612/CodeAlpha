# 🚗 Car Price Prediction

A machine learning web app that predicts the **selling price of a used car** based on real-world features like brand, year, mileage, fuel type, and transmission.

Built as part of the **CodeAlpha ML Internship**.

---

## 🖥️ Demo

| Predict Tab | Explore Data |
|---|---|
| Enter car details → get an instant price estimate | Browse the full dataset with fuel-type filters |

---

## 🧠 How It Works

1. A synthetic dataset of **600 cars** across 15 brands is generated with realistic pricing logic
2. **5 regression models** are trained and compared:
   - Linear Regression
   - Ridge Regression
   - Random Forest Regressor
   - Gradient Boosting Regressor
   - SVR (RBF kernel)
3. The best model (by R² score) is saved and used for predictions
4. A Flask API serves both the ML backend and the frontend

---

## 📁 Project Structure

```
car_price_prediction/
├── backend/
│   ├── app.py          # Flask REST API
│   ├── model.py        # Training, evaluation & prediction logic
│   └── saved_model/    # Trained model, scaler & encoders (auto-generated)
├── data/
│   └── car_data.csv    # Auto-generated dataset
├── frontend/
│   ├── index.html      # UI structure
│   ├── style.css       # Styling
│   └── script.js       # API calls & dynamic rendering
└── .gitignore
```

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install flask flask-cors scikit-learn pandas numpy joblib
```

### 2. Run the app

```bash
cd backend
python app.py
```

### 3. Open in browser

```
http://127.0.0.1:5001
```

The model trains automatically on first run if no saved model is found.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/predict` | Predict selling price |
| `GET` | `/api/metrics` | Get model evaluation metrics |
| `GET` | `/api/dataset` | Get full dataset |
| `GET` | `/api/dataset/stats` | Get per-brand summary stats |
| `POST` | `/api/retrain` | Force retrain the model |

### Predict request body

```json
{
  "brand": "Maruti",
  "year": 2019,
  "present_price": 8.5,
  "kms_driven": 35000,
  "fuel_type": "Petrol",
  "seller_type": "Dealer",
  "transmission": "Manual",
  "owner": 0
}
```

---

## 🛠️ Tech Stack

- **Backend:** Python, Flask, Scikit-learn, Pandas, NumPy
- **Frontend:** HTML, CSS, JavaScript
- **ML Models:** Random Forest, Gradient Boosting, SVR, Linear & Ridge Regression

---

## 📊 Features Used

| Feature | Type |
|---|---|
| Brand | Categorical (15 brands) |
| Year | Numeric |
| Present Price (₹ Lakh) | Numeric |
| Kms Driven | Numeric |
| Fuel Type | Categorical (Petrol/Diesel/CNG) |
| Seller Type | Categorical (Dealer/Individual) |
| Transmission | Categorical (Manual/Automatic) |
| Previous Owners | Numeric (0–3) |

---

## 👩‍💻 Author

**Astha Baheti** — CodeAlpha ML Internship

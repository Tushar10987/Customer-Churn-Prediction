import os
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List
import uvicorn

# ----------------------------
# Paths
# ----------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "churn_model.pkl")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "api")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# Load Model
# ----------------------------
try:
    model = joblib.load(MODEL_PATH)
    FEATURES = model.feature_names_in_
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI(
    title="Churn Prediction API",
    description="API for predicting customer churn (single & batch)",
    version="1.0.0"
)

# ----------------------------
# Pydantic Model for single prediction
# ----------------------------
class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# ----------------------------
# Helper functions
# ----------------------------
def preprocess_input(df: pd.DataFrame):
    """Ensure correct feature order and missing columns"""
    missing_cols = [c for c in FEATURES if c not in df.columns]
    for c in missing_cols:
        df[c] = 0  # Add missing columns with default 0
    df = df[FEATURES]  # Align order
    return df

def predict(df: pd.DataFrame):
    df = preprocess_input(df)
    proba = model.predict_proba(df)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return pred.tolist(), proba.tolist()

# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def root():
    return {"message": "Churn Prediction API is running!"}

@app.post("/predict_single")
def predict_single(customer: Customer):
    df = pd.DataFrame([customer.dict()])
    pred, proba = predict(df)
    return {"prediction": pred[0], "churn_probability": proba[0]}

@app.post("/predict_batch")
def predict_batch(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    
    try:
        df = pd.read_csv(file.file)
        pred, proba = predict(df)
        df["churn_pred"] = pred
        df["churn_proba"] = proba
        output_file = os.path.join(OUTPUT_DIR, f"predictions.csv")
        df.to_csv(output_file, index=False)
        return {"message": f"Predictions saved to {output_file}", "predictions_file": output_file}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")

@app.get("/model_metadata")
def model_metadata():
    return {"features": FEATURES.tolist(), "model_type": type(model).__name__}

# ----------------------------
# Run API
# ----------------------------
if __name__ == "__main__":
    uvicorn.run("deploy_api:app", host="0.0.0.0", port=8000, reload=True)

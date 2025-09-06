import os
import sys
import pandas as pd
import joblib
import json

MODEL_PATH = os.path.join("models", "churn_model.pkl")
META_PATH = os.path.join("models", "model_metadata.json")
OUTPUT_PATH = os.path.join("outputs", "predictions.csv")

def load_model():
    model = joblib.load(MODEL_PATH)
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    feature_names = meta["features"]
    return model, feature_names

def prepare_input(customer_dict, feature_names):
    """Create a DataFrame with all training features filled."""
    df = pd.DataFrame([customer_dict])
    # Add missing columns with 0
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    # Keep only the columns in the right order
    df = df[feature_names]
    return df

def predict_single(customer_dict):
    model, feature_names = load_model()
    X = prepare_input(customer_dict, feature_names)
    proba = model.predict_proba(X)[0,1]
    pred = int(proba >= 0.5)
    return pred, proba

def predict_batch(input_csv):
    model, feature_names = load_model()
    df = pd.read_csv(input_csv)
    df_prepared = pd.DataFrame(columns=feature_names)
    # Fill df_prepared with zeros first
    df_prepared = df_prepared.append(pd.Series(dtype=float), ignore_index=True)
    for col in df.columns:
        if col in feature_names:
            df_prepared[col] = df[col]
    # Fill remaining missing columns with 0
    df_prepared = df_prepared.fillna(0)
    df["churn_proba"] = model.predict_proba(df_prepared)[:,1]
    df["churn_pred"] = (df["churn_proba"] >= 0.5).astype(int)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Predictions saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        predict_batch(input_file)
    else:
        sample_customer = {
            "avg_monthly_charge": 70.35,
            "gender_Male": 0,
            "Partner_Yes": 1,
            "Dependents_Yes": 0,
            "PhoneService_Yes": 1,
            "MultipleLines_No phone service": 0,
            "MultipleLines_Yes": 0,
            "InternetService_Fiber optic": 1,
            "InternetService_No": 0,
            "OnlineSecurity_No internet service": 0,
            "OnlineSecurity_Yes": 0,
            "OnlineBackup_No internet service": 0,
            "OnlineBackup_Yes": 1,
            "DeviceProtection_No internet service": 0,
            "DeviceProtection_Yes": 0,
            "TechSupport_No internet service": 0,
            "TechSupport_Yes": 0,
            "StreamingTV_No internet service": 0,
            "StreamingTV_Yes": 1,
            "StreamingMovies_No internet service": 0,
            "StreamingMovies_Yes": 0,
            "Contract_One year": 0,
            "Contract_Two year": 0,
            "PaperlessBilling_Yes": 1,
            "PaymentMethod_Credit card (automatic)": 0,
            "PaymentMethod_Electronic check": 1,
            "PaymentMethod_Mailed check": 0,
            "tenure_group_1-6": 1,
            "tenure_group_7-12": 0,
            "tenure_group_13-24": 0,
            "tenure_group_25-48": 0,
            "tenure_group_49-72": 0,
            "tenure_group_72+": 0
        }
        pred, proba = predict_single(sample_customer)
        print(f"Prediction: {pred} (Churn Probability: {proba:.2f})")

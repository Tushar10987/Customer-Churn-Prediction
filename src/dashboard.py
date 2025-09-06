import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

# -------------------
# Paths
# -------------------
MODEL_PATH = os.path.join("models", "churn_model.pkl")
OUTPUT_PATH = os.path.join("outputs", "predictions.csv")
METRICS_DIR = os.path.join("outputs", "model")

# -------------------
# Load model
# -------------------
model = joblib.load(MODEL_PATH)
features = list(model.feature_names_in_)  # Ensure correct feature order

# -------------------
# Streamlit app
# -------------------
st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")
st.title("💡 Customer Churn Prediction Dashboard")

# Sidebar mode selection
mode = st.sidebar.selectbox("Select Mode", ["Single Prediction", "Batch Prediction", "Model Metrics"])

# -------------------
# SINGLE PREDICTION
# -------------------
if mode == "Single Prediction":
    st.header("Predict Churn for a Single Customer")
    
    input_data = {}
    for f in features:
        if "charge" in f.lower() or "tenure" in f.lower() or "SeniorCitizen" in f:
            input_data[f] = st.number_input(f, value=0)
        else:
            input_data[f] = st.selectbox(f, ["Yes", "No", "Female", "Male", "Month-to-month", 
                                             "One year", "Two year", "Fiber optic", "DSL", 
                                             "No internet service", "Electronic check", 
                                             "Mailed check", "Credit card (automatic)"])
    
    if st.button("Predict"):
        df = pd.DataFrame([input_data])
        # Align features with model
        missing_cols = set(features) - set(df.columns)
        for col in missing_cols:
            df[col] = 0
        df = df[features]
        proba = model.predict_proba(df)[:,1][0]
        pred = int(proba >= 0.5)
        st.success(f"Prediction: {pred} (Churn Probability: {proba:.2f})")

# -------------------
# BATCH PREDICTION
# -------------------
elif mode == "Batch Prediction":
    st.header("Predict Churn for Multiple Customers")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        # Ensure all features are present
        missing_cols = set(features) - set(df.columns)
        for col in missing_cols:
            df[col] = 0
        X = df[features]
        
        df["churn_proba"] = model.predict_proba(X)[:,1]
        df["churn_pred"] = (df["churn_proba"] >= 0.5).astype(int)
        
        st.write("✅ Predictions:")
        st.dataframe(df.head(20))
        
        # Summary stats
        total = len(df)
        churn_count = df["churn_pred"].sum()
        churn_rate = churn_count / total * 100
        st.subheader("Summary Stats")
        st.write(f"Total customers: {total}")
        st.write(f"Predicted churn: {churn_count} ({churn_rate:.2f}%)")
        
        # Plot churn distribution
        st.subheader("Churn Distribution")
        fig, ax = plt.subplots()
        df["churn_pred"].value_counts().plot(kind="bar", color=["green", "red"], ax=ax)
        ax.set_xticklabels(["No Churn", "Churn"], rotation=0)
        ax.set_ylabel("Count")
        st.pyplot(fig)
        
        # Save predictions
        os.makedirs("outputs", exist_ok=True)
        df.to_csv(OUTPUT_PATH, index=False)
        st.success(f"✅ Predictions saved to {OUTPUT_PATH}")

# -------------------
# MODEL METRICS
# -------------------
else:
    st.header("Model Metrics & Visualizations")
    try:
        st.subheader("ROC Curve")
        st.image(os.path.join(METRICS_DIR, "XGBoost_roc_curve.png"))
        
        st.subheader("Precision-Recall Curve")
        st.image(os.path.join(METRICS_DIR, "XGBoost_pr_curve.png"))
        
        st.subheader("Top Feature Importances")
        st.image(os.path.join(METRICS_DIR, "XGBoost_feature_importance.png"))
    except Exception as e:
        st.warning("Metrics images not found. Run evaluate_model.py first.")
        st.error(str(e))

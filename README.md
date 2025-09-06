Customer Churn Prediction

Build Python License
A production-grade prototype implementing an end-to-end pipeline for predicting customer churn. The system combines feature engineering, robust preprocessing, ensemble modeling (XGBoost & Random Forest), and deployment via a FastAPI REST API. It includes evaluation metrics, visualizations, and batch/single prediction capabilities.

Features :

Data Preprocessing: Automated cleaning, encoding, and scaling of numerical and categorical features
Feature Engineering: Transform raw data into predictive features including tenure groups and aggregated metrics.
Ensemble Modeling: XGBoost and Random Forest classifiers with class imbalance handling.
Model Evaluation: Computes Accuracy, Precision, Recall, F1 Score, ROC AUC, and PR AUC.
Visualization: ROC curves, Precision-Recall curves, and feature importance plots.
Batch & Single Prediction: Predict churn for single customer JSON or multiple customers via CSV.
API Deployment: FastAPI REST API with endpoints for single and batch predictions.
Robustness: Exception handling, input validation, and logging for production readiness.

Tech Stack :

Machine Learning: XGBoost, Random Forest (scikit-learn)
Preprocessing: pandas, scikit-learn pipelines
Visualization: matplotlib, seaborn
API: FastAPI + Uvicorn
Serialization: joblib for model and preprocessor persistence
Environment: Python 3.9+, venv/virtual environment

Architecture Overview :

Data Ingestion: Load and validate raw CSV data.
Preprocessing: Clean missing values, encode categorical variables, scale numeric features.
Feature Engineering: Generate derived features such as tenure groups and aggregated metrics.
Model Training: Train XGBoost and Random Forest; handle class imbalance via scale_pos_weight.
Evaluation: Generate metrics, ROC/PR curves, and feature importance; track the best model.
Model Persistence: Serialize model (churn_model.pkl) and preprocessor (preprocessor.pkl) to models/.
Prediction Pipeline: Single or batch prediction with preprocessing applied automatically.
API Deployment: Expose REST endpoints for real-time predictions.
Output Storage: Save batch predictions to outputs/predictions.csv and evaluation artifacts in outputs/model/.

Quick Start
==== Prerequisites ====
Python 3.9+
8GB+ RAM (for local ML models)
Docker (optional for containerized deployment)

BUILD AND RUN :
# Clone repository
git clone https://github.com/Tushar10987/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction

# Create virtual environment
python -m venv churn_py
# Windows
churn_py\Scripts\activate
# macOS/Linux
source churn_py/bin/activate

# Install dependencies
pip install -r requirements.txt

TRAIN MODEL:
python src/train_model.py

This will:
Train XGBoost and Random Forest models.
Save the best model (churn_model.pkl) and preprocessor (preprocessor.pkl) in models/.
Generate evaluation metrics, plots, and feature importance in outputs/model/.

Make Predictions

Single customer prediction:
python src/predict.py
Batch predictions from CSV:
python src/predict.py data/processed/sample_customers.csv

Run API :
python src/deploy_api.py

The API will start at http://localhost:8000. Available endpoints:
POST /predict_single – Accepts a JSON payload for a single customer.
POST /predict_batch – Accepts a CSV file upload for batch predictions.

Customer-Churn-Prediction/
├── src/
│   ├── data_gathering.py
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── eda.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── predict.py
│   └── deploy_api.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── churn_model.pkl
│   └── preprocessor.pkl
├── outputs/
│   ├── model/
│   └── predictions.csv
├── requirements.txt
├── README.md
└── .gitignore

Evaluation Metrics :

Accuracy, Precision, Recall, F1 Score
ROC AUC, PR AUC
Feature Importance
Prediction Confidence (Churn Probability)

Observability & Logging :

API logs requests, predictions, and processing times.
Output directories store evaluation plots, metrics, and batch predictions.
Exceptions and invalid inputs are handled gracefully.

import os
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

def safe_mkdir(p):
    """Create directory if it does not exist"""
    os.makedirs(p, exist_ok=True)

def read_label_series(path):
    """
    Read a CSV that contains a single column of labels.
    Handles both a plain Series (no header) and a one-column DataFrame.
    """
    s = pd.read_csv(path)
    if isinstance(s, pd.DataFrame):
        if s.shape[1] >= 1:
            return s.iloc[:, 0]
        else:
            raise ValueError(f"No columns found in label file: {path}")
    return s

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    proc_dir = os.path.join(project_root, "data", "processed")
    models_dir = os.path.join(project_root, "models")
    out_dir = os.path.join(project_root, "outputs", "model")
    safe_mkdir(models_dir)
    safe_mkdir(out_dir)

    # -------------------
    # Load train/test data
    # -------------------
    X_train_path = os.path.join(proc_dir, "X_train.csv")
    X_test_path = os.path.join(proc_dir, "X_test.csv")
    y_train_path = os.path.join(proc_dir, "y_train.csv")
    y_test_path = os.path.join(proc_dir, "y_test.csv")

    for p in (X_train_path, X_test_path, y_train_path, y_test_path):
        if not os.path.exists(p):
            print("❌ ERROR: required file missing:", p)
            return

    X_train = pd.read_csv(X_train_path).reset_index(drop=True)
    X_test = pd.read_csv(X_test_path).reset_index(drop=True)
    y_train = read_label_series(y_train_path).astype(int).reset_index(drop=True)
    y_test = read_label_series(y_test_path).astype(int).reset_index(drop=True)

    print("Shapes -> X_train:", X_train.shape, "X_test:", X_test.shape,
          "y_train:", y_train.shape, "y_test:", y_test.shape)

    # -------------------
    # Handle class imbalance (for XGBoost)
    # -------------------
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / max(1, pos)
    print(f"Train class counts -> neg: {neg}, pos: {pos}")

    # -------------------
    # Define models
    # -------------------
    models = {
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
    }

    # -------------------
    # Train & Evaluate both
    # -------------------
    best_model, best_name, best_f1 = None, None, -1

    for name, model in models.items():
        print(f"\n🔹 Training {name}...")
        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        # Metrics
        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        cls_report = classification_report(y_test, y_pred, digits=4, output_dict=True)
        f1 = cls_report["1"]["f1-score"]

        print(f"{name} ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}, F1: {f1:.4f}")

        # Save plots
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f'{name} ROC AUC={roc_auc:.4f}')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{name} ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{name}_roc_curve.png"))
        plt.close()

        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        plt.figure()
        plt.plot(recall, precision, label=f'{name} AP={pr_auc:.4f}')
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"{name} Precision-Recall Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{name}_pr_curve.png"))
        plt.close()

        # Save feature importances if available
        if hasattr(model, "feature_importances_"):
            fi = model.feature_importances_
            fi_df = pd.DataFrame({"feature": X_train.columns, "importance": fi})
            fi_df = fi_df.sort_values("importance", ascending=False).head(30)
            fi_df.to_csv(os.path.join(out_dir, f"{name}_feature_importance.csv"), index=False)

        # Track best model
        if f1 > best_f1:
            best_f1, best_model, best_name = f1, model, name

    # -------------------
    # Save best model
    # -------------------
    model_path = os.path.join(models_dir, "churn_model.pkl")
    joblib.dump(best_model, model_path)

    meta = {
        "best_model": best_name,
        "f1_score": best_f1,
        "features": list(X_train.columns),
        "train_shape": X_train.shape,
        "test_shape": X_test.shape
    }
    with open(os.path.join(models_dir, "model_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ Best Model: {best_name} (F1={best_f1:.4f})")
    print("✅ Saved model to:", model_path)
    print("✅ Saved metrics and plots to:", out_dir)


if __name__ == "__main__":
    main()

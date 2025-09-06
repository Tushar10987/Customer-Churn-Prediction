import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix
)


def safe_mkdir(p):
    os.makedirs(p, exist_ok=True)


def read_label_series(path):
    """Read label CSV robustly (single column)."""
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
    out_dir = os.path.join(project_root, "outputs", "evaluation")
    safe_mkdir(out_dir)

    # -------------------
    # Load model & metadata
    # -------------------
    model_path = os.path.join(models_dir, "churn_model.pkl")
    meta_path = os.path.join(models_dir, "model_metadata.json")

    if not os.path.exists(model_path):
        print("❌ ERROR: Trained model not found at", model_path)
        return

    model = joblib.load(model_path)

    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            metadata = json.load(f)
        print("Loaded model metadata:", metadata)
    else:
        metadata = {}

    # -------------------
    # Load test data
    # -------------------
    X_test_path = os.path.join(proc_dir, "X_test.csv")
    y_test_path = os.path.join(proc_dir, "y_test.csv")

    if not (os.path.exists(X_test_path) and os.path.exists(y_test_path)):
        print("❌ ERROR: Test data files missing")
        return

    X_test = pd.read_csv(X_test_path)
    y_test = read_label_series(y_test_path).astype(int)

    # -------------------
    # Predictions
    # -------------------
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.predict(X_test)
    y_pred = (y_proba >= 0.5).astype(int)

    # -------------------
    # Metrics
    # -------------------
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    print("\n📊 Model Evaluation Results:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {roc_auc:.4f}")
    print(f"PR AUC   : {pr_auc:.4f}\n")

    print("Classification Report:\n")
    print(classification_report(y_test, y_pred, digits=4))

    # -------------------
    # Confusion Matrix
    # -------------------
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["No Churn", "Churn"])
    plt.yticks(tick_marks, ["No Churn", "Churn"])
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="red")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"))
    plt.close()

    # -------------------
    # Save metrics
    # -------------------
    results = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm.tolist(),
        "model_used": metadata.get("best_model", "Unknown")
    }
    with open(os.path.join(out_dir, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Evaluation complete. Results saved in {out_dir}")


if __name__ == "__main__":
    main()

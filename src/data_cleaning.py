# src/data_cleaning.py
import os
import pandas as pd
import numpy as np

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw_path = os.path.join(project_root, "data", "raw", "telco_churn.csv")
    proc_dir = os.path.join(project_root, "data", "processed")
    os.makedirs(proc_dir, exist_ok=True)
    out_path = os.path.join(proc_dir, "cleaned.csv")

    try:
        df = pd.read_csv(raw_path)
    except FileNotFoundError:
        print(f"ERROR: file not found at: {raw_path}")
        return
    except Exception as e:
        print("ERROR reading CSV:", e)
        return

    # Basic housekeeping
    df.columns = [c.strip() for c in df.columns]
    print("Initial shape:", df.shape)

    # Drop exact duplicates
    df = df.drop_duplicates()
    print("After dropping duplicates:", df.shape)

    # Common cleaning: coerce TotalCharges to numeric (some rows are blank/spaces)
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip(), errors='coerce')

    # Detect churn column (case-insensitive)
    churn_col = next((c for c in df.columns if c.lower() == 'churn'), None)
    if churn_col is None:
        possible = [c for c in df.columns if 'churn' in c.lower()]
        churn_col = possible[0] if possible else None

    if churn_col:
        print("Detected churn column:", churn_col)
        # normalize values like 'Yes'/'No' (case-insensitive) -> 1/0
        df[churn_col] = df[churn_col].astype(str).str.strip().replace(
            {'Yes': '1', 'No': '0', 'yes': '1', 'no': '0', 'TRUE': '1', 'FALSE': '0'}
        )
        df[churn_col] = pd.to_numeric(df[churn_col], errors='coerce').fillna(0).astype(int)
        print("Churn value counts:\n", df[churn_col].value_counts())
    else:
        print("WARNING: no churn column found. Proceeding without target conversion.")

    # Split features by type
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    # if numeric_cols contains churn, remove it
    if churn_col in numeric_cols:
        numeric_cols.remove(churn_col)
    categorical_cols = [c for c in df.columns if c not in numeric_cols + ([churn_col] if churn_col else [])]

    # Impute numeric with median
    for c in numeric_cols:
        median = df[c].median()
        df[c] = df[c].fillna(median)

    # Impute categorical with 'Unknown' and strip whitespace
    for c in categorical_cols:
        df[c] = df[c].fillna('Unknown').astype(str).str.strip()

    # Example feature: average monthly charge = TotalCharges / tenure (guard divide by zero)
    tenure_col = next((c for c in df.columns if c.lower() in ('tenure','tenure_months')), None)
    if tenure_col and 'TotalCharges' in df.columns:
        df['avg_monthly_charge'] = df['TotalCharges'] / df[tenure_col].replace({0: np.nan})
        df['avg_monthly_charge'] = df['avg_monthly_charge'].fillna(0)

    # Save cleaned data
    df.to_csv(out_path, index=False)
    print("Saved cleaned data to:", out_path)
    print("Cleaned shape:", df.shape)
    print("Done.")

if __name__ == "__main__":
    main()

# src/feature_engineering.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def infer_churn_col(df):
    churn_col = next((c for c in df.columns if c.lower() == 'churn'), None)
    if churn_col is None:
        possible = [c for c in df.columns if 'churn' in c.lower()]
        churn_col = possible[0] if possible else None
    return churn_col

def create_tenure_group(df, tenure_col):
    # simple bins (adjust as needed)
    bins = [-1, 1, 6, 12, 24, 48, 72, 1000]
    labels = ['<1','1-6','7-12','13-24','25-48','49-72','72+']
    df['tenure_group'] = pd.cut(df[tenure_col].fillna(0).astype(int), bins=bins, labels=labels, include_lowest=True)
    return df

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    proc_dir = os.path.join(project_root, "data", "processed")
    os.makedirs(proc_dir, exist_ok=True)
    in_path = os.path.join(proc_dir, "cleaned.csv")

    if not os.path.exists(in_path):
        print("ERROR: cleaned.csv not found at:", in_path)
        return

    df = pd.read_csv(in_path)
    print("Loaded cleaned data:", df.shape)

    churn_col = infer_churn_col(df)
    if not churn_col:
        print("ERROR: No churn column found. Aborting.")
        return
    print("Using target column:", churn_col)

    # Drop obvious ID columns
    id_candidates = [c for c in df.columns if c.lower() in ('customerid', 'customer_id', 'id')]
    if id_candidates:
        print("Dropping ID columns:", id_candidates)
        df = df.drop(columns=id_candidates)

    # Ensure numeric columns are numeric
    for c in df.columns:
        if df[c].dtype == object:
            # try to coerce numeric-like columns (e.g., 'TotalCharges' issues)
            coerced = pd.to_numeric(df[c].astype(str).str.strip(), errors='coerce')
            # if many non-nulls after coercion and the original was object, convert
            if coerced.notna().sum() / max(1, len(df)) > 0.5:
                df[c] = coerced

    # Create tenure_group if possible
    tenure_col = next((c for c in df.columns if c.lower() in ('tenure','tenure_months')), None)
    if tenure_col:
        df = create_tenure_group(df, tenure_col)
        print("Created tenure_group from", tenure_col)

    # avg_monthly_charge: if not present but TotalCharges & tenure exist, create it
    if 'avg_monthly_charge' not in df.columns and 'TotalCharges' in df.columns and tenure_col:
        df['avg_monthly_charge'] = df['TotalCharges'] / df[tenure_col].replace({0: np.nan})
        df['avg_monthly_charge'] = df['avg_monthly_charge'].fillna(0)
        print("Created avg_monthly_charge")

    # Separate features and target
    y = df[churn_col].astype(int)
    X = df.drop(columns=[churn_col])

    # Identify numeric and categorical
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    print("Numeric cols:", len(numeric_cols), "Categorical cols:", len(categorical_cols))

    # Impute numeric with median
    for c in numeric_cols:
        med = X[c].median()
        X[c] = X[c].fillna(med)

    # Encoding categorical features:
    low_card_cols = [c for c in categorical_cols if X[c].nunique() <= 50]
    high_card_cols = [c for c in categorical_cols if X[c].nunique() > 50]

    print("Low-cardinality categorical columns (one-hot):", len(low_card_cols))
    print("High-cardinality categorical columns (frequency-encode):", len(high_card_cols))

    # Frequency encoding for high-cardinality
    for c in high_card_cols:
        freq = X[c].map(X[c].value_counts(normalize=True))
        X[c + "_freq"] = freq
        X.drop(columns=[c], inplace=True)

    # One-hot for low-cardinality (use get_dummies; drop_first to avoid multicollinearity)
    if low_card_cols:
        X = pd.get_dummies(X, columns=low_card_cols, drop_first=True)

    # After encoding, make sure no object dtypes remain
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = X[c].astype(str)

    # Save feature list
    feature_list_path = os.path.join(proc_dir, "feature_list.txt")
    with open(feature_list_path, "w", encoding="utf-8") as f:
        for col in X.columns:
            f.write(col + "\n")
    print("Saved feature list to:", feature_list_path)

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Save to CSV
    X_train.to_csv(os.path.join(proc_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(proc_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(proc_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(proc_dir, "y_test.csv"), index=False)

    # Also save a combined processed CSV for convenience
    processed_combined = pd.concat([X, y.rename('churn')], axis=1)
    processed_combined.to_csv(os.path.join(proc_dir, "processed_features.csv"), index=False)

    print("Saved processed features and train/test splits to", proc_dir)
    print("X_train shape:", X_train.shape, "X_test shape:", X_test.shape)
    print("Done.")

if __name__ == "__main__":
    main()

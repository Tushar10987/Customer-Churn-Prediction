# src/data_gathering.py
import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw_path = os.path.join(project_root, "data", "raw", "telco_churn.csv")
    outputs_dir = os.path.join(project_root, "outputs", "eda")
    os.makedirs(outputs_dir, exist_ok=True)

    try:
        df = pd.read_csv(raw_path)
    except FileNotFoundError:
        print(f"ERROR: file not found at: {raw_path}")
        return
    except Exception as e:
        print("ERROR reading CSV:", e)
        return

    print("Loaded file:", raw_path)
    print("Shape:", df.shape)
    print("\nColumns:")
    for c in df.columns:
        print(" -", c)
    print("\nData types:")
    print(df.dtypes)

    # Missing values summary
    miss = df.isna().sum().sort_values(ascending=False)
    print("\nMissing values (top):")
    print(miss[miss > 0].head(20))

    # Try to find the churn column (case-insensitive)
    churn_col = None
    for candidate in ("churn", "Churn", "CHURN"):
        if candidate in df.columns:
            churn_col = candidate
            break

    if churn_col is None:
        # try fuzzy match by name
        possible = [c for c in df.columns if "churn" in c.lower()]
        if possible:
            churn_col = possible[0]
            print(f"\nInferred churn column as: {churn_col}")
        else:
            print("\nWARNING: No churn column found. Please check the column names.")
    else:
        print(f"\nDetected churn column: {churn_col}")

    if churn_col:
        print("\nChurn value counts:")
        print(df[churn_col].value_counts(dropna=False))

        # Save a small plot of churn distribution
        try:
            plt.figure(figsize=(6,4))
            df[churn_col].value_counts().plot(kind='bar')
            plt.title("Churn distribution")
            plt.ylabel("Count")
            plt.tight_layout()
            plot_path = os.path.join(outputs_dir, "churn_dist.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved churn distribution plot to: {plot_path}")
        except Exception as e:
            print("Could not create plot:", e)

    # Save a header sample for inspection
    head_path = os.path.join(outputs_dir, "head.csv")
    df.head(50).to_csv(head_path, index=False)
    print(f"Saved sample rows to: {head_path}")

    print("\nDone. If everything looks good, reply 'done' and we'll proceed to data cleaning.")

if __name__ == "__main__":
    main()

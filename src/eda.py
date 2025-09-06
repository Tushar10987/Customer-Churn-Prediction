# src/eda.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    proc_path = os.path.join(project_root, "data", "processed", "cleaned.csv")
    out_dir = os.path.join(project_root, "outputs", "eda")
    safe_mkdir(out_dir)

    try:
        df = pd.read_csv(proc_path)
    except FileNotFoundError:
        print(f"ERROR: cleaned dataset not found at: {proc_path}")
        return
    except Exception as e:
        print("ERROR reading cleaned CSV:", e)
        return

    print("Loaded cleaned data:", proc_path)
    print("Shape:", df.shape)

    # infer churn column
    churn_col = next((c for c in df.columns if c.lower() == 'churn'), None)
    if churn_col is None:
        possible = [c for c in df.columns if 'churn' in c.lower()]
        churn_col = possible[0] if possible else None

    if not churn_col:
        print("WARNING: No churn column found. EDA will continue without target-specific plots.")
    else:
        print("Using churn column:", churn_col)
        print(df[churn_col].value_counts())

    # 1. Summary stats
    try:
        summary_num = df.select_dtypes(include=[np.number]).describe().T
        summary_cat = df.select_dtypes(include=['object']).describe().T
        summary_num.to_csv(os.path.join(out_dir, "summary_numeric.csv"))
        summary_cat.to_csv(os.path.join(out_dir, "summary_categorical.csv"))
        print("Saved summary stats (numeric + categorical) to outputs/eda/")
    except Exception as e:
        print("Could not save summary stats:", e)

    # 2. Numeric distributions (KDE) per churn
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if churn_col and churn_col in num_cols:
            num_cols = [c for c in num_cols if c != churn_col]

        for col in num_cols:
            try:
                plt.figure(figsize=(8,4))
                if churn_col:
                    sns.kdeplot(df[df[churn_col]==0][col].dropna(), label='No churn', fill=True)
                    sns.kdeplot(df[df[churn_col]==1][col].dropna(), label='Churn', fill=True)
                    plt.title(f'Distribution of {col} by churn')
                else:
                    sns.histplot(df[col].dropna(), kde=True)
                    plt.title(f'Distribution of {col}')
                plt.xlabel(col)
                plt.legend()
                plt.tight_layout()
                fname = os.path.join(out_dir, f"dist_{col}.png")
                plt.savefig(fname)
                plt.close()
            except Exception as e:
                print("Could not plot numeric column", col, ":", e)
        print("Saved numeric distribution plots to outputs/eda/")
    except Exception as e:
        print("Numeric distribution step failed:", e)

    # 3. Categorical churn rates (for columns with reasonable cardinality)
    try:
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        for col in cat_cols:
            try:
                top_levels = df[col].value_counts().nlargest(15).index.tolist()
                tmp = df[df[col].isin(top_levels)].copy()
                if churn_col:
                    ct = pd.crosstab(tmp[col], tmp[churn_col], normalize='index') * 100
                    ax = ct.plot(kind='bar', stacked=True, figsize=(10,5))
                    plt.ylabel('Percent')
                    plt.title(f'Churn % by {col} (top levels)')
                    plt.tight_layout()
                    fname = os.path.join(out_dir, f"churn_by_{col}.png")
                    plt.savefig(fname)
                    plt.close()
                else:
                    vc = tmp[col].value_counts().nlargest(15)
                    plt.figure(figsize=(10,5))
                    sns.barplot(x=vc.values, y=vc.index)
                    plt.title(f'Top categories for {col}')
                    plt.tight_layout()
                    plt.savefig(os.path.join(out_dir, f"top_{col}.png"))
                    plt.close()
            except Exception as e:
                print("Could not plot categorical column", col, ":", e)
        print("Saved categorical plots to outputs/eda/")
    except Exception as e:
        print("Categorical distribution step failed:", e)

    # 4. Correlation heatmap (numerical)
    try:
        num_df = df.select_dtypes(include=[np.number])
        corr = num_df.corr()
        plt.figure(figsize=(12,10))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0)
        plt.title("Correlation heatmap (numeric features)")
        plt.tight_layout()
        corr_path = os.path.join(out_dir, "corr_heatmap.png")
        plt.savefig(corr_path)
        plt.close()
        print("Saved correlation heatmap to:", corr_path)
    except Exception as e:
        print("Could not create correlation heatmap:", e)

    # 5. Save churn-rate-by-category CSVs (if churn exists)
    if churn_col:
        try:
            cat_cols = df.select_dtypes(include=['object']).columns.tolist()
            for col in cat_cols:
                try:
                    rates = (df.groupby(col)[churn_col].mean().sort_values(ascending=False).head(50) * 100).round(2)
                    rates.to_csv(os.path.join(out_dir, f"churn_rate_by_{col}.csv"))
                except Exception:
                    # skip columns that cause grouping issues
                    pass
            print("Saved churn-rate-by-category CSVs to outputs/eda/")
        except Exception as e:
            print("Could not save churn-rate-by-category CSVs:", e)

    print("EDA complete. Check the outputs in:", out_dir)
    print("Done.")

if __name__ == "__main__":
    main()

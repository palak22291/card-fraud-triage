import os
import pandas as pd

DATA_PATH = os.path.join("data", "raw", "creditcard.csv")

def audit_dataset():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"File not found at {DATA_PATH}. Please place creditcard.csv in data/raw/")

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    print("\n--- 1. BASIC SHAPE & COLUMNS ---")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")
    print(f"Column Names: {list(df.columns)}")

    print("\n--- 2. MISSING VALUES & DUPLICATES ---")
    missing_count = df.isnull().sum().sum()
    print(f"Total Missing Values: {missing_count}")
    
    duplicate_count = df.duplicated().sum()
    print(f"Exact Duplicate Rows: {duplicate_count:,} ({(duplicate_count / len(df)) * 100:.3f}%)")

    print("\n--- 3. CLASS DISTRIBUTION ---")
    class_counts = df['Class'].value_counts()
    legit_count = class_counts.get(0, 0)
    fraud_count = class_counts.get(1, 0)
    print(f"Legitimate (Class 0): {legit_count:,} ({legit_count / len(df) * 100:.3f}%)")
    print(f"Fraudulent (Class 1): {fraud_count:,} ({fraud_count / len(df) * 100:.3f}%)")
    print(f"Imbalance Ratio: {legit_count / fraud_count:.1f} : 1")

    print("\n--- 4. TRANSACTION AMOUNT SUMMARY ---")
    print(df['Amount'].describe())

    print("\n--- 5. TIME RANGE ---")
    min_time = df['Time'].min()
    max_time = df['Time'].max()
    print(f"Min Time: {min_time:.1f}s | Max Time: {max_time:.1f}s (Approx {max_time / 3600:.2f} hours)")

if __name__ == "__main__":
    audit_dataset()
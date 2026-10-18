import os
import pandas as pd

DATA_PATH = os.path.join("data", "raw", "creditcard.csv")
EXPECTED_ROW_COUNT = 284807
EXPECTED_COL_COUNT = 31


def audit_dataset():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"File not found at {DATA_PATH}. Please place creditcard.csv in data/raw/"
        )

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    print("\n--- 1. BASIC SHAPE & COLUMNS ---")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")
    print(f"Column Names: {list(df.columns)}")

    print("\n--- 2. MISSING VALUES & DUPLICATES ---")
    missing_count = int(df.isnull().sum().sum())
    print(f"Total Missing Values: {missing_count}")

    duplicate_count = int(df.duplicated().sum())
    print(
        f"Exact Duplicate Rows: {duplicate_count:,} "
        f"({(duplicate_count / len(df)) * 100:.3f}%)"
    )

    print("\n--- 3. CLASS DISTRIBUTION ---")
    class_counts = df["Class"].value_counts()
    legit_count = int(class_counts.get(0, 0))
    fraud_count = int(class_counts.get(1, 0))
    print(f"Legitimate (Class 0): {legit_count:,} ({legit_count / len(df) * 100:.3f}%)")
    print(f"Fraudulent (Class 1): {fraud_count:,} ({fraud_count / len(df) * 100:.3f}%)")
    print(f"Imbalance Ratio: {legit_count / fraud_count:.1f} : 1")

    print("\n--- 4. TRANSACTION AMOUNT SUMMARY ---")
    print(df["Amount"].describe())

    print("\n--- 5. TIME RANGE ---")
    min_time = float(df["Time"].min())
    max_time = float(df["Time"].max())
    is_monotonic = bool(df["Time"].is_monotonic_increasing)
    print(
        f"Min Time: {min_time:.1f}s | Max Time: {max_time:.1f}s "
        f"(Approx {max_time / 3600:.2f} hours)"
    )
    print(f"Time Monotonically Non-Decreasing: {is_monotonic}")

    print("\n--- 6. ENFORCING DATA INVARIANT ASSERTIONS ---")
    assert df.shape[1] == EXPECTED_COL_COUNT, (
        f"Expected {EXPECTED_COL_COUNT} columns, found {df.shape[1]}"
    )
    assert df.shape[0] == EXPECTED_ROW_COUNT, (
        f"Expected {EXPECTED_ROW_COUNT:,} rows, found {df.shape[0]:,}"
    )
    expected_cols = {"Time", "Amount", "Class"} | {f"V{i}" for i in range(1, 29)}
    assert set(df.columns) == expected_cols, (
        f"Unexpected schema difference: {set(df.columns) ^ expected_cols}"
    )
    assert missing_count == 0, f"Dataset contains {missing_count} missing values!"
    assert set(df["Class"].unique()) == {0, 1}, (
        f"Class column contains invalid values: {set(df['Class'].unique())}"
    )
    assert (df["Amount"] >= 0).all(), "Negative transaction amounts detected!"
    assert is_monotonic, "Time column is not monotonically non-decreasing!"

    print("All data quality assertions passed successfully!")


if __name__ == "__main__":
    audit_dataset()
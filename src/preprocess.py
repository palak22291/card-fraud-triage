import os
import pandas as pd


def process_credit_card_data():
    raw_data_path = os.path.join("data", "raw", "creditcard.csv")
    processed_dir = os.path.join("data", "processed")

    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Error: {raw_data_path} does not exist.")

    os.makedirs(processed_dir, exist_ok=True)

    print(f"Loading raw dataset from {raw_data_path}...")
    df = pd.read_csv(raw_data_path)

    # 1. Sort chronologically by Time to prevent future-to-past data leakage
    df = df.sort_values(by="Time").reset_index(drop=True)

    # 2. Compute 60 / 20 / 20 chronological split indices
    n_total = len(df)
    train_end = int(n_total * 0.60)
    val_end = int(n_total * 0.80)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    n_train_total = len(train_df)
    n_train_fraud_dropped = int((train_df["Class"] == 1).sum())

    # 3. Filter training data to legitimate transactions ONLY (Class == 0)
    #    Drop 'Class' (unsupervised) and 'Time' (recording artifact / leakage risk)
    train_legit_df = (
        train_df[train_df["Class"] == 0]
        .drop(columns=["Class", "Time"])
        .reset_index(drop=True)
    )

    # 4. Drop 'Time' from val and test sets so all splits share the exact 29 features
    val_df = val_df.drop(columns=["Time"]).reset_index(drop=True)
    test_df = test_df.drop(columns=["Time"]).reset_index(drop=True)

    # 5. Save processed splits
    train_path = os.path.join(processed_dir, "train_legit.csv")
    val_path = os.path.join(processed_dir, "val.csv")
    test_path = os.path.join(processed_dir, "test.csv")

    train_legit_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    # 6. Print summary statistics for audit verification
    val_fraud = int((val_df["Class"] == 1).sum())
    val_legit = int((val_df["Class"] == 0).sum())
    test_fraud = int((test_df["Class"] == 1).sum())
    test_legit = int((test_df["Class"] == 0).sum())

    print("\n" + "=" * 60)
    print("PREPROCESSING & CHRONOLOGICAL SPLIT SUMMARY")
    print("=" * 60)
    print(f"Total Raw Transactions      : {n_total:,}")
    print(f"Features Retained (X)       : {train_legit_df.shape[1]} (V1-V28, Amount)")
    print("-" * 60)
    print(f"Train Split (60% raw)       : {n_train_total:,} rows ({n_train_fraud_dropped:,} fraud excluded)")
    print(f" -> train_legit.csv         : {len(train_legit_df):,} legit rows | Shape: {train_legit_df.shape}")
    print(
        f"Val Split   (20% -> val.csv): {len(val_df):,} rows "
        f"({val_legit:,} legit, {val_fraud:,} fraud [{val_fraud / len(val_df) * 100:.3f}%])"
    )
    print(
        f"Test Split  (20% -> test.csv): {len(test_df):,} rows "
        f"({test_legit:,} legit, {test_fraud:,} fraud [{test_fraud / len(test_df) * 100:.3f}%])"
    )
    print("-" * 60)
    print(f"Files written to            : {os.path.abspath(processed_dir)}")
    print("=" * 60)
    print("Data processing successfully completed!")


if __name__ == "__main__":
    process_credit_card_data()
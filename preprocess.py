import os
import pandas as pd

def process_credit_card_data():
    raw_data_path = os.path.join("data", "raw", "creditcard.csv")
    processed_dir = os.path.join("data", "processed")
    
    if not os.path.exists(raw_data_path):
        print(f"Error: {raw_data_path} does not exist.")
        return

    os.makedirs(processed_dir, exist_ok=True)
    
    df = pd.read_csv(raw_data_path)
    df = df.sort_values(by="Time").reset_index(drop=True)
    
    n_total = len(df)
    train_end = int(n_total * 0.60)
    val_end = int(n_total * 0.80)
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    # Filter legitimate transactions and drop the 'Class' column for unsupervised training
    train_legit_df = train_df[train_df["Class"] == 0].drop(columns=["Class"]).reset_index(drop=True)
    
    # Save datasets with correct filenames
    train_legit_df.to_csv(os.path.join(processed_dir, "train_legit.csv"), index=False)
    val_df.to_csv(os.path.join(processed_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(processed_dir, "test.csv"), index=False)
    
    print("Data processing successfully completed!")

if __name__ == "__main__":
    process_credit_card_data()
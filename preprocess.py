import os
import pandas as pd
from sklearn.model_selection import train_test_split

def process_credit_card_data():
    raw_data_path = os.path.join("data", "raw", "creditcard.csv")
    processed_dir = os.path.join("data", "processed")
    
    if not os.path.exists(raw_data_path):
        print(f"Error: {raw_data_path} does not exist.")
        print("Please place 'creditcard.csv' inside the 'data/raw/' directory and re-run.")
        return

    os.makedirs(processed_dir, exist_ok=True)
    
    print("Loading raw dataset...")
    df = pd.read_csv(raw_data_path)
    
    print("Sorting dataset by 'Time'...")
    df = df.sort_values(by="Time").reset_index(drop=True)
    
    print("Splitting dataset into 60/20/20 train/val/test...")
    train_df, temp_df = train_test_split(df, test_size=0.40, shuffle=False)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, shuffle=False)
    
    print("Filtering training set to Class == 0...")
    train_df = train_df[train_df["Class"] == 0].reset_index(drop=True)
    
    print("Saving datasets to data/processed/...")
    train_df.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(processed_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(processed_dir, "test.csv"), index=False)
    
    print("Data processing successfully completed!")

if __name__ == "__main__":
    process_credit_card_data()
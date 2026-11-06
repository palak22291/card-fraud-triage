"""
Chronological Anti-Leakage Preprocessing Pipeline (Phase 1 ML & Phase 2 DL).
- Sorts transactions globally by Time (60% Train / 20% Val / 20% Test).
- Drops Time feature to prevent temporal covariate shift across non-overlapping windows.
- Filters training set strictly to legitimate transactions (Class == 0).
- Audits and removes exact feature duplicates on the training split (Phase-2 enhancement)
  so neither the GMM nor the Deep Autoencoder overfits density/manifold capacity to
  repeated logging artifacts, while retaining natural stream arrivals in Val/Test.
"""

from pathlib import Path
import pandas as pd

RAW_DATA_PATH = Path("data/raw/creditcard.csv")
PROCESSED_DIR = Path("data/processed")

FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COL = "Class"


def run_preprocessing() -> None:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw dataset not found at {RAW_DATA_PATH}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW_DATA_PATH)

    # 1. Global duplicate audit (before dropping Time)
    n_raw_dups = int(df.duplicated().sum())
    print(f"[Audit] Raw dataset exact duplicates (including Time): {n_raw_dups:,} / {len(df):,} ({n_raw_dups/len(df)*100:.3f}%)")

    # 2. Strict chronological sort by Time
    df_sorted = df.sort_values("Time", kind="mergesort").reset_index(drop=True)

    n_total = len(df_sorted)
    train_end = int(0.60 * n_total)
    val_end = int(0.80 * n_total)

    train_raw = df_sorted.iloc[:train_end].copy()
    val_df = df_sorted.iloc[train_end:val_end].copy()
    test_df = df_sorted.iloc[val_end:].copy()

    # 3. Filter training set strictly to legitimate transactions (Class == 0)
    train_legit_full = train_raw[train_raw[TARGET_COL] == 0][FEATURE_COLS].reset_index(drop=True)

    # 4. Deduplicated legitimate training split (Phase 2 DL enhancement)
    n_train_dups = int(train_legit_full.duplicated().sum())
    train_legit_dedup = train_legit_full.drop_duplicates().reset_index(drop=True)

    # 5. Drop Time from Val and Test (keep 29 features + Class)
    val_out = val_df[FEATURE_COLS + [TARGET_COL]].reset_index(drop=True)
    test_out = test_df[FEATURE_COLS + [TARGET_COL]].reset_index(drop=True)

    # Save artifacts
    train_legit_full.to_csv(PROCESSED_DIR / "train_legit.csv", index=False)
    train_legit_dedup.to_csv(PROCESSED_DIR / "train_legit_dedup.csv", index=False)
    val_out.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test_out.to_csv(PROCESSED_DIR / "test.csv", index=False)

    print("=" * 68)
    print("CHRONOLOGICAL PREPROCESSING SUMMARY (29 BEHAVIORAL FEATURES)")
    print("=" * 68)
    print(f"Train Legit (Full)        : {len(train_legit_full):>7,} rows -> data/processed/train_legit.csv")
    print(f"Train Duplicate Rows      : {n_train_dups:>7,} rows ({n_train_dups/len(train_legit_full)*100:.3f}% repeated feature vectors)")
    print(f"Train Legit (Deduplicated): {len(train_legit_dedup):>7,} rows -> data/processed/train_legit_dedup.csv")
    print(f"Validation Split (20%)    : {len(val_out):>7,} rows ({int(val_out[TARGET_COL].sum())} frauds) -> data/processed/val.csv")
    print(f"Held-Out Test Split (20%) : {len(test_out):>7,} rows ({int(test_out[TARGET_COL].sum())} frauds) -> data/processed/test.csv")
    print("=" * 68)


if __name__ == "__main__":
    run_preprocessing()
# src/make_splits.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
METADATA_DIR = os.path.join(BASE_DIR, "metadata")

INPUT_PATH = os.path.join(METADATA_DIR, "metadata_with_captions.parquet")
TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")

def main():
    print(f"Loading: {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    print("Total rows:", len(df))

    # First: train vs temp
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,      # 30% goes to val+test
        random_state=42,
        shuffle=True,
    )

    # Second: split temp into val and test (50/50 of 30% => 15% each)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        shuffle=True,
    )

    print("Train size:", len(train_df))
    print("Val size:  ", len(val_df))
    print("Test size: ", len(test_df))

    train_df.to_parquet(TRAIN_PATH, index=False)
    val_df.to_parquet(VAL_PATH, index=False)
    test_df.to_parquet(TEST_PATH, index=False)

    print("Saved:")
    print("  ", TRAIN_PATH)
    print("  ", VAL_PATH)
    print("  ", TEST_PATH)

if __name__ == "__main__":
    main()

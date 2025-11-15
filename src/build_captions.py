# src/build_captions.py

import os
import pandas as pd
from captions import build_caption

# Adjust if your paths differ
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
METADATA_DIR = os.path.join(BASE_DIR, "metadata")

INPUT_PATH = os.path.join(METADATA_DIR, "merged.parquet")
OUTPUT_PATH = os.path.join(METADATA_DIR, "metadata_with_captions.parquet")

def main():
    print(f"Loading metadata from: {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    print("Loaded:", df.shape)

    # Create caption column
    print("Building captions...")
    df["caption"] = df.apply(build_caption, axis=1)

    print("Sample captions:")
    for i in range(3):
        print("----")
        print("Name:   ", df.iloc[i].get("productDisplayName", ""))
        print("Caption:", df.iloc[i]["caption"])

    print(f"Saving to: {OUTPUT_PATH}")
    df.to_parquet(OUTPUT_PATH, index=False)
    print("Done.")

if __name__ == "__main__":
    main()

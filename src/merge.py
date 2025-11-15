import pandas as pd
from metadata import load_metadata
from image_ing import load_images
import os

def merge_data():
    df_meta = load_metadata()
    df_imgs = load_images()

    print("\nMerging metadata + images...")
    df = df_meta.merge(
        df_imgs,
        left_on="id",
        right_on="image_id",
        how="inner"
    )

    print("Merged shape:", df.shape)
    print(df.head())

    # Save final merged dataset
    output_path = os.path.join("..", "metadata", "merged.parquet")
    df.to_parquet(output_path, index=False)

    print("Saved merged dataset to:", output_path)

    return df


if __name__ == "__main__":
    merge_data()

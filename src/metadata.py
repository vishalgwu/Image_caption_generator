import pandas as pd
import os

def load_metadata():
    metadata_path = os.path.join("..", "metadata", "styles.csv")

    print("Loading metadata from:", metadata_path)

    # FIX: use python engine + skip bad lines
    df = pd.read_csv(
        metadata_path,
        engine="python",
        on_bad_lines="skip",
        quotechar='"'
    )

    print("Loaded metadata:", df.shape)

    # convert id to string
    df["id"] = df["id"].astype(str)

    # drop rows with missing id
    df = df.dropna(subset=["id"])

    print("Cleaned metadata:", df.shape)
    print(df.head())

    return df


if __name__ == "__main__":
    load_metadata()

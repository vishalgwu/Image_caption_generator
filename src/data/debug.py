import pandas as pd
from PIL import Image

df = pd.read_parquet("../metadata/merged.parquet")

sample = df.sample(5)

for idx, row in sample.iterrows():
    try:
        img = Image.open(row["image_path"])
        img.verify()  # check if corrupted
        print("OK:", row["image_path"])
    except:
        print("CORRUPT:", row["image_path"])

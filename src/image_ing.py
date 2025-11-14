import os
import pandas as pd

# Path to images/small directory
BASE_DIR = os.path.dirname(__file__)
IMAGE_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "images", "small"))

print("Looking in:", IMAGE_DIR)

image_rows = []

# Walk through all subfolders inside images/small/
for root, dirs, files in os.walk(IMAGE_DIR):
    for fname in files:
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            fpath = os.path.join(root, fname)

            # filename without extension → product_id
            product_id_guess = os.path.splitext(fname)[0]

            image_rows.append({
                "image_path": fpath,
                "product_id_guess": product_id_guess
            })

df_images = pd.DataFrame(image_rows)

print("Images found:", len(df_images))
print(df_images.head())

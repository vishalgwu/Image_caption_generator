import os
import pandas as pd

def load_images():
    images_folder = os.path.join("..", "images")

    print("Looking in:", images_folder)

    all_files = os.listdir(images_folder)
    images = [f for f in all_files if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    print("Images found:", len(images))

    data = []

    for img in images:
        img_id = os.path.splitext(img)[0]   # filename without extension
        img_path = os.path.join(images_folder, img)

        data.append([img_id, img_path])

    df_images = pd.DataFrame(data, columns=["image_id", "image_path"])

    print(df_images.head())
    return df_images


if __name__ == "__main__":
    load_images()

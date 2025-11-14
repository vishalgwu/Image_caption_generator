import os
import json
import pandas as pd

metadata_dir = os.path.join(os.path.dirname(__file__), "..", "listings", "metadata")
metadata_dir = os.path.abspath(metadata_dir)

print("Looking in:", metadata_dir)

all_rows = []

for fname in os.listdir(metadata_dir):
    if fname.startswith("listings_"):
        fpath = os.path.join(metadata_dir, fname)

        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                all_rows.extend(data)
            elif isinstance(data, dict):
                if "listings" in data and isinstance(data["listings"], list):
                    all_rows.extend(data["listings"])
                else:
                    all_rows.extend(list(data.values()))

        except json.JSONDecodeError:
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        all_rows.append(obj)
                    except json.JSONDecodeError:
                        pass

df_metadata = pd.DataFrame(all_rows)

print("Metadata shape:", df_metadata.shape)
print("First few columns:", df_metadata.columns.tolist()[:20])
print(df_metadata.head(3))

# src/data/dataset_vllm.py

import os
from typing import Optional, Dict, Any

import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


class VLLMDataset(Dataset):
    """
    Dataset for BLIP-2 fine-tuning using:
    - product image
    - structured metadata -> converted into a text prompt
    - ground-truth caption

    Each item returns:
      {
        "image": PIL.Image,
        "metadata_text": str,
        "caption": str,
      }

    The BLIP-2 processor/tokenization is applied in the collate_fn or training loop.
    """

    def __init__(
        self,
        parquet_path: str,
        images_dir: str,
    ) -> None:
        super().__init__()
        self.df = pd.read_parquet(parquet_path)
        self.images_dir = images_dir

        # Basic sanity columns – change if your names differ
        required_cols = ["id", "caption"]
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(
                    f"Column '{col}' not found in {parquet_path}. "
                    f"Available columns: {list(self.df.columns)}"
                )

    def __len__(self) -> int:
        return len(self.df)

    def _build_metadata_text(self, row: pd.Series) -> str:
        """
        Turn structured metadata into a natural-language string that
        the VLM can use as additional context.
        """
        parts = []

        def add(label: str, key: str):
            if key in row and pd.notna(row[key]):
                parts.append(f"{label}: {row[key]}")

        add("Gender", "gender")
        add("Category", "masterCategory")
        add("Subcategory", "subCategory")
        add("Article type", "articleType")
        add("Base colour", "baseColour")
        add("Season", "season")
        add("Usage", "usage")

        meta_text = ". ".join(parts)
        if meta_text:
            meta_text = meta_text + ". "
        # final instruction for the model
        meta_text += "Generate a concise, descriptive caption for this fashion product."

        return meta_text

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        image_id = str(row["id"])
        # adjust extension if your images are .png
        img_path = os.path.join(self.images_dir, f"{image_id}.jpg")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")
        metadata_text = self._build_metadata_text(row)
        caption = str(row["caption"])

        return {
            "image": image,
            "metadata_text": metadata_text,
            "caption": caption,
            "id": image_id,
        }

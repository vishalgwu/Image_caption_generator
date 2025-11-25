import os
from pathlib import Path
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


def get_image_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def simple_tokenize(text: str):
    return text.lower().strip().split()


class FashionDatasetV2(Dataset):
    """
    Upgraded dataset for model1:
    - returns (image, caption_ids, metadata_ids_dict)
    - metadata columns: gender, masterCategory, subCategory,
      articleType, baseColour, season, usage, year
    """

    META_COLS = [
        "gender",
        "masterCategory",
        "subCategory",
        "articleType",
        "baseColour",
        "season",
        "usage",
        "year"
    ]

    def __init__(
        self,
        parquet_path,
        vocab,
        meta_label_encoders,
        images_dir="images",
        max_len=30,
        caption_column="caption"
    ):
        self.df = pd.read_parquet(parquet_path)
        self.images_dir = Path(images_dir)
        self.max_len = max_len
        self.caption_col = caption_column
        self.transform = get_image_transform()

        # vocab
        self.vocab = vocab
        self.word2idx = vocab.word2idx

        self.pad_idx = vocab.word2idx["<pad>"]
        self.bos_idx = vocab.word2idx["<sos>"]
        self.eos_idx = vocab.word2idx["<eos>"]
        self.unk_idx = vocab.word2idx["<unk>"]

        # label encoders for metadata
        self.meta_LE = meta_label_encoders

    def __len__(self):
        return len(self.df)

    def numericalize(self, text):
        tokens = simple_tokenize(text)
        ids = [self.bos_idx]

        for tok in tokens:
            ids.append(self.word2idx.get(tok, self.unk_idx))

        ids.append(self.eos_idx)

        if len(ids) < self.max_len:
            ids += [self.pad_idx] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]

        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # -------------------------
        # Load image
        # -------------------------
        img_id = str(row["id"])
        img_path = self.images_dir / f"{img_id}.jpg"
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # -------------------------
        # Captions
        # -------------------------
        caption = self.numericalize(str(row[self.caption_col]))

        # -------------------------
        # Metadata IDs
        # -------------------------
        meta_dict = {}
        for col in self.META_COLS:
            meta_id = self.meta_LE[col].transform([str(row[col])])[0]
            meta_dict[col] = torch.tensor(meta_id, dtype=torch.long)

        return image, caption, meta_dict

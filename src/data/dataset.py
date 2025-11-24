import os
from pathlib import Path

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from src.utils import load_vocab


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
    # very basic whitespace tokenizer
    return text.lower().strip().split()


class FashionCaptionDataset(Dataset):
    def __init__(
            self,
            parquet_path,
            images_dir="images",


            max_len=30,
            caption_column="caption"  # Change this if your column name is different
    ):
        self.df = pd.read_parquet(parquet_path)
        self.images_dir = Path(images_dir)
        self.max_len = max_len
        self.caption_col = caption_column

        self.transform = get_image_transform()

        # Load vocab
        vocab = load_vocab()
        self.word2idx = vocab["word2idx"]
        self.pad_idx = vocab["pad_idx"]
        self.bos_idx = vocab["bos_idx"]
        self.eos_idx = vocab["eos_idx"]
        self.unk_idx = vocab["unk_idx"]

    def __len__(self):
        return len(self.df)

    def numericalize(self, text):
        tokens = simple_tokenize(text)

        # Start with <sos>
        ids = [self.bos_idx]

        for tok in tokens:
            ids.append(self.word2idx.get(tok, self.unk_idx))

        # End with <eos>
        ids.append(self.eos_idx)

        # pad or cut to fixed length
        if len(ids) < self.max_len:
            ids += [self.pad_idx] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]

        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # assuming your parquet contains column "id" for image filename
        img_id = str(row["id"])
        img_path = self.images_dir / f"{img_id}.jpg"

        # Load and transform image
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Caption text
        caption_text = str(row[self.caption_col])
        caption_ids = self.numericalize(caption_text)

        return {
            "image": image,  # tensor (3, 224, 224)
            "caption_ids": caption_ids,  # tensor (max_len,)
        }

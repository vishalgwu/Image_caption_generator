import os
from pathlib import Path
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from src.data.vocab import PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX

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


class FashionDataset(Dataset):
    def __init__(
            self,
            parquet_path,
            vocab,
            images_dir="images",
            max_len=30,
            caption_column="caption"
    ):
        self.df = pd.read_parquet(parquet_path)
        self.images_dir = Path(images_dir)
        self.max_len = max_len
        self.caption_col = caption_column
        self.transform = get_image_transform()

        # Use FashionVocab object
        self.vocab = vocab
        self.word2idx = vocab.word2idx
        self.pad_idx = PAD_IDX
        self.bos_idx = SOS_IDX
        self.eos_idx = EOS_IDX
        self.unk_idx = UNK_IDX

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
        img_id = str(row["id"])
        img_path = self.images_dir / f"{img_id}.jpg"

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        caption_text = str(row[self.caption_col])
        caption_ids = self.numericalize(caption_text)

        return image, caption_ids

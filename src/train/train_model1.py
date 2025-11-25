import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os

from src.models.model1 import Model1
from src.data.dataset_model1 import FashionDatasetV2
from src.data.collate_model1 import collate_fn_model1
from src.data.vocab import FashionVocab


META_COLS = [
    "gender", "masterCategory", "subCategory",
    "articleType", "baseColour", "season", "usage", "year"
]


def build_meta_encoders(df):
    LE = {}
    meta_sizes = {}
    for col in META_COLS:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        le.fit(df[col].values)
        LE[col] = le
        meta_sizes[col] = len(le.classes_)
    return LE, meta_sizes


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load vocab
    vocab = FashionVocab.load("metadata/vocab.pkl")

    # load train df to build label encoders
    df = pd.read_parquet("metadata/merged.parquet")
    LE, meta_sizes = build_meta_encoders(df)

    # dataset
    train_ds = FashionDatasetV2(
        parquet_path="metadata/train.parquet",
        vocab=vocab,
        meta_label_encoders=LE,
        images_dir="images",
        max_len=30
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn_model1
    )

    # model
    model = Model1(vocab, meta_sizes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=vocab.word2idx["<pad>"])

    # training loop
    for epoch in range(8):
        model.train()
        total_loss = 0

        for images, captions, meta_dict in train_dl:
            images = images.to(device)
            captions = captions.to(device)
            meta_dict = {k: v.to(device) for k, v in meta_dict.items()}

            opt.zero_grad()

            logits = model(images, captions[:, :-1], meta_dict)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), captions[:, 1:].reshape(-1))

            loss.backward()
            opt.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "model1_best.pth")


if __name__ == "__main__":
    main()

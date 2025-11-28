# src/train/train_baseline.py
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import nltk

# make project root importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

EARLY_STOPPING_PATIENCE = 2

# NLTK BLEU
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

from src.data.vocab import FashionVocab
from src.data.dataset import FashionDataset
from src.data.collate import collate_fn
from src.models.caption_model import CaptionModel


# -----------------------------
# CONFIG
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
LR = 3e-4
NUM_EPOCHS = 15
MAX_LEN = 30
IMAGES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "images"))

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "metadata"))
TRAIN_PATH = os.path.join(DATA_DIR, "train.parquet")
VAL_PATH = os.path.join(DATA_DIR, "val.parquet")
VOCAB_PATH = os.path.join(DATA_DIR, "vocab.pkl")

BEST_MODEL_PATH = "baseline_best.pth"


# -----------------------------
# LOAD VOCAB
# -----------------------------
print("Loading vocabulary...")
vocab = FashionVocab.load(VOCAB_PATH)
vocab_size = len(vocab)
print(f"Vocab size = {vocab_size}")


# -----------------------------
# LOAD DATASETS
# -----------------------------
print("Loading datasets...")
train_ds = FashionDataset(
    parquet_path=TRAIN_PATH,
    vocab=vocab,
    images_dir=IMAGES_DIR
)

val_ds = FashionDataset(
    parquet_path=VAL_PATH,
    vocab=vocab,
    images_dir=IMAGES_DIR
)


train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
)


# -----------------------------
# INITIALIZE MODEL
# -----------------------------
print("Initializing model...")
model = CaptionModel(vocab=vocab, d_model=512).to(DEVICE)

PAD_INDEX = vocab.pad_idx  # or vocab.word2idx["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=PAD_INDEX)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2
)



# -----------------------------
# TRAIN ONE EPOCH
# -----------------------------
def train_one_epoch():
    model.train()
    running_loss = 0.0

    for images, captions in tqdm(train_loader, desc="Training"):
        images = images.to(DEVICE)
        captions = captions.to(DEVICE)

        optimizer.zero_grad()

        # teacher forcing: input = all tokens except last
        logits = model(images, captions[:, :-1])

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            captions[:, 1:].reshape(-1),
        )

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)


# -----------------------------
# VALIDATION
# -----------------------------
def validate():
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, captions in tqdm(val_loader, desc="Validating"):
            images = images.to(DEVICE)
            captions = captions.to(DEVICE)

            logits = model(images, captions[:, :-1])
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                captions[:, 1:].reshape(-1),
            )

            running_loss += loss.item()

    return running_loss / len(val_loader)


# -----------------------------
# BLEU SCORE EVALUATION
# -----------------------------
def compute_bleu():
    model.eval()
    score_list = []

    with torch.no_grad():
        for images, captions in tqdm(val_loader, desc="Computing BLEU"):
            images = images.to(DEVICE)
            captions = captions.to(DEVICE)

            preds = model.generate(images, max_len=MAX_LEN)  # (B, T)

            for p, t in zip(preds, captions):
                pred_sentence = vocab.decode(p)
                true_sentence = vocab.decode(t)

                score = nltk.translate.bleu_score.sentence_bleu(
                    [true_sentence.split()],
                    pred_sentence.split(),
                    weights=(0.25, 0.25, 0.25, 0.25),
                )
                score_list.append(score)

    return sum(score_list) / len(score_list)


# -----------------------------
# TRAINING LOOP
# -----------------------------
print("Starting training...")
best_loss = float("inf")
no_improve_epochs = 0

for epoch in range(1, NUM_EPOCHS + 1):
    print("\n---------------------")
    print(f"Epoch {epoch}/{NUM_EPOCHS}")
    print("---------------------")

    train_loss = train_one_epoch()
    val_loss = validate()
    bleu = compute_bleu()

    scheduler.step(val_loss)

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss:   {val_loss:.4f}")
    print(f"BLEU Score: {bleu:.4f}")

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"✔ Saved new best model → {BEST_MODEL_PATH}")
        no_improve_epochs = 0
    else:
        no_improve_epochs += 1
        print(f"⚠ No improvement for {no_improve_epochs} epoch(s).")

    if no_improve_epochs >= EARLY_STOPPING_PATIENCE:
        print("Early stopping triggered.")
        break

print("\nTraining complete.")
print(f"Best validation loss: {best_loss:.4f}")



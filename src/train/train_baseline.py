import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import nltk
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# IMPORTANT: make sure NLTK BLEU is installed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Project imports
from src.models.caption_model import CaptionModel
from src.data.dataset import FashionDataset
from src.data.vocab import PAD_IDX, FashionVocab
from src.data.collate import collate_fn


# -----------------------------
# CONFIG
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
LR = 3e-4
NUM_EPOCHS = 15
MAX_LEN = 30

DATA_DIR = r"..\metadata"     # adjust if needed
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
    transform=True
)

val_ds = FashionDataset(
    parquet_path=VAL_PATH,
    vocab=vocab,
    transform=True
)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)


# -----------------------------
# INITIALIZE MODEL
# -----------------------------
print("Initializing model...")
model = CaptionModel(d_model=512, vocab_size=vocab_size).to(DEVICE)

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2, verbose=True
)


# -----------------------------
# TRAIN ONE EPOCH
# -----------------------------
def train_one_epoch():
    model.train()
    running_loss = 0

    for images, captions in tqdm(train_loader, desc="Training"):
        images = images.to(DEVICE)
        captions = captions.to(DEVICE)

        optimizer.zero_grad()

        # Input: all tokens except last
        logits = model(images, captions[:, :-1])

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            captions[:, 1:].reshape(-1)
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
    running_loss = 0

    with torch.no_grad():
        for images, captions in tqdm(val_loader, desc="Validating"):
            images = images.to(DEVICE)
            captions = captions.to(DEVICE)

            logits = model(images, captions[:, :-1])
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                captions[:, 1:].reshape(-1)
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

            preds = model.generate(images, max_len=MAX_LEN)

            for p, t in zip(preds, captions):
                pred_sentence = vocab.decode(p)
                true_sentence = vocab.decode(t)

                score = nltk.translate.bleu_score.sentence_bleu(
                    [true_sentence],
                    pred_sentence,
                    weights=(0.25, 0.25, 0.25, 0.25)
                )
                score_list.append(score)

    return sum(score_list) / len(score_list)


# -----------------------------
# TRAINING LOOP
# -----------------------------
best_loss = float("inf")

print("Starting training...")

for epoch in range(1, NUM_EPOCHS + 1):

    print(f"\n---------------------")
    print(f"Epoch {epoch}/{NUM_EPOCHS}")
    print(f"---------------------")

    train_loss = train_one_epoch()
    val_loss = validate()
    bleu = compute_bleu()

    scheduler.step(val_loss)

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss:   {val_loss:.4f}")
    print(f"BLEU Score: {bleu:.4f}")

    # Save best model
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"✔ Saved new best model → {BEST_MODEL_PATH}")

print("\nTraining complete.")
print(f"Best validation loss: {best_loss:.4f}")

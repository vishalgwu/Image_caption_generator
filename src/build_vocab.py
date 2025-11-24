# src/build_vocab.py
import sys, os

# Add project root to path dynamically
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

import os
import json
import pickle
import pandas as pd

from src.data.text_preprocess import tokenize
from src.data.vocab import FashionVocab
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
METADATA_DIR = os.path.join(BASE_DIR, "metadata")

INPUT_PATH = os.path.join(METADATA_DIR, "metadata_with_captions.parquet")
VOCAB_PKL_PATH = os.path.join(METADATA_DIR, "vocab.pkl")
VOCAB_JSON_PATH = os.path.join(METADATA_DIR, "vocab.json")


def main():
    print(f"Loading captions from: {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    print("Loaded:", df.shape)

    vocab = FashionVocab(min_freq=2)

    total_tokens = 0

    # Build frequency dictionary
    for caption in df["caption"].astype(str):
        tokens = tokenize(caption)
        vocab.add_sentence(tokens)
        total_tokens += len(tokens)

    print("Total raw tokens:", total_tokens)
    print("Unique tokens before pruning:", len(vocab.word_freq))

    # Build final vocab
    vocab.build()
    print("Vocab size (including special tokens):", len(vocab))

    # ---------------------------
    # SAVE AS PICKLE (used by training)
    # ---------------------------
    print(f"Saving PKL vocab to: {VOCAB_PKL_PATH}")
    vocab.save(VOCAB_PKL_PATH)

    # ---------------------------
    # SAVE AS JSON (optional debugging)
    # ---------------------------
    vocab_json_data = {
        "min_freq": vocab.min_freq,
        "word2idx": vocab.word2idx,
        "idx2word": {str(k): v for k, v in vocab.idx2word.items()}
    }

    print(f"Saving JSON vocab to: {VOCAB_JSON_PATH}")
    with open(VOCAB_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(vocab_json_data, f, indent=2, ensure_ascii=False)

    print("Done building vocabulary.")


if __name__ == "__main__":
    main()

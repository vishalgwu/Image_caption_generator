# src/build_vocab.py

import os
import json
import pandas as pd
from collections import Counter

from text_preprocess import tokenize
from vocab import Vocabulary

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
METADATA_DIR = os.path.join(BASE_DIR, "metadata")

INPUT_PATH = os.path.join(METADATA_DIR, "metadata_with_captions.parquet")
VOCAB_PATH = os.path.join(METADATA_DIR, "vocab.json")

def main():
    print(f"Loading captions from: {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    print("Loaded:", df.shape)

    vocab = Vocabulary(min_freq=2)  # you can tune min_freq

    total_tokens = 0

    for caption in df["caption"].astype(str):
        tokens = tokenize(caption)
        total_tokens += len(tokens)
        vocab.add_sentence(tokens)

    print("Total raw tokens:", total_tokens)
    print("Unique tokens before prune:", len(vocab.word_freq))

    vocab.build()
    print("Vocab size (including special tokens):", len(vocab))

    # Save vocab as JSON: { "word2idx": {...}, "idx2word": {...} }
    data = {
        "min_freq": vocab.min_freq,
        "word2idx": vocab.word2idx,
        "idx2word": {str(k): v for k, v in vocab.idx2word.items()},
    }

    print(f"Saving vocab to: {VOCAB_PATH}")
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("Done.")

if __name__ == "__main__":
    main()

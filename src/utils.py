import json
from pathlib import Path

# Path to your vocab.json file (adjust path if needed)
VOCAB_PATH = Path("../metadata/vocab.json")

def load_vocab(vocab_path: str | Path = VOCAB_PATH):
    vocab_path = Path(vocab_path)

    with open(vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Convert keys/values to correct types
    word2idx = {k: int(v) for k, v in data["word2idx"].items()}
    idx2word = {int(k): v for k, v in data["idx2word"].items()}

    # Correct special tokens based on your vocab.json
    PAD = "<pad>"
    BOS = "<sos>"
    EOS = "<eos>"
    UNK = "<unk>"

    pad_idx = word2idx[PAD]
    bos_idx = word2idx[BOS]
    eos_idx = word2idx[EOS]
    unk_idx = word2idx[UNK]

    vocab_size = len(word2idx)

    return {
        "word2idx": word2idx,
        "idx2word": idx2word,
        "pad_idx": pad_idx,
        "bos_idx": bos_idx,
        "eos_idx": eos_idx,
        "unk_idx": unk_idx,
        "vocab_size": vocab_size
    }

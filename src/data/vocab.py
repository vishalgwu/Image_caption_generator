# src/data/vocab.py
import torch
import pickle
from collections import Counter
from typing import List, Dict
import os

PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

# Optional: keep index constants (not strictly required, but ok)
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3


class FashionVocab:
    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.word_freq = Counter()

        # add special tokens
        self._add_special_tokens()

        # set special indices from mapping (safe)
        self.pad_idx = self.word2idx[PAD_TOKEN]
        self.sos_idx = self.word2idx[SOS_TOKEN]
        self.eos_idx = self.word2idx[EOS_TOKEN]
        self.unk_idx = self.word2idx[UNK_TOKEN]

    def _add_special_tokens(self):
        for token in [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]:
            self._add_word_internal(token)

    def _add_word_internal(self, word: str):
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def add_sentence(self, tokens: List[str]):
        self.word_freq.update(tokens)

    def build(self):
        for word, freq in self.word_freq.items():
            if freq >= self.min_freq and word not in self.word2idx:
                self._add_word_internal(word)

    def __len__(self):
        return len(self.word2idx)

    def word_to_index(self, word: str):
        return self.word2idx.get(word, self.unk_idx)

    def index_to_word(self, idx: int):
        return self.idx2word.get(idx, UNK_TOKEN)

    def numericalize(self, tokens: List[str], add_special_tokens=True):
        ids = [self.word_to_index(t) for t in tokens]
        if add_special_tokens:
            ids = [self.sos_idx] + ids + [self.eos_idx]
        return ids

    def decode(self, token_ids):
        if torch.is_tensor(token_ids):
            token_ids = token_ids.tolist()

        words = []
        for idx in token_ids:
            word = self.idx2word.get(idx, UNK_TOKEN)
            if word in [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]:
                continue
            words.append(word)

        return " ".join(words)

    def encode(self, text: str):
        tokens = text.lower().strip().split()
        ids = [self.sos_idx] + [self.word_to_index(t) for t in tokens] + [self.eos_idx]
        return ids

    # ---------- save / load ----------
    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "FashionVocab":
        with open(path, "rb") as f:
            vocab = pickle.load(f)

        # 🔥 Backfill attributes if loading an old pickle 🔥
        if not hasattr(vocab, "pad_idx"):
            vocab.pad_idx = vocab.word2idx.get(PAD_TOKEN, 0)
        if not hasattr(vocab, "sos_idx"):
            vocab.sos_idx = vocab.word2idx.get(SOS_TOKEN, 1)
        if not hasattr(vocab, "eos_idx"):
            vocab.eos_idx = vocab.word2idx.get(EOS_TOKEN, 2)
        if not hasattr(vocab, "unk_idx"):
            vocab.unk_idx = vocab.word2idx.get(UNK_TOKEN, 3)

        return vocab

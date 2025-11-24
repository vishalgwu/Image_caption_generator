# src/data/vocab.py

import pickle
from collections import Counter
from typing import List, Dict

# Special tokens
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

# Make these available for training loop
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

        self._add_special_tokens()

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

    def word_to_index(self, word: str) -> int:
        return self.word2idx.get(word, UNK_IDX)

    def index_to_word(self, idx: int) -> str:
        return self.idx2word.get(idx, UNK_TOKEN)

    def numericalize(self, tokens: List[str], add_special_tokens=True):
        ids = [self.word_to_index(t) for t in tokens]
        if add_special_tokens:
            ids = [SOS_IDX] + ids + [EOS_IDX]
        return ids

    # ---------- Saving & Loading ----------
    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "FashionVocab":
        with open(path, "rb") as f:
            return pickle.load(f)

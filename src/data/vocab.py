# src/data/vocab.py
import torch
import pickle
from collections import Counter
from typing import List, Dict

# Special tokens
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"


class FashionVocab:
    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.word_freq = Counter()

        self._add_special_tokens()

    def _add_special_tokens(self):
        # Order matters
        for token in [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]:
            self._add_word_internal(token)

        # Store special indices
        self.pad_idx = self.word2idx[PAD_TOKEN]
        self.sos_idx = self.word2idx[SOS_TOKEN]
        self.eos_idx = self.word2idx[EOS_TOKEN]
        self.unk_idx = self.word2idx[UNK_TOKEN]

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
        return self.word2idx.get(word, self.unk_idx)

    def index_to_word(self, idx: int) -> str:
        return self.idx2word.get(idx, UNK_TOKEN)

    def numericalize(self, tokens: List[str], add_special_tokens=True):
        ids = [self.word_to_index(t) for t in tokens]
        if add_special_tokens:
            ids = [self.sos_idx] + ids + [self.eos_idx]
        return ids

    # ---------- Saving & Loading ----------
    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "FashionVocab":
        with open(path, "rb") as f:
            return pickle.load(f)

    # ---------- Decode & Encode ----------
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

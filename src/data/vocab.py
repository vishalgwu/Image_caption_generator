# src/vocab.py

from collections import Counter
from typing import List, Dict

PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

class Vocabulary:
    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.word_freq = Counter()

        # Reserve indices for special tokens
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
        """
        Build final vocab from word_freq using min_freq threshold.
        Special tokens are already in.
        """
        for word, freq in self.word_freq.items():
            if freq >= self.min_freq and word not in self.word2idx:
                self._add_word_internal(word)

    def __len__(self):
        return len(self.word2idx)

    def word_to_index(self, word: str) -> int:
        return self.word2idx.get(word, self.word2idx[UNK_TOKEN])

    def index_to_word(self, idx: int) -> str:
        return self.idx2word.get(idx, UNK_TOKEN)

    def numericalize(self, tokens: List[str], add_special_tokens: bool = True) -> List[int]:
        """
        Turn tokens into a list of indices, optionally adding <sos> and <eos>.
        """
        ids = [self.word_to_index(t) for t in tokens]
        if add_special_tokens:
            ids = [self.word_to_index(SOS_TOKEN)] + ids + [self.word_to_index(EOS_TOKEN)]
        return ids

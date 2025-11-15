# src/text_preprocess.py

import re
from typing import List

# Simple regex to keep letters, numbers, and spaces
_CLEAN_RE = re.compile(r"[^a-z0-9\s]+")

def normalize_text(text: str) -> str:
    """
    Lowercase, remove weird characters, and normalize spaces.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = _CLEAN_RE.sub(" ", text)    # remove punctuation
    text = re.sub(r"\s+", " ", text)   # collapse multiple spaces
    return text.strip()


def tokenize(text: str) -> List[str]:
    """
    Simple whitespace tokenizer. You can swap this later with
    spaCy, NLTK, or a BPE tokenizer if desired.
    """
    text = normalize_text(text)
    if not text:
        return []
    return text.split()

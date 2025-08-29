import os
from typing import Dict, List

# Minimal placeholder dataset utilities for neural embedding LM
# Intentionally lightweight so you can fill in logic.


class Vocabulary:
    def __init__(self):
        self.stoi: Dict[str, int] = {}
        self.itos: List[str] = []

    def add_token(self, tok: str) -> int:
        if tok not in self.stoi:
            idx = len(self.itos)
            self.stoi[tok] = idx
            self.itos.append(tok)
        return self.stoi[tok]

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.stoi[t] for t in tokens if t in self.stoi]

    def decode(self, ids: List[int]) -> List[str]:
        return [self.itos[i] for i in ids]

    def __len__(self):
        return len(self.itos)


def load_or_build_vocab(merges_path: str) -> Vocabulary:
    # placeholder: adapt to your BPE vocab artifacts
    vocab = Vocabulary()
    # add specials first to stabilise indices
    for sp in ("<s>", "</s>"):
        vocab.add_token(sp)
    if os.path.exists(merges_path):
        with open(merges_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                # naive: treat each pair component as token
                for p in parts[:2]:
                    vocab.add_token(p)
    return vocab


class NgramWindowDataset:
    """Create (context, target) pairs from a flat list of token ids.

    context_len = n (# previous tokens).
    """

    def __init__(self, ids: List[int], n: int):
        self.ids = ids
        self.n = n

    def __len__(self):
        return max(0, len(self.ids) - self.n)

    def __getitem__(self, idx: int):  # Tuple[List[int], int]
        # context: previous n tokens; target: next token
        start = idx
        end = idx + self.n
        ctx = self.ids[start:end]
        target = self.ids[end]
        return ctx, target

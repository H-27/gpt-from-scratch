# Minimal placeholder neural embedding language models
# Fill in with PyTorch implementation later.


class BigramEmbeddingLM:
    def __init__(self, vocab_size: int, embed_dim: int):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        # placeholder structures
        self.embeddings = None  # to be replaced by torch.nn.Embedding

    def forward(self, context_ids):
        # placeholder forward signature
        raise NotImplementedError


class NgramMLP:
    def __init__(self, vocab_size: int, n: int, embed_dim: int, hidden_dim: int):
        self.vocab_size = vocab_size
        self.n = n
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

    def forward(self, context_ids):
        raise NotImplementedError

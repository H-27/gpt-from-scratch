import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ngram_engine.nge_utils import apply_bpe


class ngram_lm(nn.Module):
    def __init__(
        self,
        n: int,  # number of tokens in context (n-1) + target (1)
        k: int,  # vocab size
        embed_dim: int,  # embedding dimension
        hidden_dim: int,  # hidden layer dimension
        alpha: float,  # learning rate
        optimizer: str = "adam",  # or pass in torch optimizer directly
        patience: int = 5,  # for early stopping
    ):
        super(ngram_lm, self).__init__()
        self.vocab, self.merge_rules = self.load_vocab_and_merge_rules(k, "adv")
        self.vocab_size = len(self.vocab)
        self.n = n
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.patience = patience
        self.embeddings = nn.Embedding(len(self.vocab), embed_dim)
        self.fc1 = nn.Linear(embed_dim * (n - 1), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, len(self.vocab))
        self.validation_set = self.load__and_prepare_validation_set()
        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        else:
            self.optimizer = optimizer
        self.train_set = self.apply_bpe_and_encode(
            open("data/corpora/Shakespeare_clean_train.txt", "r").read()
        )
        self.test_set = self.apply_bpe_and_encode(
            open("data/corpora/Shakespeare_clean_test.txt", "r").read()
        )
        self.validation_set = self.apply_bpe_and_encode(
            open("data/corpora/Shakespeare_clean_val.txt", "r").read()
        )

    def load_vocab_and_merge_rules(self, k: int, suffix: str):
        vocab = (
            open(f"data/bpe_outputs/vocab_k{k}{suffix}.txt", "r").read().splitlines()
        )
        merge_rules = (
            open(f"data/bpe_outputs/merges_k{k}{suffix}.txt", "r").read().splitlines()
        )
        return vocab, merge_rules

    def apply_bpe_and_encode(self, text: str):
        text = apply_bpe(text, self.merge_rules)
        stoi = {c: i for i, c in enumerate(self.vocab)}
        encode = lambda s: [stoi[c] for c in s]
        return encode(text)

    def decode(self, text):
        itos = {i: c for i, c in enumerate(self.vocab)}
        decode = lambda l: "".join([itos[i] for i in l])
        return decode(text)

    def get_batch(self, batch_size=32, block_size=8, fromset="train"):
        # Sample random indices
        if fromset == "train":
            text = self.train_set
        elif fromset == "val":
            text = self.validation_set
        elif fromset == "test":
            text = self.test_set

        ix = torch.randint(len(text) - block_size, (batch_size,))
        context = torch.stack([text[i : i + block_size] for i in ix])
        target = torch.stack([text[i + 1 : i + block_size + 1] for i in ix])
        return context, target

    def forward(self, context, target):
        # context_ids shape: (batch_size, n-1)
        embeds = self.embeddings(context)  # (batch_size, n-1, embed_dim)
        embeds = embeds.view(embeds.size(0), -1)  # (batch_size, (n-1)*embed_dim)
        hidden = torch.relu(self.fc1(embeds))  # (batch_size, hidden_dim)
        output = self.fc2(hidden)  # (batch_size, vocab_size)
        loss = F.cross_entropy(output, target)
        return output, loss

    def early_stopping_with_patience_check(self, loss_history):
        raise NotImplementedError

    def generate(self, context, n_new_tokens: int):
        raise NotImplementedError

    def calculate_perplexity(self, data_loader):
        raise NotImplementedError

    def train(self, epochs: int, num_steps: int):
        # train, then validate, test only after training complete
        for epoch in range(epochs):
            for step in range(num_steps):
                xb, yb = self.get_batch()
                logits, loss = self.forward(xb, yb)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                if step % 1000 == 0:
                    print(f"step {step}: loss {loss.item():.4f}")

    def optimize():
        # embeddings sizes:
        # k ≤ 150: d=64
        # 150 < k ≤ 400: d=96 or 128
        # 400 < k ≤ 800: d=160 or 192
        # 800 < k ≤ 1500: d=192–256

        raise NotImplementedError

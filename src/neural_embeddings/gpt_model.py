import torch
import torch.nn.functional as F
from torch import nn

from src.neural_embeddings.neural_ngrams import NgramLM


class CausalSelfAttentionHead(nn.Module):
    def __init__(self, embed_dim, head_size, context_size, dropout=0.1):
        super(CausalSelfAttentionHead, self).__init__()
        self.keys = nn.Linear(embed_dim, head_size, bias=False)
        self.queries = nn.Linear(embed_dim, head_size, bias=False)
        self.values = nn.Linear(embed_dim, head_size, bias=False)
        self.mask = torch.tril(torch.ones(context_size, context_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # x: (B, T, C)
        B, T, C = x.shape
        # Linear projections
        q = self.queries(x)
        k = self.keys(x)
        v = self.values(x)
        # Scaled dot-product attention logits (B, T, T)
        x = (q @ k.transpose(-2, -1)) / (C**0.5)
        # Causal mask (keep lower triangle incl diag)
        x = x.masked_fill(self.mask == 0, float("-inf"))

        # Softmax to probabilities; masked positions become 0
        x = torch.softmax(x, dim=-1)
        x = self.dropout(x)
        # Weighted sum of values
        x = x @ v  # (B, T, C)
        return x


class GPTLayer(nn.Module):
    def __init__(self, num_heads, embed_dim, context_size, dropout=0.1):
        super(GPTLayer, self).__init__()
        self.head_dims = embed_dim // num_heads
        # self.emb = nn.Embedding(vocab_size, embed_dim)
        self.head_layers = nn.ModuleList(
            [
                CausalSelfAttentionHead(
                    embed_dim,
                    self.head_dims,
                    context_size=context_size,
                    dropout=dropout,
                )
                for _ in range(num_heads)
            ]
        )
        self.dense_one = nn.Linear(embed_dim, 4 * embed_dim)
        self.dense_two = nn.Linear(4 * embed_dim, embed_dim)
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        attention_outputs = []
        # Run each head and collect outputs
        for layer in self.head_layers:
            x = layer(input)
            attention_outputs.append(x)
        x = torch.cat(attention_outputs, dim=-1)  # (B, T, num_heads * embed_dim)
        xp = self.projection(x)
        xp = self.dropout(xp)
        # Residual connection
        x = input + self.ln1(xp)
        # Feedforward network
        xf = self.dense_one(x)
        xf = nn.GELU()(xf)
        xf = self.dense_two(xf)
        xf = self.dropout(xf)
        # Residual connection
        x = x + self.ln2(xf)
        return x


class GPTModel(NgramLM):
    def __init__(
        self,
        n,
        k,
        embed_dim,
        hidden_dim,
        alpha,
        lam,
        dropout=0.1,
        num_heads=2,
        num_layers=2,
        device: str | None = None,
    ):
        # Inherit ngram base (loads vocab, BPE data, etc.)
        super().__init__(n, k, embed_dim, hidden_dim, alpha, lam, device=device)
        self.embedding_layer = nn.Embedding(self.vocab_size, embed_dim)
        self.attention_layers = nn.Sequential(
            *[
                GPTLayer(
                    num_heads=num_heads,
                    embed_dim=embed_dim,
                    context_size=n,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.out = nn.Linear(embed_dim, self.vocab_size)

    def forward(self, x):
        x = self.embedding_layer(x)
        x = self.attention_layers(x)
        x = self.ln_f(x)
        x = self.out(x)
        return x

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.n :]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


# For quick testing
size = 32
num_heads = 4
context_size = size
head = GPTLayer(embed_dim=size, num_heads=num_heads, context_size=context_size)
test_input = torch.randn(1, size, size)  # (B,T,C)
output = head(test_input)
print(output[0])
print(output.shape)
print(output[0].shape)

import torch
import torch.nn.functional as F
from torch import nn

from src.neural_embeddings.neural_ngrams import NgramLM


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
        # GPT specific embeddings
        self.token_embedding_table = nn.Embedding(self.vocab_size, embed_dim)
        self.position_embedding_table = nn.Embedding(n, embed_dim)
        self.blocks = nn.Sequential(
            *[
                Block(
                    n_embd=embed_dim,
                    n_head=num_heads,
                    n_ctx=n,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, self.vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.n_layers = num_layers
        # ensure all newly created GPT layers are moved to device
        self.to(self.device)

    def forward(self, idx, targets=None):
        idx = idx.to(self.device)
        if targets is not None:
            targets = targets.to(self.device)
        B, T = idx.shape
        if T > self.n:
            idx = idx[:, -self.n :]
            T = self.n
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits_full = self.lm_head(x)  # (B,T,V)
        loss = None
        if targets is not None:
            # Allow two target formats:
            # 1) (B,) single next-token targets (n-gram style)
            # 2) (B,T) full causal language modeling targets
            if targets.dim() == 1:  # single next-token targets (B,)
                loss = F.cross_entropy(logits_full[:, -1, :], targets)
                # Return only last-step logits to mimic (B,V) shape expected by interpolation code
                return logits_full[:, -1, :], loss
            else:  # (B,T) causal targets
                targets = targets[:, -T:]
                B2, T2, V = logits_full.shape
                logits_flat = logits_full.view(B2 * T2, V)
                targets_flat = targets.reshape(B2 * T2)
                loss = F.cross_entropy(logits_flat, targets_flat)
        return logits_full, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        idx = idx.to(self.device)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.n :]
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                vals, inds = torch.topk(logits, k)
                mask = torch.full_like(logits, float("-inf"))
                mask.scatter_(1, inds, vals)
                logits = mask
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B,1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx.cpu()


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size, n, embed_dim=256, dropout=0.1):
        super().__init__()
        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(n, n)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B,T,C)
        q = self.query(x)  # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * (C**-0.5)  # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,C)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size, n, embed_dim=256, dropout=0.1):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size, n, embed_dim, dropout) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(num_heads * head_size, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),  # switched to GELU per spec
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head, n_ctx, dropout=0.1):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_ctx, n_embd, dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

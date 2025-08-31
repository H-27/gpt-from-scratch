import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from src.neural_embeddings.neural_ngrams import NgramLM


class CausalSelfAttentionHead(nn.Module):
    """Single causal self-attention head.

    Projects input embeddings to queries, keys, values of size head_size.
    Applies scaled dot-product attention with a causal mask so position t
    can only attend to <= t positions.
    """

    def __init__(self, embed_dim, head_size, context_size, dropout=0.1):
        super(CausalSelfAttentionHead, self).__init__()
        self.keys = nn.Linear(embed_dim, head_size, bias=False)
        self.queries = nn.Linear(embed_dim, head_size, bias=False)
        self.values = nn.Linear(embed_dim, head_size, bias=False)
        # register buffer so it follows .to(device)
        self.register_buffer("mask", torch.tril(torch.ones(context_size, context_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # x: (B, T, C)
        B, T, C = x.shape
        q = self.queries(x)
        k = self.keys(x)
        v = self.values(x)
        x = (q @ k.transpose(-2, -1)) / (C**0.5)
        # slice to current length and ensure device match (buffer already on same device)
        causal = self.mask[:T, :T]
        x = x.masked_fill(causal == 0, float("-inf"))
        x = torch.softmax(x, dim=-1)
        x = self.dropout(x)
        x = x @ v  # (B, T, C)
        return x


class GPTLayer(nn.Module):
    def __init__(self, num_heads, embed_dim, context_size, dropout=0.1):
        super(GPTLayer, self).__init__()
        self.head_dims = embed_dim // num_heads
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
    """GPT model overriding n-gram training/eval utilities for sequence LM.

    Overrides:
      - get_batch: returns (inputs, targets) where targets is next token for each position
      - forward: returns (logits, loss) compatible with base signature
      - evaluate: computes mean token-level loss & perplexity
      - train_model: token-level training loop (does not use base n-gram fc layers)
    """

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
        context_size=32,
        device: str | None = None,
    ):
        # Inherit ngram base (loads vocab, BPE data, etc.)
        super().__init__(n, k, embed_dim, hidden_dim, alpha, lam, device=device)
        # Block size (context length) reuse n
        self.block_size = n
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
        self.dropout = nn.Dropout(dropout)
        # Move newly created submodules to device (base already set device)
        self.to(self.device)

    # ---------------- GPT overrides ----------------
    def get_batch(self, split="train", batch_size=16):
        """Return batch of token sequences for GPT training.
        Builds input (B, T) and target (B, T) where target is input shifted left.
        T == block_size (or remaining tokens if near end)."""
        if split == "train":
            ids = self.train_ids
        elif split == "val":
            ids = self.val_ids
        else:
            ids = self.test_ids
        T = self.block_size
        max_start = len(ids) - T - 1
        if max_start <= 0:
            raise ValueError("Not enough tokens for chosen block_size")
        starts = torch.randint(0, max_start, (batch_size,))
        x_batch = []
        y_batch = []
        for s in starts:
            chunk = ids[s : s + T + 1]  # (T+1)
            x_batch.append(chunk[:-1])  # first T
            y_batch.append(chunk[1:])  # next T
        x = torch.stack(x_batch).to(self.device)
        y = torch.stack(y_batch).to(self.device)
        return x, y

    def forward(self, x, targets=None):  # x: (B,T)
        x = self.embedding_layer(x)  # (B,T,C)
        x = self.attention_layers(x)  # (B,T,C)
        x = self.ln_f(x)
        logits = self.out(x)  # (B,T,V)
        loss = None
        if targets is not None:
            # Flatten for cross-entropy
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    def evaluate(self, eval_set="val", batch_size: int = 64):
        """Compute mean token-level loss & perplexity on eval split."""
        if eval_set == "val":
            ids = self.val_ids
        else:
            ids = self.test_ids
        self.eval()
        losses = []
        with torch.no_grad():
            T = self.block_size
            # Iterate over non-overlapping chunks for speed
            for i in range(0, len(ids) - T - 1, T):
                chunk = ids[i : i + T + 1]
                if len(chunk) < T + 1:
                    continue
                x = chunk[:-1].unsqueeze(0).to(self.device)
                y = chunk[1:].unsqueeze(0).to(self.device)
                _, loss = self.forward(x, y)
                losses.append(loss.item())
        if not losses:
            return float("nan"), float("nan"), float("nan")
        avg_loss = sum(losses) / len(losses)
        ppl = float(torch.exp(torch.tensor(avg_loss)))
        # No interpolation here; return same ppl for consistency with parent
        return avg_loss, ppl, ppl

    def train_model(self, epochs: int, num_steps: int, batch_size: int = 32):
        """Token-level training loop for GPT model."""
        for epoch in range(epochs):
            self.train()
            running = 0.0
            pbar = tqdm(total=num_steps, desc=f"GPT Epoch {epoch + 1}/{epochs}")
            for _ in range(num_steps):
                xb, yb = self.get_batch("train", batch_size=batch_size)
                logits, loss = self.forward(xb, yb)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                running += loss.item()
                pbar.update(1)
            train_loss = running / num_steps
            val_loss, val_ppl, _ = self.evaluate("val")
            print(
                f"epoch {epoch} train loss {train_loss:.4f} val loss {val_loss:.4f} val ppl {val_ppl:.4f}"
            )

    @torch.no_grad()
    def generate(
        self, idx, max_new_tokens, temperature: float = 1.0, top_k: int | None = None
    ):
        """Generate tokens autoregressively with temperature and optional top-k filtering.

        Args:
            idx: (B, T) LongTensor initial context
            max_new_tokens: number of tokens to sample
            temperature: >0 scaling factor applied to logits (lower = sharper)
            top_k: if set, keep only top_k logits each step before softmax
        Returns:
            (B, T + max_new_tokens) tensor of sampled token indices
        """
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.n :]
            logits, _ = self(idx_cond)  # unpack logits
            logits = logits[:, -1, :]  # last time step (B, vocab)
            if temperature != 1.0:
                logits = logits / temperature
            if top_k is not None:
                # Top-k filtering
                v, ix = torch.topk(logits, top_k, dim=-1)
                mask = torch.full_like(logits, float("-inf"))
                mask.scatter_(1, ix, v)
                logits = mask
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# For quick testing
if __name__ == "__main__":
    size = 32
    num_heads = 4
    context_size = size
    head = GPTLayer(embed_dim=size, num_heads=num_heads, context_size=context_size)
    test_input = torch.randn(1, size, size)
    output = head(test_input)
    print(output.shape)

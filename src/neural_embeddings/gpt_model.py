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
        super().__init__(n, k, embed_dim, hidden_dim, alpha, lam, device=device)
        self.block_size = context_size
        self.embedding_layer = nn.Embedding(self.vocab_size, embed_dim)
        self.pos_embeddings = nn.Embedding(self.block_size, embed_dim)
        self.attention_layers = nn.Sequential(
            *[
                GPTLayer(
                    num_heads=num_heads,
                    embed_dim=embed_dim,
                    context_size=self.block_size,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.out = nn.Linear(embed_dim, self.vocab_size)
        self.dropout = nn.Dropout(dropout)
        # Remove unused base n-gram layers so optimizer only sees GPT stack
        self.embeddings = None
        self.fc1 = None
        self.fc2 = None
        # Rebuild optimizer to include ONLY current (GPT) parameters (base optimizer held stale param list)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        self.to(self.device)
        print(
            f"[GPTModel] n={self.n} block_size={self.block_size} params={sum(p.numel() for p in self.parameters())}"
        )

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
        x = x + self.pos_embeddings(torch.arange(x.size(1), device=x.device))
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
        self,
        idx,
        max_new_tokens,
        temperature: float = 1.0,
        top_k: int | None = None,
        sampling: bool = True,
        stop_on_eos: bool = True,
    ):
        """Autoregressive generation with temperature, top-k, sampling/argmax and optional EOS early stop."""
        self.eval()
        stoi = {c: i for i, c in enumerate(self.vocab)}
        eos_idx = stoi.get("</s>")
        if idx.dim() == 1:
            idx = idx.unsqueeze(0)
        idx = idx.to(self.device)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            if temperature != 1.0:
                logits = logits / temperature
            if top_k is not None and top_k < logits.size(-1):
                v, ix = torch.topk(logits, top_k, dim=-1)
                mask = torch.full_like(logits, float("-inf"))
                mask.scatter_(1, ix, v)
                logits = mask
            probs = F.softmax(logits, dim=-1)
            if sampling:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = probs.argmax(dim=-1, keepdim=True)
            idx = torch.cat((idx, idx_next), dim=1)
            if stop_on_eos and eos_idx is not None and (idx_next == eos_idx).all():
                break
        return idx.cpu()

    @torch.no_grad()
    def token_accuracy(self, eval_set: str = "val") -> float:
        """Next-token accuracy using greedy argmax on validation/test set."""
        if eval_set == "val":
            ids = self.val_ids
        elif eval_set == "test":
            ids = self.test_ids
        else:
            ids = self.train_ids
        total = 0
        correct = 0
        T = self.block_size
        for i in range(0, len(ids) - T - 1, T):
            chunk = ids[i : i + T + 1]
            if len(chunk) < T + 1:
                continue
            x = chunk[:-1].unsqueeze(0).to(self.device)
            y = chunk[1:].to(self.device)
            logits, _ = self.forward(x, y.unsqueeze(0))  # logits (1,T,V)
            preds = logits.argmax(dim=-1).squeeze(0)
            correct += (preds == y).sum().item()
            total += y.numel()
        return correct / total if total else float("nan")


# For quick testing
if __name__ == "__main__":
    # Quick component shape test for a single layer
    size = 32
    num_heads = 4
    context_size = size
    head = GPTLayer(embed_dim=size, num_heads=num_heads, context_size=context_size)
    test_input = torch.randn(1, size, size)
    output = head(test_input)
    print("Single GPTLayer output shape:", output.shape)

    # Sanity training run for GPTModel to verify loss decreases
    try:
        gpt = GPTModel(
            n=3,  # n-gram order (not the block size for GPT)
            k=1000,  # pick a k you have cached BPE data for
            embed_dim=256,
            hidden_dim=256,
            alpha=1e-1,  # learning rate
            lam=1.0,
            num_heads=4,
            num_layers=2,
            context_size=64,  # block length for GPT context
        )
    except Exception as e:
        print("Failed to construct GPTModel (check data paths / k):", e)
        raise SystemExit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpt.to(device)
    print("Starting quick sanity training (200 steps)...")
    gpt.train()
    report_every = 50
    losses = []
    for step in range(200):
        xb, yb = gpt.get_batch("train", batch_size=32)
        _, loss = gpt.forward(xb, yb)
        gpt.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gpt.optimizer.step()
        losses.append(loss.item())
        if (step + 1) % report_every == 0:
            avg_recent = sum(losses[-report_every:]) / report_every
            print(f"step {step + 1} avg_loss {avg_recent:.4f}")
    print(f"Initial 50-step avg: {sum(losses[:report_every]) / report_every:.4f}")
    print(f"Final 50-step avg:   {sum(losses[-report_every:]) / report_every:.4f}")
    gpt.eval()
    val_loss, val_ppl, _ = gpt.evaluate("val")
    print(f"Validation after sanity run: loss {val_loss:.4f} ppl {val_ppl:.2f}")

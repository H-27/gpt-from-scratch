import ast
import os
from collections import defaultdict
from typing import List, Tuple  # moved to top

import torch  # ensure single import
import torch.nn as nn
import torch.nn.functional as F

from src.neural_embeddings.neural_utils import (
    append_run_result,
    build_aggregate_from_checkpoints,
    save_history_csv,
    save_history_jsonl,
)
from src.ngram_engine import nge_utils
from src.ngram_engine.nge_utils import apply_bpe


class NgramLM(nn.Module):
    def __init__(
        self,
        n: int,  # number of tokens in context (n-1) + target (1)
        k: int,  # vocab size reference (merges)
        embed_dim: int,
        hidden_dim: int,
        alpha: float,
        lam: float = 1.0,
        optimizer: str = "adam",
        patience: int = 5,
        device: str = None,
    ):
        print("Initializing NgramLM model...")
        super(NgramLM, self).__init__()
        print("Loading vocab and merge rules...")
        self.vocab, self.merge_rules = self.load_vocab_and_merge_rules(k, "_adv")
        self.vocab.extend(["<s>", "</s>"])
        self.vocab_size = len(self.vocab)
        self.n = n
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.lam = lam
        self.patience = patience
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.embeddings = nn.Embedding(self.vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim * (n - 1), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, self.vocab_size)
        # dataset ids (convert once to tensors)
        print("Loading training set with k =", k, "...")
        self.train_ids = torch.tensor(
            [
                int(s)
                for s in open(f"data/emb_lm/encoded_texts/bpe_cache_k{k}_adv_train.txt")
                .read()
                .split()
            ],
            dtype=torch.long,
        )
        self.val_ids = torch.tensor(
            [
                int(s)
                for s in open(f"data/emb_lm/encoded_texts/bpe_cache_k{k}_adv_valid.txt")
                .read()
                .split()
            ],
            dtype=torch.long,
        )
        self.test_ids = torch.tensor(
            [
                int(s)
                for s in open(f"data/emb_lm/encoded_texts/bpe_cache_k{k}_adv_test.txt")
                .read()
                .split()
            ],
            dtype=torch.long,
        )
        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        elif optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=alpha)
        else:  # allow passing an instantiated optimizer
            self.optimizer = optimizer

        self.contexts, self.ngrams = nge_utils.load_ngrams_and_contexts(
            n, k, suffix="_adv"
        )
        self.encode_ngrams()
        self.to(self.device)

    def load_vocab_and_merge_rules(self, k: int, suffix: str):
        vocab = (
            open(f"data/bpe_outputs/vocab_with_k{k}{suffix}.txt", "r")
            .read()
            .splitlines()
        )
        merge_rules = (
            open(f"data/bpe_outputs/merges_k{k}{suffix}.txt", "r").read().splitlines()
        )
        return vocab, merge_rules

    def apply_bpe_and_encode(self, text: str):
        # track progress if train_ids not yet set
        track_progress = not hasattr(self, "train_ids")
        tokens = apply_bpe(text, self.merge_rules, track_progress=track_progress)
        # tokens is list of token sequences (sentences); flatten
        flat = []
        for sent in tokens:
            flat.extend(sent)
        flat = [token for token in flat if token != "0"]
        stoi = {c: i for i, c in enumerate(self.vocab)}

        def encode_seq(seq):
            return [stoi[c] for c in seq]

        return encode_seq(flat)

    def decode(self, text):
        itos = {i: c for i, c in enumerate(self.vocab)}

        def decode_indices(indices):
            return "".join([itos[i] for i in indices])

        return decode_indices(text)

    def encode_ngrams(self):
        stoi = {c: i for i, c in enumerate(self.vocab)}
        print("Encoding ngrams...")
        encoded_contexts = defaultdict(int)
        # contexts keys may already be tuples; if strings, parse safely
        for key, count in self.contexts.items():
            try:
                if isinstance(key, str):
                    tup = ast.literal_eval(key)
                else:
                    tup = key
                if not isinstance(tup, (list, tuple)):
                    continue
                enc = [stoi[t] for t in tup if t in stoi]
                if not enc:
                    continue
                encoded_contexts[tuple(enc)] += count
            except (ValueError, SyntaxError, TypeError) as e:
                print(f"Skipping malformed context key: {key}. Error: {e}")

        encoded_ngrams = defaultdict(int)
        for key, count in self.ngrams.items():
            try:
                if isinstance(key, str):
                    tup = ast.literal_eval(key)
                else:
                    tup = key
                if not isinstance(tup, (list, tuple)):
                    continue
                enc = [stoi[t] for t in tup if t in stoi]
                if not enc:
                    continue
                encoded_ngrams[tuple(enc)] += count
            except (ValueError, SyntaxError, TypeError) as e:
                print(f"Skipping malformed ngram key: {key}. Error: {e}")
        self.contexts = dict(encoded_contexts)
        self.ngrams = dict(encoded_ngrams)

    def get_batch(self, split="train", batch_size=32):
        if split == "train":
            ids = self.train_ids
        elif split == "val":
            ids = self.val_ids
        else:
            ids = self.test_ids
        # contexts of length n-1 predicting next token
        max_start = len(ids) - self.n
        if max_start <= 0:
            raise ValueError("Not enough tokens for the chosen n")
        idx = torch.randint(0, max_start, (batch_size,))
        # build tensors
        contexts = torch.stack([ids[i : i + self.n - 1] for i in idx])  # (B, n-1)
        targets = torch.tensor(
            [ids[i + self.n - 1] for i in idx], dtype=torch.long
        )  # (B,)
        return contexts.to(self.device), targets.to(self.device)

    def get_distribution_from_context(self, context, n):
        # load ngrams and contexts from file

        # Get all possible next words
        possible_next_words = {}
        while len(possible_next_words) == 0 and n > 1:
            backoff_weight = pow(0.4, self.n - n)  # backoff weight
            # if context is empty, use unigram
            if len(context) == 0:
                unigrams_and_probs = self.get_unigram_probs()
                for tok, p in unigrams_and_probs.items():
                    possible_next_words[tok] = p * backoff_weight
            # find all ngrams that match the context
            for token in self.vocab:
                ngram = tuple(context.tolist()) + (token,)
                next_word_probs = self.get_ngram_probs(ngram, n) * backoff_weight
                possible_next_words[token] = next_word_probs
            # if no possible next word probability is above the threshold, backoff to n-1 gram
            if not possible_next_words:
                n -= 1
                context = context[-(n - 1) :]
        # draw a word based on the probabilities
        if possible_next_words:
            probs = list(possible_next_words.values())
            # This converts the scores into a probability distribution
            total_prob = sum(probs)
            if total_prob == 0:
                unigram_probs = self.get_unigram_probs()
                if not unigram_probs:
                    return "</s>"  # Should not happen
                # Get the unigram with the highest probability
                best_unigram = max(unigram_probs, key=unigram_probs.get)
                return best_unigram  # Remove the [0] indexing
            probs = [p / total_prob for p in probs]
            return torch.tensor(probs, dtype=torch.float32, device=self.device)
        return None

    def get_ngram_probs(self, ngram, n):
        # if unigram
        if n == 1:
            return self.get_unigram_probs()
        # load ngrams and contexts from file
        ngrams, contexts = self.ngrams, self.contexts

        context = ngram[:-1]
        ngram_count = ngrams.get(ngram, 0)
        context_count = contexts.get(context, 0)

        # Add-one smoothing
        return (ngram_count + 1) / (context_count + self.vocab_size)

    def get_ngram_prob_distribution(self, context):
        """Return add-one smoothed n-gram probability distribution P(token|context).
        context: list/tuple/tensor of token ids (desired length n-1; if longer we take last n-1)."""
        # normalize context length
        if isinstance(context, torch.Tensor):
            context_list = context.tolist()
        else:
            context_list = list(context)
        if len(context_list) > self.n - 1:
            context_list = context_list[-(self.n - 1) :]
        # convert to tuple for dict lookup
        contexts = tuple(context_list)
        # context count (may be 0 if unseen)
        context_counts = self.contexts.get(contexts, 0)
        probs = []
        for tok_id in range(self.vocab_size):
            ngram_count = self.ngrams.get(contexts + (tok_id,), 0)
            probs.append((ngram_count + 1) / (context_counts + self.vocab_size))
        return torch.tensor(probs, dtype=torch.float32, device=self.device)

    def get_interpolated_probability(self, neural_logits, context, lam):
        """Interpolate neural softmax probs with n-gram add-one distribution.
        neural_logits: (1,V) tensor
        context: sequence of token ids (length <= n-1)
        lam: weight on neural model (0..1)
        Returns (1,V) probs."""
        neural_probs = F.softmax(neural_logits, dim=-1)
        ngram_probs = self.get_ngram_prob_distribution(context)  # (V,)
        ngram_probs = ngram_probs.unsqueeze(0)  # (1,V)
        interpolated_probs = lam * neural_probs + (1 - lam) * ngram_probs
        return interpolated_probs

    def forward(self, context, target=None):
        # context: (B, n-1)
        x = self.embeddings(context)  # (B, n-1, d)
        x = x.view(x.size(0), -1)  # (B, (n-1)*d)
        x = torch.relu(self.fc1(x))  # (B, hidden)
        x = self.fc2(x)  # (B, V)
        loss = None
        if target is not None:
            loss = F.cross_entropy(x, target)
        return x, loss

    def early_stopping_with_patience_check(self, loss_history):
        raise NotImplementedError

    def generate(self, context, n_new_tokens: int):
        """Generate tokens autoregressively.
        context: (B, <= n-1) LongTensor. If shorter, left-pad with <s>. If longer, last n-1 tokens are used.
        Maintains only sliding window of size n-1 for conditioning, but accumulates full sequence for return.
        """
        self.eval()
        with torch.no_grad():
            # Build vocab index map once
            stoi = {c: i for i, c in enumerate(self.vocab)}
            start_idx = stoi.get("<s>")
            # Ensure 2D
            if context.dim() == 1:
                context = context.unsqueeze(0)
            B, L = context.shape
            # Move to device
            context = context.to(self.device)
            # If length < n-1, pad on left with <s>
            need = self.n - 1 - L
            if need > 0:
                pad = torch.full(
                    (B, need), start_idx, dtype=torch.long, device=self.device
                )
                context = torch.cat([pad, context], dim=1)
            generated = context.clone()
            for _ in range(n_new_tokens):
                window = generated[:, -(self.n - 1) :]
                logits, _ = self.forward(window)
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # (B,1)
                generated = torch.cat([generated, next_token], dim=1)
            return generated.cpu()

    def evaluate(self, eval_set="val"):
        if eval_set == "val":
            eval_data = self.val_ids
        else:
            eval_data = self.test_ids
        loss_list = []
        loss_interpolated = 0.0
        ctx_len = self.n - 1
        c = 0.0
        for i in range(ctx_len, len(eval_data)):
            context = (
                eval_data[i - ctx_len : i].unsqueeze(0).to(self.device)
            )  # (1, n-1)
            target = eval_data[i].unsqueeze(0).to(self.device)  # (1,)
            logits, loss = self.forward(context, target)
            loss_list.append(loss.item())
            if self.lam == 1.0:
                continue
            else:
                l_interpolated = torch.log(
                    self.get_interpolated_probability(logits, context[0], self.lam)[
                        0, target
                    ]
                )
            loss_interpolated += -l_interpolated.item()
            c += 1

        if not loss_list:  # safety
            return float("nan"), float("nan")
        avg_loss = sum(loss_list) / len(loss_list)
        perplexity = torch.exp(torch.tensor(avg_loss))
        perplexity_interpolated = loss_interpolated / (c + 1e-8)
        perplexity_interpolated = torch.exp(torch.tensor(perplexity_interpolated))
        return avg_loss, perplexity, perplexity_interpolated

    def train_model(self, epochs: int, num_steps: int):
        ckpt_dir = "data/emb_lm/checkpoints"
        os.makedirs(ckpt_dir, exist_ok=True)
        for epoch in range(epochs):
            epoch_loss = 0.0
            for step in range(num_steps):
                xb, yb = self.get_batch()
                logits, loss = self.forward(xb, yb)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            print(f"epoch {epoch} train loss {epoch_loss / num_steps:.4f}")
            val_loss, val_ppl, val_ppl_interpolated = self.evaluate("val")
            print(
                f"epoch {epoch} val loss {val_loss:.4f} val ppl {val_ppl:.4f} val ppl interpolated {val_ppl_interpolated:.4f}"
            )


# New: make early stopping utility importable (moved to module level)


def train_with_early_stopping(
    model: NgramLM,
    max_epochs: int,
    num_steps: int,
    patience: int,
    improvement_delta: float = 1e-2,
) -> Tuple[float, dict, List[Tuple[int, float, float, float, float]]]:
    """Train model with early stopping.
    Returns (best_val_loss, best_state_dict (cpu tensors), history list of tuples).
    history tuple schema: (epoch, avg_train_loss, val_loss, val_ppl, val_ppl_interpolated)
    """
    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0
    history: List[Tuple[int, float, float, float, float]] = []
    for epoch in range(max_epochs):
        model.train()
        running = 0.0
        for _ in range(num_steps):
            xb, yb = model.get_batch()
            logits, loss = model.forward(xb, yb)
            model.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            model.optimizer.step()
            running += loss.item()
        avg_train = running / num_steps
        val_loss, val_ppl, val_ppl_interpolated = model.evaluate("val")
        history.append(
            (
                epoch,
                avg_train,
                val_loss,
                float(val_ppl),
                float(val_ppl_interpolated),
            )
        )
        print(
            f"epoch {epoch} train {avg_train:.4f} val_loss {val_loss:.4f} val_ppl {val_ppl:.4f} val_ppl_interpolated {val_ppl_interpolated:.4f}"
        )
        if val_loss < best_val - improvement_delta:
            best_val = val_loss
            epochs_no_improve = 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break
    return best_val, best_state, history


if __name__ == "__main__":
    # Hyperparameter value lists (adjust as needed)
    n_values = [1, 2, 3, 4, 5]  # keep n fixed here for simplicity
    k_values = [50, 100, 150, 250, 350, 500, 750, 1000, 1250, 1500, 2000]
    a_values = [1e-1, 1e-2, 1e-3, 1e-4]  # learning rates
    h_values = [128, 256]  # hidden layer sizes
    max_epochs = 30
    num_steps = 500  # steps (batches) per epoch
    patience = 4  # early stopping patience
    TOP_K = 3  # keep best K models by validation loss

    ckpt_root = "data/emb_lm/grid_ckpts"
    os.makedirs(ckpt_root, exist_ok=True)
    aggregate_csv = os.path.join(ckpt_root, "grid_results.csv")

    top_k_models = []  # list of (val_loss, ckpt_path, config)

    for n in n_values:
        for k in k_values:
            for lr in a_values:
                for h in h_values:
                    print("\n=== Config n", n, "k", k, "lr", lr, "h", h, "===")
                    model = NgramLM(
                        n=n,
                        k=k,
                        hidden_dim=h,
                        embed_dim=256,
                        alpha=lr,
                        lam=1.0,  # train pure neural model
                        patience=patience,
                    )
                    best_val, best_state, history = train_with_early_stopping(
                        model, max_epochs, num_steps, patience
                    )
                    # derive metrics
                    last_val_ppl = history[-1][3] if history else None
                    ckpt_name = f"n{n}_k{k}_h{h}_lr{lr:.0e}_val{best_val:.4f}.pt"
                    ckpt_path = os.path.join(ckpt_root, ckpt_name)
                    torch.save(
                        {
                            "model_state": best_state,
                            "config": {
                                "n": n,
                                "k": k,
                                "h": h,
                                "lr": lr,
                                "lam": 1.0,
                            },
                            "best_val_loss": best_val,
                            "history": history,
                        },
                        ckpt_path,
                    )
                    # save histories in CSV & JSONL
                    base_hist = ckpt_name.replace(".pt", "")
                    save_history_csv(
                        history,
                        os.path.join(ckpt_root, base_hist + "_history.csv"),
                    )
                    save_history_jsonl(
                        history,
                        os.path.join(ckpt_root, base_hist + "_history.jsonl"),
                    )
                    # append aggregate row
                    append_run_result(
                        aggregate_csv,
                        {"n": n, "k": k, "h": h, "lr": lr, "lam": 1.0},
                        best_val,
                        last_val_ppl,
                        len(history),
                    )
                    top_k_models.append(
                        (best_val, ckpt_path, {"n": n, "k": k, "h": h, "lr": lr})
                    )
                    top_k_models.sort(key=lambda x: x[0])
                    if len(top_k_models) > TOP_K:
                        worst = top_k_models.pop()  # remove worst
                        try:
                            os.remove(worst[1])
                            print("Removed worst checkpoint", worst[1])
                        except OSError:
                            pass
                    print("Current top-k (best first):")
                    for rank, (vl, path, cfg) in enumerate(top_k_models, 1):
                        print(
                            f" {rank}. val_loss={vl:.4f} cfg={cfg} file={os.path.basename(path)}"
                        )

    # build / refresh aggregate CSV from checkpoints (ensures consistency)
    build_aggregate_from_checkpoints(ckpt_root, aggregate_csv)

    print("\nFinal top-k models:")
    for rank, (vl, path, cfg) in enumerate(top_k_models, 1):
        print(f" {rank}. val_loss={vl:.4f} cfg={cfg} file={os.path.basename(path)}")

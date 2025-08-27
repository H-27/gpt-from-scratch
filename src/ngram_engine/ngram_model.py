"""Task 2: N-gram language modeling utilities (Laplace smoothing, interpolation,
perplexity, and generation) operating on BPE-tokenized sentences.

Usage sketch:
    from .nge_utils import split_into_sentences_and_normalize
    from .ngram_model import NGramModel, build_and_evaluate

    vocab = open("data/bpe_outputs/vocab_with_k1000.txt", encoding="utf-8").read().splitlines()
    train_txt = open("data/corpora/Shakespeare_clean_train.txt", encoding="utf-8").read()
    val_txt = open("data/corpora/Shakespeare_clean_valid.txt", encoding="utf-8").read()
    test_txt = open("data/corpora/Shakespeare_clean_test.txt", encoding="utf-8").read()

    train_sent = split_into_sentences_and_normalize(train_txt, vocab)
    val_sent = split_into_sentences_and_normalize(val_txt, vocab)
    test_sent = split_into_sentences_and_normalize(test_txt, vocab)

    model = NGramModel.from_sentences(train_sent, max_n=4, vocab_tokens=vocab)
    lambdas = model.tune_lambdas(val_sent, max_n=4, step=0.1)  # interpolation weights
    ppl = model.perplexity(test_sent, max_n=4, lambdas=lambdas)
    print("Test perplexity", ppl)
    print(model.generate(["<s>"], max_tokens=40))
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

# ------------------------- Data Structures -------------------------


@dataclass
class NGramCounts:
    ngrams: defaultdict[Tuple[str, ...], int]
    contexts: defaultdict[Tuple[str, ...], int]
    max_n: int
    vocab_size: int


# ------------------------- Core Model -------------------------


class NGramModel:
    def __init__(self, counts: NGramCounts):
        self.counts = counts

    @classmethod
    def from_sentences(
        cls,
        sentences: Sequence[Sequence[str]],
        max_n: int,
        vocab_tokens: Sequence[str],
    ) -> "NGramModel":
        ngrams = defaultdict(int)
        contexts = defaultdict(int)
        for sent in sentences:
            # ensure list/tuple
            tokens = list(sent)
            L = len(tokens)
            for n in range(1, max_n + 1):
                for i in range(L - n + 1):
                    ngram = tuple(tokens[i : i + n])
                    context = ngram[:-1]
                    ngrams[ngram] += 1
                    contexts[context] += 1
        vocab_size = len(vocab_tokens)
        return cls(
            NGramCounts(
                ngrams=ngrams, contexts=contexts, max_n=max_n, vocab_size=vocab_size
            )
        )

    # --------------------- Probability Estimation ---------------------
    def _laplace_prob(self, ngram: Tuple[str, ...]) -> float:
        """Add-one (Laplace) smoothed probability for given ngram.
        ngram: (w_{i-n+1}, ..., w_i)
        Returns P(w_i | context) where context length = len(ngram)-1.
        """
        count_full = self.counts.ngrams.get(ngram, 0)
        context = ngram[:-1]
        count_context = self.counts.contexts.get(context, 0)
        V = self.counts.vocab_size
        return (count_full + 1) / (count_context + V)

    def prob(
        self,
        history: Sequence[str],
        token: str,
        *,
        max_n: Optional[int] = None,
        lambdas: Optional[Sequence[float]] = None,
        backoff: bool = True,
    ) -> float:
        """Compute probability of token given history.
        If lambdas provided -> linear interpolation across orders 1..max_n.
        Else if backoff True -> highest order available else shorten history.
        history: sequence ending with last seen tokens (excluding current token).
        """
        if max_n is None:
            max_n = self.counts.max_n
        history = list(history)
        # Ensure we consider at most max_n-1 length of history
        if len(history) > max_n - 1:
            history = history[-(max_n - 1) :]

        if lambdas is not None:
            # Interpolation: lambdas[order-1] corresponds to n=order
            # Normalize lambdas defensively
            s = sum(lambdas)
            if s == 0:
                raise ValueError("Sum of lambdas is zero.")
            norm_lambdas = [weight / s for weight in lambdas]
            total = 0.0
            for n, lam in enumerate(norm_lambdas, start=1):
                # context for this order is last n-1 tokens
                ctx = history[-(n - 1) :] if n > 1 else []
                ngram = tuple(ctx + [token])
                total += lam * self._laplace_prob(ngram)
            return total

        # Backoff: try longest context down to unigram
        for n in range(min(max_n, len(history) + 1), 0, -1):
            ctx = history[-(n - 1) :] if n > 1 else []
            ngram = tuple(ctx + [token])
            prob = self._laplace_prob(ngram)
            if (
                prob > 0
            ):  # Laplace always > 0; keep logic for potential alternative smoothing
                return prob
        # Should not reach (Laplace ensures non-zero) but fallback:
        return 1.0 / self.counts.vocab_size

    # --------------------- Perplexity ---------------------
    def sentence_log_prob(
        self,
        sentence: Sequence[str],
        *,
        max_n: Optional[int] = None,
        lambdas: Optional[Sequence[float]] = None,
    ) -> float:
        if max_n is None:
            max_n = self.counts.max_n
        logp = 0.0
        # iterate tokens starting after first (since first usually <s>)
        for i in range(1, len(sentence)):
            history = sentence[max(0, i - (max_n - 1)) : i]
            token = sentence[i]
            p = self.prob(history, token, max_n=max_n, lambdas=lambdas)
            logp += math.log(p)
        return logp

    def perplexity(
        self,
        sentences: Sequence[Sequence[str]],
        *,
        max_n: Optional[int] = None,
        lambdas: Optional[Sequence[float]] = None,
    ) -> float:
        token_count = 0
        total_logp = 0.0
        for sent in sentences:
            if len(sent) < 2:
                continue
            total_logp += self.sentence_log_prob(sent, max_n=max_n, lambdas=lambdas)
            token_count += max(1, len(sent) - 1)  # exclude first <s>
        if token_count == 0:
            return float("inf")
        # perplexity = exp(- total_logp / token_count)
        return math.exp(-total_logp / token_count)

    # --------------------- Interpolation Weight Tuning ---------------------
    def tune_lambdas(
        self,
        validation_sentences: Sequence[Sequence[str]],
        *,
        max_n: Optional[int] = None,
        step: float = 0.1,
    ) -> List[float]:
        """Grid search over simplex with given step for interpolation weights.
        Only practical for max_n <= 4 with coarse steps.
        Returns list of weights length=max_n.
        """
        if max_n is None:
            max_n = self.counts.max_n
        if max_n == 1:
            return [1.0]
        increments = int(1 / step) + 1
        best = None
        best_ppl = float("inf")

        def recurse(prefix: List[float], remaining: int, remaining_sum: float):
            nonlocal best, best_ppl
            if remaining == 1:
                weights = prefix + [remaining_sum]
                ppl = self.perplexity(
                    validation_sentences, max_n=max_n, lambdas=weights
                )
                if ppl < best_ppl:
                    best_ppl = ppl
                    best = weights
                return
            for i in range(increments + 1):
                w = i * step
                if w > remaining_sum:
                    break
                recurse(prefix + [w], remaining - 1, remaining_sum - w)

        recurse([], max_n, 1.0)
        # Normalize (floating error) and return
        if best is None:
            return [1.0] + [0.0] * (max_n - 1)
        s = sum(best)
        return [w / s for w in best]

    # --------------------- Generation ---------------------
    def generate(
        self,
        start_tokens: Sequence[str],
        *,
        max_tokens: int = 50,
        stop_token: str = "</s>",
        mode: str = "argmax",  # or "sample"
        temperature: float = 1.0,
        max_n: Optional[int] = None,
        lambdas: Optional[Sequence[float]] = None,
    ) -> List[str]:
        if max_n is None:
            max_n = self.counts.max_n
        generated = list(start_tokens)
        for _ in range(max_tokens):
            history = generated[-(max_n - 1) :]
            # Collect candidate probs (unigram vocab approximation via observed next tokens)
            # For practicality, derive candidate set: all tokens seen as last element in ngrams of order 1
            # Extract once lazily
            if not hasattr(self, "_unigram_cache"):
                self._unigram_cache = [
                    ng[-1] for ng in self.counts.ngrams.keys() if len(ng) == 1
                ]
            candidates = self._unigram_cache
            probs = []
            for tok in candidates:
                p = self.prob(history, tok, max_n=max_n, lambdas=lambdas)
                probs.append(p)
            # Normalize
            total = sum(probs)
            if total == 0:
                # fallback uniform
                probs = [1 / len(probs)] * len(probs)
            else:
                probs = [p / total for p in probs]

            if mode == "sample":
                # temperature scaling
                if temperature != 1.0:
                    scaled = [p ** (1.0 / temperature) for p in probs]
                    s = sum(scaled)
                    probs = [p / s for p in scaled]
                r = random.random()
                cum = 0.0
                chosen = candidates[-1]
                for tok, p in zip(candidates, probs):
                    cum += p
                    if r <= cum:
                        chosen = tok
                        break
            else:  # argmax
                max_idx = max(range(len(candidates)), key=lambda i: probs[i])
                chosen = candidates[max_idx]
            generated.append(chosen)
            if chosen == stop_token:
                break
        return generated


# ------------------------- High-level helper -------------------------


def build_and_evaluate(
    train_sentences: Sequence[Sequence[str]],
    val_sentences: Sequence[Sequence[str]],
    test_sentences: Sequence[Sequence[str]],
    *,
    max_n: int,
    vocab_tokens: Sequence[str],
    interp_step: float = 0.1,
) -> Dict[str, object]:
    """Build model, compute perplexities for orders 1..max_n and interpolation.

    Returns dict with keys:
        model: NGramModel
        per_order_ppl: {n: ppl}
        lambdas: list
        interp_test_ppl: float
    """
    model = NGramModel.from_sentences(
        train_sentences, max_n=max_n, vocab_tokens=vocab_tokens
    )
    per_order = {}
    for n in range(1, max_n + 1):
        ppl = model.perplexity(val_sentences, max_n=n)
        per_order[n] = ppl
    lambdas = model.tune_lambdas(val_sentences, max_n=max_n, step=interp_step)
    test_ppl = model.perplexity(test_sentences, max_n=max_n, lambdas=lambdas)
    return {
        "model": model,
        "per_order_ppl": per_order,
        "lambdas": lambdas,
        "interp_test_ppl": test_ppl,
    }

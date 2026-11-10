import re

import numpy as np
from tqdm import tqdm

from src.ngram_engine.nge_utils import apply_bpe, load_ngrams_and_contexts
from src.tokenizer import bpe_utils


class NgramEngine:
    def __init__(self, n, k, advanced=False):
        self.n = n
        self.k = k
        self.adv_suffix = "_adv" if advanced else ""
        self.vocab = (
            open(f"data/bpe_outputs/vocab_with_k{k}{self.adv_suffix}.txt", "r")
            .read()
            .splitlines()
        )
        # </s> is a possible next token (it ends a sentence), so it belongs to the
        # candidates and to V. <s> is never predicted, so it is left out.
        self.vocab.append("</s>")
        self.vocab_size = len(self.vocab)
        self.merge_rules = (
            open(f"data/bpe_outputs/merges_k{k}{self.adv_suffix}.txt", "r")
            .read()
            .splitlines()
        )
        self.ngrams, self.contexts = load_ngrams_and_contexts(n, k, self.adv_suffix)
        self.SENT_SPLIT_PATTERN = re.compile(
            r"(?<!\.)[.?!]+(?!\.)\s+|(?<!\.)[.?!]+(?!\.)$"
        )

    def load_vocab(self, n: int, k: int, suffix: str):
        vocab = (
            open(f"data/bpe_outputs/vocab_with_k{k}{suffix}.txt", "r")
            .read()
            .splitlines()
        )
        return vocab

    def merge_tokens_ranked(self, tokens, merges):
        """
        Merge tokens based on a ranked list of merges.

        Args:
            tokens (list[str]): Current token sequence (typically chars + '_').
            merges (list[str]): Ordered merges (earlier means higher priority).
        Returns:
            list[str]: Tokens after applying all possible merges.
        """
        # Build quick lookup: token -> rank
        ranked = {m: r for r, m in enumerate(merges) if len(m) > 1}
        if not ranked:
            return tokens
        max_len = max(len(m) for m in ranked)

        while True:
            best = None  # (rank, start_index, span_len, merged_token)
            tlen = len(tokens)
            # Brute-force scan of all spans up to max_len
            for i in range(tlen):
                # Remaining length check
                remaining = tlen - i
                if remaining < 2:
                    break
                limit = min(max_len, remaining)
                for span_len in range(2, limit + 1):
                    candidate = "".join(tokens[i : i + span_len])
                    if candidate in ranked:
                        cand_rank = ranked[candidate]
                        if best is None or cand_rank < best[0]:
                            best = (cand_rank, i, span_len, candidate)
            if best is None:
                break
            _, i, span_len, merged = best
            tokens = tokens[:i] + [merged] + tokens[i + span_len :]
        return tokens

    def apply_bpe(self, corpus, merges, track_progress=False):
        """
        Apply Byte Pair Encoding (BPE) to the given corpus using the provided merges.
        Ellipses ("...") are preserved inside sentences (not treated as boundary).

        Args:
            corpus (str): The input text corpus.
            merges (list): The BPE merges to apply.
            track_progress (bool): Whether to show a progress bar.

        Returns:
            list: A list of processed sentences, each represented as a list of tokens.
        """
        # Regex split (same as above) to ensure consistent sentence boundaries
        sentences = re.split(self.SENT_SPLIT_PATTERN, corpus.strip())
        processed_sentences = []
        if track_progress:
            pbar = tqdm(total=len(sentences), desc="Applying BPE to sentences")
        for sentence in sentences:
            if not sentence.strip():
                if track_progress:
                    pbar.update(1)
                continue
            sentence = bpe_utils.normalize_text(sentence)
            sentence = list(sentence.replace(" ", "_"))
            sentence = self.merge_tokens_ranked(sentence, merges)
            sentence.insert(0, "<s>")  # Append start of sentence token
            sentence.append("</s>")  # Append end of sentence token
            processed_sentences.append(sentence)
            if track_progress:
                pbar.update(1)

        return processed_sentences

    def get_unigram_probs(self):
        # Extract unigrams from existing ngrams data (should always exist since ngrams contains 1 to n)
        vocab = self.vocab

        # Get all ngrams of length 1 from existing data
        unigrams = {key: value for key, value in self.ngrams.items() if len(key) == 1}
        total_count = sum(unigrams.values())

        # Add-one smoothing for unigrams
        # Convert tuple keys to single tokens
        unigram_probs = {
            key[0]: (count + 1) / (total_count + len(vocab))
            for key, count in unigrams.items()
        }
        return unigram_probs

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

    def get_word_from_context_with_backoff(self, context, n):
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
                ngram = context + (token,)
                next_word_probs = self.get_ngram_probs(ngram, n) * backoff_weight
                possible_next_words[token] = next_word_probs
            # if no possible next word probability is above the threshold, backoff to n-1 gram
            if not possible_next_words:
                n -= 1
                context = context[-(n - 1) :]
        # draw a word based on the probabilities
        if possible_next_words:
            words = list(possible_next_words.keys())  # Remove the [0] indexing
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
            return (
                np.random.choice(words, p=probs),
                words,
                probs,
            )  # Return the chosen word along with the words and their probabilities
        return None

    def get_word_from_context_with_interpolation(
        self,
        lambdas,
        context,
        n,
        k=1000,
        suffix="_adv",
    ):
        all_probs = []
        # Iterate through the vocab
        # For each possible token, calculate the probability of that token for all n
        for token in self.vocab:
            # Get the probability of the token given the context for all n
            probs = []
            for i in range(1, n + 1):
                # Get current context
                context = tuple(context)
                current_context = context[-(i - 1) :] if i > 1 else ()
                current_ngram = current_context + (token,)

                # Current ngram already contains n occurrences, so we can just get the probability of the token given the context
                occurences = self.ngrams.get(current_ngram, 0)

                context_count = self.contexts.get(current_context, 0)
                if current_context == ():
                    # <s> is counted as a unigram but is never predicted, so leave
                    # it out of the unigram total (otherwise the level sums to < 1)
                    context_count -= self.ngrams.get(("<s>",), 0)
                probs.append((occurences + 1) / (context_count + len(self.vocab)))

            all_probs.append(probs)
        processed_probs = np.array(all_probs) @ lambdas
        # Get index of the max probability
        max_index = np.argmax(processed_probs)

        return (
            self.vocab[max_index],
            self.vocab,
            processed_probs,
        )  # Return the chosen word along with the words and their probabilities

    def generate_sentence(self, start_context, lambdas=None, max_length=20, sample=True):
        """
        Generates a sentence starting with a given context, using the
        interpolated n-gram distribution.
        Args:
            start_context: list of tokens to start from, e.g. ["<s>", "the_"]
            lambdas: interpolation weights for orders 1..n (sum to 1).
                Defaults to equal weights.
            max_length: maximum number of tokens to add
            sample: draw the next token from the distribution (True)
                or always take the most likely one (False)
        """
        n = self.n
        if lambdas is None:
            lambdas = np.full(n, 1 / n)
        sentence = list(start_context)

        for _ in range(max_length):
            context = tuple(sentence[-(n - 1) :]) if n > 1 else ()
            best_word, words, probs = self.get_word_from_context_with_interpolation(
                lambdas, context, n
            )
            if sample:
                next_word = np.random.choice(words, p=probs / probs.sum())
            else:
                next_word = best_word

            if next_word == "</s>":
                break
            sentence.append(next_word)

        return sentence

    def calculate_perplexity(self):
        # load validation files
        text = open("data/corpora/Shakespeare_clean_valid.txt", "r").read()
        sentences = apply_bpe(text, self.merge_rules, track_progress=True)
        total_log_prob = 0.0
        total_tokens = 0
        pbar = tqdm(total=len(sentences) - 1, desc="Calculating Perplexity")

        for sentence in sentences:
            sentence_length = len(sentence)
            # Extract n-grams from the sentence
            for i in range(sentence_length - self.n + 1):
                ngram = tuple(sentence[i : i + self.n])
                context = ngram[:-1]
                target = ngram[-1]

                # Get probability distribution using backoff
                _, predictions, probs = self.get_probs_from_context_with_backoff(
                    context, self.n
                )

                # Find probability of target token
                if predictions and target in predictions:
                    target_idx = predictions.index(target)
                    target_prob = probs[target_idx]
                else:
                    # If target not found, use a very small probability (smoothing fallback)
                    target_prob = 1e-10

                total_log_prob += np.log(target_prob)
                total_tokens += 1
            pbar.update(1)

        # Calculate perplexity: exp(-1/N * sum(log p(w_i)))
        perplexity = (
            np.exp(-total_log_prob / total_tokens) if total_tokens > 0 else float("inf")
        )

        print(f"Validation perplexity: {perplexity:.2f}")
        print(f"Total tokens evaluated: {total_tokens}")

        return perplexity

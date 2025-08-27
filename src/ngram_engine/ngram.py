import json
import re
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from src.ngram_engine.nge_utils import apply_bpe
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
        self.vocab_size = len(self.vocab)
        self.merge_rules = (
            open(f"data/bpe_outputs/merges_k{k}{self.adv_suffix}.txt", "r")
            .read()
            .splitlines()
        )
        self.ngrams, self.contexts = self.load_ngrams_and_contexts(
            n, k, self.adv_suffix
        )
        self.SENT_SPLIT_PATTERN = re.compile(
            r"(?<!\.)[.?!]+(?!\.)\s+|(?<!\.)[.?!]+(?!\.)$"
        )

    def load_ngrams_and_contexts(self, n, k, suffix=""):
        """
        Load n-grams and contexts from cached JSON files.
        Args:
            n (int): The 'n' of the n-gram model.
            k (int): The 'k' used in BPE vocabulary size.
        Returns:
            tuple: (ngrams, contexts) where both are defaultdict(int)
        """
        ngram_file = f"data/ngram_outputs/ngrams_n{n}_k{k}{suffix}.json"
        context_file = f"data/ngram_outputs/contexts_n{n}_k{k}{suffix}.json"
        with open(ngram_file, "r") as f:
            ngrams_dict = json.load(f)
            # Convert string keys back to tuples
            ngrams = defaultdict(int)
            for key, value in ngrams_dict.items():
                ngrams[tuple(eval(key))] = value

        with open(context_file, "r") as f:
            contexts_dict = json.load(f)
            # Convert string keys back to tuples
            contexts = defaultdict(int)
            for key, value in contexts_dict.items():
                contexts[tuple(eval(key))] = value
        return ngrams, contexts

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

    def create_ngrams(sentences, n, track_progress=False):
        """
        Create n-grams from the given sentences.
        Uses window size from n down to 1 for backoff.
        Args:
            sentences (list of list of str): List of sentences, where each sentence is a list of tokens.
            n (int): The 'n' in n-grams, indicating the size of the n-grams to create.
            track_progress (bool): Whether to display a progress bar.
        Returns:
            tuple: A tuple containing two defaultdicts:
        """
        ngrams = defaultdict(int)
        contexts = defaultdict(int)
        if track_progress:
            pbar = tqdm(total=len(sentences) - 1, desc="Creating ngrams")
        for sentence in sentences:
            window_size = n  # start with n and decrease to 1 for backoff
            sentence_length = len(sentence)
            while window_size > 0:
                for i in range(sentence_length - window_size + 1):
                    ngram = tuple(sentence[i : i + window_size])
                    context = tuple(sentence[i : i + window_size - 1])
                    ngrams[ngram] += 1
                    contexts[context] += 1
                window_size -= 1
            if track_progress:
                pbar.update(1)

        return ngrams, contexts

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
        ngrams, contexts = self.ngrams.copy(), self.contexts.copy()

        context = ngram[:-1]
        ngram_count = ngrams.get(ngram, 0)
        context_count = contexts.get(context, 0)

        # Add-one smoothing
        return (ngram_count + 1) / (context_count + self.vocab_size)

    def get_word_from_context(self, context, n, k, suffix=""):
        # load ngrams and contexts from file

        # Get all possible next words
        possible_next_words = {}
        while len(possible_next_words) == 0 and n > 1:
            backoff_weight = pow(0.4, self.n - n)  # backoff weight
            # if context is empty, use unigram
            if len(context) == 0:
                unigrams_and_probs = self.get_unigram_probs(k, suffix)
                for tok, p in unigrams_and_probs.items():
                    possible_next_words[tok] = p * backoff_weight
            # find all ngrams that match the context
            for ngram in self.ngrams:
                if len(ngram) == len(context) + 1 and ngram[:-1] == context:
                    possible_next_word = ngram[-1]
                    next_word_probs = self.get_ngram_probs(ngram, n) * backoff_weight
                    possible_next_words[possible_next_word] = next_word_probs
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
                unigram_probs = self.get_unigram_probs(k, suffix)
                if not unigram_probs:
                    return "</s>"  # Should not happen
                # Get the unigram with the highest probability
                best_unigram = max(unigram_probs, key=unigram_probs.get)
                return best_unigram  # Remove the [0] indexing
            probs = [p / total_prob for p in probs]
            return np.random.choice(words, p=probs)
        return None

    def generate_sentence(self, start_context, max_length=20):
        """
        Generates a sentence starting with a given context.
        """
        sentence = list(start_context)
        n = self.n
        k = self.k
        suffix = self.adv_suffix

        for _ in range(max_length):
            context = tuple(sentence[-(n - 1) :])
            next_word = self.get_word_from_context(context, n, k, suffix)

            if next_word == "</s>" or next_word is None:
                break
            sentence.append(next_word)

        return sentence

    def get_probs_from_context(self, context, n, k, suffix=""):
        # load ngrams and contexts from file

        # Get all possible next words
        possible_next_words = {}
        while len(possible_next_words) == 0 and n > 1:
            backoff_weight = pow(0.4, self.n - n)  # backoff weight
            # if context is empty, use unigram
            if len(context) == 0:
                unigrams_and_probs = self.get_unigram_probs(k, suffix)
                for tok, p in unigrams_and_probs.items():
                    possible_next_words[tok] = p * backoff_weight
            # find all ngrams that match the context
            for ngram in self.ngrams:
                if len(ngram) == len(context) + 1 and ngram[:-1] == context:
                    possible_next_word = ngram[-1]
                    next_word_probs = self.get_ngram_probs(ngram, n) * backoff_weight
                    possible_next_words[possible_next_word] = next_word_probs
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
                unigram_probs = self.get_unigram_probs(k, suffix)
                if not unigram_probs:
                    return "</s>"  # Should not happen
                # Get the unigram with the highest probability
                best_unigram = max(unigram_probs, key=unigram_probs.get)
                return best_unigram  # Remove the [0] indexing
            probs = [p / total_prob for p in probs]
            return words, probs
        return None, 0

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
                predictions, probs = self.get_probs_from_context(
                    context, self.n, self.k, self.adv_suffix
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

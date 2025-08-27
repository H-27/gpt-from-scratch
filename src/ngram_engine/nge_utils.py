import json
import math
import re
from collections import defaultdict

from tqdm import tqdm

from src.tokenizer import bpe_utils

# Pattern: sentence end punctuation (., ?, !) not part of an ellipsis (triple or more dots)
SENT_SPLIT_PATTERN = re.compile(r"(?<!\.)[.?!]+(?!\.)\s+|(?<!\.)[.?!]+(?!\.)$")


def load_ngrams_and_contexts(n, k, suffix=""):
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


def load_vocab(n: int, k: int, suffix: str):
    vocab = open(f"data/bpe_outputs/vocab_k{k}{suffix}.txt", "r").read().splitlines()
    return vocab


def merge_tokens_ranked(tokens, merges):
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


def apply_bpe(corpus, merges, track_progress=False):
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
    sentences = re.split(SENT_SPLIT_PATTERN, corpus.strip())
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
        sentence = merge_tokens_ranked(sentence, merges)
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


def get_ngram_prob(ngram, n, k, suffix=""):
    # load ngrams and contexts from file
    ngrams, contexts = load_ngrams_and_contexts(n, k, suffix)
    vocab = load_vocab(n, k, suffix)

    context = ngram[:-1]
    ngram_count = ngrams.get(ngram, 0)
    context_count = contexts.get(context, 0)

    # Add-one smoothing
    return (ngram_count + 1) / (context_count + len(vocab))


def get_word_from_context(context, n, k, suffix=""):
    # load ngrams and contexts from file
    ngrams, contexts = load_ngrams_and_contexts(n, k, suffix)
    vocab = load_vocab(n, k, suffix)

    # Get all possible next words
    # possible next words as defaultdict with word as key and probability as value
    possible_next_words = {}

    for ngram in ngrams:
        if len(ngram) == len(context) + 1 and ngram[:-1] == context:
            possible_next_word = ngram[-1]
            next_word_prob = get_ngram_prob(ngram, n, k, suffix)
            possible_next_words[possible_next_word] = next_word_prob

    if not possible_next_words:
        return None, 0


def get_unigram_probs(self, k, suffix=""):
    # load ngrams and contexts from file
    ngrams, contexts = self.load_ngrams_and_contexts(1, k, suffix)
    vocab = self.load_vocab(1, k, suffix)

    # only get ngrams of length 1
    unigrams = {key: value for key, value in ngrams.items() if len(key) == 1}
    total_count = sum(unigrams.values())
    # Add-one smoothing for unigrams
    unigram_probs = {
        key: (count + 1) / (total_count + len(vocab)) for key, count in unigrams.items()
    }
    return unigram_probs


def calculate_ngram_probability(ngram, n, k, suffix=""):
    # if unigram
    if n == 1:
        return get_unigram_probs(ngram, k, suffix)
    # load ngrams and contexts from file
    ngrams, contexts = load_ngrams_and_contexts(n, k, suffix)
    vocab = load_vocab(n, k, suffix)

    context = ngram[:-1]
    ngram_count = ngrams.get(ngram, 0)
    context_count = contexts.get(context, 0)

    # Add-one smoothing
    return (ngram_count + 1) / (context_count + len(vocab))


def calculate_perplexity(sentences, n, ngrams, contexts, vocab_size):
    """
    Calculates the perplexity of a model on a given set of sentences.

    Args:
        sentences (list of list of str): The test/validation sentences.
        n (int): The 'n' of the n-gram model.
        ngrams (defaultdict): A dictionary with n-gram counts from training.
        contexts (defaultdict): A dictionary with context counts from training.
        vocab_size (int): The size of the vocabulary from training.

    Returns:
        float: The perplexity.
    """
    total_log_prob = 0
    total_tokens = 0

    for sentence in sentences:
        # We evaluate on n-grams, so we need to count them
        total_tokens += len(sentence) - (n - 1)
        for i in range(len(sentence) - n + 1):
            ngram = tuple(sentence[i : i + n])
            prob = calculate_ngram_probability(ngram, ngrams, contexts, vocab_size)
            total_log_prob += -math.log(prob)

    if total_tokens == 0:
        return float("inf")  # Avoid division by zero

    avg_log_prob = total_log_prob / total_tokens
    perplexity = math.exp(avg_log_prob)
    return perplexity


def generate_next_word(context, ngrams, contexts, vocab_size, n):
    """
    Generates the next word based on the given context using argmax.
    Implements a backoff strategy if the context is not found.

    Args:
        context (tuple): The current context (n-1 gram).
        ngrams (defaultdict): A dictionary with n-gram counts.
        contexts (defaultdict): A dictionary with context counts.
        vocab_size (int): The size of the vocabulary.
        n (int): The 'n' of the n-gram model.

    Returns:
        str: The predicted next word.
    """
    possible_next_words = []
    for ngram in ngrams:
        if len(ngram) == len(context) + 1 and ngram[:-1] == context:
            possible_next_words.append(ngram[-1])

    if not possible_next_words:
        # Backoff to a shorter context
        if len(context) > 1:
            return generate_next_word(context[1:], ngrams, contexts, vocab_size, n)
        else:
            # Backoff to unigrams - find the most frequent unigram
            unigrams = {k: v for k, v in ngrams.items() if len(k) == 1}
            if not unigrams:
                return "</s>"  # Should not happen if vocab is not empty
            # find the most likely unigram that is not a start token
            best_unigram = max(
                (ug for ug in unigrams if ug != ("<s>",)),
                key=unigrams.get,
                default=None,
            )
            return best_unigram[0] if best_unigram else "</s>"

    best_word = None
    max_prob = -1

    for word in possible_next_words:
        ngram = context + (word,)
        prob = calculate_ngram_probability(ngram, ngrams, contexts, vocab_size)
        if prob > max_prob:
            max_prob = prob
            best_word = word

    return best_word


def generate_sentence(start_context, n, ngrams, contexts, vocab_size, max_length=20):
    """
    Generates a sentence starting with a given context.

    Args:
        start_context (list): A list of words to start the sentence.
        n (int): The 'n' of the n-gram model.
        ngrams (defaultdict): A dictionary with n-gram counts.
        contexts (defaultdict): A dictionary with context counts.
        vocab_size (int): The size of the vocabulary.
        max_length (int): The maximum length of the generated sentence.

    Returns:
        list: The generated sentence as a list of tokens.
    """
    sentence = list(start_context)
    # Make sure context is n-1
    current_context = (
        tuple(sentence[-(n - 1) :]) if len(sentence) >= n - 1 else tuple(sentence)
    )

    for _ in range(max_length):
        next_word = generate_next_word(current_context, ngrams, contexts, vocab_size, n)
        if next_word == "</s>" or not next_word:
            break
        sentence.append(next_word)
        current_context = tuple(sentence[-(n - 1) :])

    return sentence

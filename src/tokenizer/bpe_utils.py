import re  # added for digit mapping
import string
from collections import defaultdict

from tqdm import tqdm


def split_text(text):
    text_len = len(text)
    part_size = text_len // 100
    train_size = (text_len - part_size) // 2
    train_set_1 = text[:train_size]
    test_set = text[train_size : train_size + part_size]
    train_set_2 = text[train_size + part_size :]
    train_set = train_set_1 + train_set_2
    return train_set, test_set


def normalize_text(text, remove_punctuation=True):
    text = text.lower()  # Convert to lowercase
    text = " ".join(text.split())  # Remove extra spaces
    text = text.replace("\n", " ")  # Replace newlines with spaces
    if remove_punctuation:
        text = text.translate(
            str.maketrans("", "", string.punctuation)
        )  # Remove punctuation
    return text


def advanced_normalize(
    text: str,
    keep_sentence_final: bool = True,
    keep_apostrophes: bool = True,
    map_digits: bool = True,
) -> str:
    """Advanced normalization applying three optional strategies:
    1. Selective punctuation retention (keep .?! and/or apostrophes) while removing others.
    2. Quote / ellipsis normalization: fancy quotes -> ' or ", ellipsis … -> three dots.
    3. Digit mapping: map all digits 0-9 to 0 to reduce sparsity.

    Args:
        text: Raw input string.
        keep_sentence_final: Keep . ? ! if True.
        keep_apostrophes: Keep apostrophes (') if True.
        map_digits: Replace every digit with 0 if True.
    Returns:
        Normalized string.
    """
    # Normalize whitespace early (preserve baseline behavior order)
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    # Lowercase (reuse baseline assumption)
    text = text.lower()

    # 2. Quote / ellipsis normalization before punctuation stripping
    replacements = {
        "“": '"',
        "”": '"',
        "„": '"',
        "′": "'",
        "’": "'",
        "‘": "'",
        "‛": "'",
        "…": "...",
    }
    text = "".join(replacements.get(ch, ch) for ch in text)

    # 1. Selective punctuation removal
    # Build punctuation set to remove
    allowed = set()
    if keep_sentence_final:
        allowed.update([".", "!", "?"])
    if keep_apostrophes:
        allowed.add("'")
    # Remove all other ASCII punctuation
    remove_chars = "".join(ch for ch in string.punctuation if ch not in allowed)
    if remove_chars:
        text = text.translate(str.maketrans("", "", remove_chars))

    # 3. Digit mapping
    if map_digits:
        text = re.sub(r"\d", "0", text)

    # Collapse whitespace again in case removals created doubles
    text = " ".join(text.split())
    return text


def update_vocab(vocab, addition):
    vocab.append(addition)
    return vocab


def prep_text(text):
    words = text.split()
    new_text = []
    for word in words:
        word = word + "_"
        new_text.append(word)

    new_text = " ".join(new_text)
    new_text = list(new_text)
    return new_text


def get_max_pair(text, track_progress=True):
    pairs = defaultdict(int)
    if track_progress:
        pbar = tqdm(total=len(text) - 1, desc="Getting max pairs")
    for i in range(len(text) - 1):
        pair = text[i : i + 2]
        pair = "".join(pair)
        if pair in pairs:
            pairs[pair] += 1
        else:
            pairs[pair] = 1
        if track_progress:
            pbar.update(1)

    # Find the most frequent pair
    most_frequent_pair = max(pairs, key=pairs.get)
    most_frequent_count = pairs[most_frequent_pair]
    return most_frequent_pair, most_frequent_count


def update_text(text, substitute_pair, track_progress=False):
    i = 0
    new_text = []
    if track_progress:
        pbar = tqdm(total=len(text) - 1, desc="Updating text")
    while i < len(text):
        if i == len(text) - 1:
            new_text.append(text[i])
            break
        pair = text[i] + text[i + 1]
        if pair == substitute_pair:
            new_text.append(substitute_pair)
            i += 2  # Skip the next token as it was merged
            # Do not increment i, check for overlapping pairs
        else:
            new_text.append(text[i])
            i += 1
        if track_progress:
            pbar.update(1)
    # if new text is empty return the original text
    if new_text == []:
        return text
    elif new_text == text:
        return text
    else:
        return new_text


def test_vocab(text, vocab=None, k=2000):
    """Test the vocabulary against a new text."""
    # load vocab
    if vocab is None:
        vocab = open(f"data/bpe_outputs/vocab_with_k{k}.txt", "r").read()
    print(f"Testing with vocabulary of size {len(vocab.splitlines())}")
    # prepare text
    text = normalize_text(text)
    words = text.split()
    tokens = 0
    unknown_tokens = []

    for word in words:
        word = word + "_"
        i = 0
        while i < len(word):
            found_token = False
            # Try to find longest matching token
            for j in range(len(word), i, -1):
                if word[i:j] in vocab:
                    tokens += 1
                    i = j
                    found_token = True
                    break
                else:
                    tokens += 1
                    unknown_tokens.append(word[i:j])
                    i = j
            if not found_token:
                unknown_tokens.append(word[i])
                i += 1
                tokens += 1

    coverage = (tokens - len(unknown_tokens)) / tokens * 100

    return coverage, unknown_tokens


def get_token_counts(vocab, corpus):
    counted_vocab = defaultdict(int)
    for token in vocab.splitlines():
        counted_vocab[token] = corpus.count(token)
    return counted_vocab


def reprocess_corpus(corpus, vocab):
    """
    Reprocess the corpus to create a vocabulary of size k.
    Args:
        corpus (str): The input text.
        k (int): The desired vocabulary size.
    Returns:
        str: The processed text with the vocabulary of size k.
    """
    corpus = normalize_text(corpus)
    corpus = list(corpus.replace(" ", "_"))
    for token in vocab.splitlines():
        if len(token) < 2:
            continue
        else:
            update_text(corpus, token, track_progress=False)
    pass

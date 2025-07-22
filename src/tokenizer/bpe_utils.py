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


def normalize_text(text):
    text = text.lower()
    text = " ".join(text.split())
    text = text.replace("\n", " ")
    return text


def create_initial_vocab(text):
    """Create an initial vocabulary from the text."""
    vocab_dict = defaultdict(int)
    vocab = list(string.ascii_uppercase) + list(string.ascii_lowercase) + [" "]
    vocab = list(set(text))
    return vocab


def update_vocab(vocab, text, addition):
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


def update_text(text, substitute_pair, track_progress=True):
    i = 0
    new_text = []
    if track_progress:
        pbar = tqdm(total=len(text) - 1, desc="Updating text")
    while i < len(text) - 1:
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
    return text

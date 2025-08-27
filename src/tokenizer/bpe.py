import string

from tqdm import tqdm

from .bpe_utils import (
    advanced_normalize,
    get_max_pair,
    normalize_text,
    split_text,
    update_text,
)


def perform_bpe(
    text="data/corpora/shakespeare.txt",
    k=2000,
    normalization=None,
    track_progress=False,
    save_to=None,
):
    k_start = k
    if save_to == None:
        save_to = f"data/bpe_outputs/vocab_with_k{k_start}.txt"
    # Load and normalize the text
    text = open(text, "r").read()
    if normalization == "advanced":
        text = advanced_normalize(text)
    else:
        text = normalize_text(text)

    # Split the text into training and test sets
    text = list(text.replace(" ", "_"))

    # Create the initial vocabulary
    vocab = list(string.ascii_lowercase) + ["_"]

    pbar = tqdm(total=k, desc="Merging pairs")

    while k > 0:
        # Get the most frequent pair in the text
        if track_progress:
            print("Getting most frequent pair...")
        most_frequent_pair, count = get_max_pair(text, track_progress)

        # If no pairs found, break
        if count < 2:
            break

        # Replace the most frequent pair in the text
        if track_progress:
            print("Updating text with most frequent pair...")
        text = update_text(text, most_frequent_pair, track_progress)

        # Add the new token to the vocabulary
        vocab.append(most_frequent_pair)

        k -= 1
        pbar.update(1)

    with open(save_to, "w", encoding="utf-8") as f:
        for token in vocab:
            f.write(token + "\n")
    # Derive merges (Option B: tokens length > 1)
    merges = [t for t in vocab if len(t) > 1]
    merges_path = save_to.replace("vocab_with_k", "merges_k")
    with open(merges_path, "w", encoding="utf-8") as mf:
        for m in merges:
            mf.write(m + "\n")
    print(f"Vocabulary saved to {save_to} (merges derived -> {merges_path})")
    return save_to


if __name__ == "__main__":
    track_progress = False
    text = open("data/corpora/shakespeare.txt", "r").read()
    text = normalize_text(text)

    train, test = split_text(text)
    # train = train[:100000]
    text = list(train.replace(" ", "_"))
    vocab = list(string.ascii_lowercase) + ["_"]
    k = 1000
    k_start = k

    pbar = tqdm(total=k, desc="Merging pairs")

    while k > 0:
        # Get the most frequent pair in the text
        if track_progress:
            print("Getting most frequent pair...")
        most_frequent_pair, count = get_max_pair(text, track_progress)
        # If no pairs found, break
        if count < 2:
            break

        # Replace the most frequent pair in the text
        if track_progress:
            print("Updating text with most frequent pair...")
        text = update_text(text, most_frequent_pair, track_progress)
        # Add the new token to the vocabulary
        vocab.append(most_frequent_pair)

        k -= 1
        pbar.update(1)

    print("Vocabulary done, saving...")
    # Save the vocabulary to a file
    with open(f"data/bpe_outputs/vocab_with_k{k_start}.txt", "w") as f:
        for token in vocab:
            f.write(token + "\n")
    print("Vocabulary saved.")

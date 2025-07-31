from tqdm import tqdm

from .bpe_utils import (
    create_initial_vocab,
    get_max_pair,
    normalize_text,
    split_text,
    update_text,
)


def perform_bpe(
    text="data/shakespeare.txt", k=2000, track_progress=False, save_to=None
):
    k_start = k
    if save_to == None:
        save_to = f"data/{text}_vocab_with_k{k_start}.txt"
    # Load and normalize the text
    text = open(text, "r").read()
    text = normalize_text(text)

    # Split the text into training and test sets
    text = list(text.replace(" ", "_"))

    # Create the initial vocabulary
    vocab = create_initial_vocab(text)

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
    with open(f"data/vocab_with_k{k_start}.txt", "w") as f:
        for token in vocab:
            f.write(token + "\n")
    print("Vocabulary saved.")
    return save_to


if __name__ == "__main__":
    track_progress = False
    text = open("data/shakespeare.txt", "r").read()
    text = normalize_text(text)

    train, test = split_text(text)
    # train = train[:100000]
    text = list(train.replace(" ", "_"))
    vocab = create_initial_vocab(text)
    k = 2000
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
    with open(f"data/vocab_with_k{k_start}.txt", "w") as f:
        for token in vocab:
            f.write(token + "\n")
    print("Vocabulary saved.")

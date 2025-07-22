from bpe_utils import (
    create_initial_vocab,
    get_max_pair,
    normalize_text,
    split_text,
    update_text,
)
from tqdm import tqdm

track_progress = False
text = open("data/shakespeare.txt", "r").read()
text = normalize_text(text)


train, test = split_text(text)
# train = train[:100000]
text = list(train.replace(" ", "_"))
vocab = create_initial_vocab(text)
counted_vocab = {char: text.count(char) for char in vocab}
k = 500

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
counted_vocab = {char: text.count(char) for char in vocab}
# Print the final vocabulary
print("Final vocabulary:", vocab)
# Save the vocabulary to a file
with open(f"vocab_with_k{k}.txt", "w") as f:
    for token in vocab:
        f.write(token + "\n")
# Save with counts
with open(f"data/vocab_with_k{k}.txt", "w") as f:
    for token, count in counted_vocab.items():
        f.write(f"{token} {count}\n")

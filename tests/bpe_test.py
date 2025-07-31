from src.tokenizer.bpe import perform_bpe
from src.tokenizer.bpe_utils import split_text, test_vocab

if __name__ == "__main__":
    k = 1000
    version = "clean"  # Choose between "clean" or "dirty" or empty string "" for original text
    if version == "clean":
        train_text = "data/shakespeare_clean_train.txt"
        val_text = "data/shakespeare_clean_valid.txt"
    elif version == "dirty":
        train_text = "data/shakespeare_dirty_train.txt"
        val_text = "data/shakespeare_dirty_valid.txt"
    else:
        # Load and normalize the text
        text = open("data/shakespeare.txt", "r").read()
        train_text, val_text = split_text(text)

    print("Running BPE tokenizer...")
    vocab_location = perform_bpe(
        text=train_text, k=k, track_progress=False, save_to=None
    )

    print("Testing vocabulary...")
    coverage, unknown_tokens = test_vocab(val_text, vocab=vocab_location, k=k)
    print(f"Coverage: {coverage:.2f}%")

    # In Mr. Knox's Country
    text = open("data/In Mr. Knox's Country.txt", "r", encoding="utf-8").read()
    print("Testing vocabulary...")
    coverage, unknown_tokens = test_vocab(text, vocab=vocab_location, k=k)
    print(f"Coverage: {coverage:.2f}%")

    # A Room with a View
    text = open("data/A Room with a View.txt", "r", encoding="utf-8").read()
    print("Testing vocabulary...")
    coverage, unknown_tokens = test_vocab(text, vocab=vocab_location, k=k)
    print(f"Coverage: {coverage:.2f}%")

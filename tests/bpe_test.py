import os

from src.tokenizer.bpe import perform_bpe
from src.tokenizer.bpe_utils import split_text, test_vocab

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Matplotlib not available. Visualization will be skipped.")

# --- Added flag for advanced normalization run naming ---
ADVANCED = False  # Set True when using advanced normalization pipeline
ADV_SUFFIX = "_adv" if ADVANCED else ""


def visualize_coverage_results(results):
    """Create visualizations for BPE vocabulary coverage results."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available. Skipping visualization.")
        return

    k_values = list(results.keys())
    text_names = list(results[k_values[0]].keys())

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Coverage vs Vocabulary Size
    for text_name in text_names:
        coverages = [results[k][text_name]["coverage"] for k in k_values]
        ax1.plot(k_values, coverages, marker="o", linewidth=2, label=text_name)

    ax1.set_xlabel("Vocabulary Size (k)")
    ax1.set_ylabel("Coverage (%)")
    ax1.set_title("BPE Vocabulary Coverage vs Size")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(60, 100)

    # Plot 2: Unknown Tokens vs Vocabulary Size (log scale)
    for text_name in text_names:
        unknown_counts = [results[k][text_name]["unknown_count"] for k in k_values]
        ax2.plot(k_values, unknown_counts, marker="s", linewidth=2, label=text_name)

    ax2.set_xlabel("Vocabulary Size (k)")
    ax2.set_ylabel("Unknown Tokens (log scale)")
    ax2.set_title("Unknown Tokens vs Vocabulary Size")
    ax2.set_yscale("log")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "data/bpe_outputs/bpe_coverage_analysis.png", dpi=300, bbox_inches="tight"
    )
    plt.show()


def print_coverage_summary(results):
    """Print a formatted summary table of coverage results."""
    k_values = list(results.keys())
    text_names = list(results[k_values[0]].keys())

    print("\n" + "=" * 80)
    print("COVERAGE SUMMARY TABLE")
    print("=" * 80)
    print(
        f"{'Vocab Size':<12} {'Shakespeare':<15} {'Knox Country':<15} {'Room w/ View':<15}"
    )
    print("-" * 80)

    for k in k_values:
        row = f"{k:<12}"
        for text_name in text_names:
            coverage = results[k][text_name]["coverage"]
            row += f"{coverage:>10.1f}%    "
        print(row)


if __name__ == "__main__":
    # Test different vocabulary sizes
    k_values = [50, 500, 1000, 1250, 1500, 2000]
    version = "clean"  # Choose between "clean" or "dirty" or empty string "" for original text

    # Set up file paths based on version
    if version == "clean":
        train_text = "data/corpora/Shakespeare_clean_train.txt"
        val_text = "data/corpora/Shakespeare_clean_valid.txt"
    elif version == "dirty":
        train_text = "data/corpora/shakespeare_dirty_train.txt"
        val_text = "data/corpora/shakespeare_dirty_valid.txt"
    else:
        # Load and normalize the text
        text = open("data/corpora/shakespeare.txt", "r").read()
        train_text, val_text = split_text(text)

    # Print which version  and files are being used
    print(f"Using '{version}' version of Shakespeare text.")
    print(f"Training text file: {train_text}")
    print(f"Validation text file: {val_text}")
    print(f"Advanced flag: {ADVANCED} (vocab files will have suffix '{ADV_SUFFIX}')")

    # Test texts to evaluate vocabulary coverage
    test_texts = [
        ("Shakespeare Validation", val_text),
        ("In Mr. Knox's Country", "data/corpora/In Mr. Knox's Country.txt"),
        ("A Room with a View", "data/corpora/A Room with a View.txt"),
    ]

    print("=" * 60)
    print("BPE Vocabulary Coverage Test")
    print("=" * 60)

    # Store results for visualization
    coverage_results = {k: {} for k in k_values}

    for k in k_values:
        print(f"\n--- Testing with k={k} ---")

        vocab_filename = f"data/bpe_outputs/vocab_with_k{k}{ADV_SUFFIX}.txt"
        # Check if vocabulary exists, create if not
        if os.path.exists(vocab_filename):
            print(f"Using existing vocabulary file: {vocab_filename}")
            vocab_location = vocab_filename
        else:
            print(f"Running BPE tokenizer for k={k} (creating {vocab_filename})...")
            vocab_location = perform_bpe(
                text=train_text,
                k=k,
                track_progress=False,
                save_to=vocab_filename,
            )

        # Test vocabulary on different texts
        for text_name, text_path in test_texts:
            print(f"\nTesting {text_name}:")

            if text_name == "Shakespeare Validation":
                text_content = open(text_path, "r", encoding="utf-8").read()
            else:
                text_content = open(text_path, "r", encoding="utf-8").read()

            vocab_content = open(vocab_location, "r", encoding="utf-8").read()

            coverage, unknown_tokens = test_vocab(
                text_content, vocab=vocab_content, k=k
            )
            print(f"  Coverage: {coverage:.2f}%")
            if len(unknown_tokens) > 0:
                print(f"  Unknown tokens (first 10): {list(unknown_tokens)[:10]}")
                print(f"  Total unknown tokens: {len(unknown_tokens)}")

            coverage_results[k][text_name] = {
                "coverage": coverage,
                "unknown_count": len(unknown_tokens),
            }

    print_coverage_summary(coverage_results)
    visualize_coverage_results(coverage_results)

    print("\n" + "=" * 60)
    print("Test complete!")

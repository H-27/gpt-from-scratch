import json
import os
from collections import defaultdict

from src.ngram_engine.ngram import NgramEngine

from .nge_utils import apply_bpe, create_ngrams

if __name__ == "__main__":
    # define parameters for the NGramEngine
    n = 3
    k = 500
    advanced = True
    adv_suffix = "_adv" if advanced else ""
    # load vocab
    merge_rules = (
        open(f"data/bpe_outputs/merges_k{k}{adv_suffix}.txt", "r").read().splitlines()
    )
    text = open("data/corpora/Shakespeare_clean_train.txt", "r").read()
    # Create output directory if it doesn't exist
    os.makedirs("data/ngram_outputs", exist_ok=True)

    # load sentences, if no file found, create it
    try:
        with open(f"data/ngram_outputs/sentences_k{k}.txt", "r") as f:
            sentences = [line.strip().split() for line in f.readlines()]
    except FileNotFoundError:
        sentences = apply_bpe(text, merge_rules, track_progress=True)
        # save sentences to a file in data/ngram_outputs/sentences_k.txt
        with open(f"data/ngram_outputs/sentences_k{k}.txt", "w") as f:
            for sentence in sentences:
                f.write(" ".join(sentence) + "\n")

    # Load or create n-grams and contexts
    ngram_file = f"data/ngram_outputs/ngrams_n{n}_k{k}.json"
    context_file = f"data/ngram_outputs/contexts_n{n}_k{k}.json"

    try:
        print("Loading existing n-grams and contexts...")
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

        print(f"Loaded {len(ngrams)} n-grams and {len(contexts)} contexts from cache")

    except FileNotFoundError:
        print("Creating n-grams and contexts...")
        ngrams, contexts = create_ngrams(sentences, n)

        # Save n-grams and contexts
        print("Saving n-grams and contexts...")

        # Convert tuples to strings for JSON serialization
        ngrams_dict = {str(key): value for key, value in ngrams.items()}
        contexts_dict = {str(key): value for key, value in contexts.items()}

        with open(ngram_file, "w") as f:
            json.dump(ngrams_dict, f)

        with open(context_file, "w") as f:
            json.dump(contexts_dict, f)

        print(f"Saved {len(ngrams)} n-grams and {len(contexts)} contexts to cache")

    # Extrinsic test: print some n-grams and contexts
    print("\nSample n-grams:")
    engine = NgramEngine(n=n, k=k, advanced=advanced)
    sample_ngrams = list(engine.ngrams.items())[:10]
    for ngram, count in sample_ngrams:
        print(f"  {ngram}: {count}")
        sentence = engine.generate_sentence(ngram[:-1])
        sentence_str = (
            " ".join(sentence)
            .replace("<s>", "")
            .replace("</s>", ".")
            .replace(" ", "")
            .replace("▁", " ")
        )
        print(f"    Generated sentence: {sentence_str}")

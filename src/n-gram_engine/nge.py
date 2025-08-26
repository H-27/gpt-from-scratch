import json
import os
from collections import defaultdict

from .nge_utils import create_ngrams, split_into_sentences_and_normalize

if __name__ == "__main__":
    # define parameters for the NGramEngine
    n = 3
    k = 1250
    # load vocab
    vocab = open(f"data/vocab_with_k{k}.txt", "r").read().splitlines()
    text = open("data/Shakespeare_clean_train.txt", "r").read()
    # Create output directory if it doesn't exist
    os.makedirs("data/ngram_outputs", exist_ok=True)

    # load sentences, if no file found, create it
    try:
        with open(f"data/ngram_outputs/sentences_k{k}.txt", "r") as f:
            sentences = [line.strip().split() for line in f.readlines()]
    except FileNotFoundError:
        sentences = split_into_sentences_and_normalize(text, vocab, track_progress=True)
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

    print(f"Number of {n}-grams: {len(ngrams)}")
    print(f"Number of contexts: {len(contexts)}")
    # print some ngrams
    for i, (ngram, count) in enumerate(ngrams.items()):
        if i < 10:
            print(f"{ngram}: {count}")
        else:
            break
    # print some contexts
    for i, (context, count) in enumerate(contexts.items()):
        if i < 10:
            print(f"{context}: {count}")
        else:
            break

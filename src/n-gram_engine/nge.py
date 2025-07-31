def run_ngram_engine():
    # TODO: Implement the main function to run the NGramEngine
    raise NotImplementedError("NGramEngine main function is not implemented yet.")


def check_perplexity():
    # TODO: Implement the function to check perplexity
    raise NotImplementedError("Perplexity check function is not implemented yet.")


if __name__ == "__main__":
    # define parameters for the NGramEngine
    n = 3
    k = 2000
    # load vocab
    vocab = open(f"data/vocab_with_k{k}.txt", "r").read().splitlines()

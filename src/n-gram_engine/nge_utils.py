from collections import defaultdict

from tqdm import tqdm

from src.tokenizer import bpe_utils


def split_into_sentences_and_normalize(text, vocab, track_progress=False):
    """
    Splits the text into sentences and normalizes it.

    Args:
        text (str): The input text.

    Returns:
        str: The normalized text with sentences split.
    """
    # Split text into sentences
    sentences = text.split(". ")
    normalized_sentences = []
    if track_progress:
        pbar = tqdm(total=len(sentences) - 1, desc="Updating sentences")
    for sentence in sentences:
        # if sentencs is empty because of a ...
        if not sentence.strip():
            continue
        # Normalize each sentence
        normalized_sentence = sentence.strip()
        normalized_sentence = bpe_utils.normalize_text(normalized_sentence)
        normalized_sentence = list(normalized_sentence.replace(" ", "_"))
        normalized_sentence.insert(0, "<s>")  # Append start of sentence token
        normalized_sentence.append("</s>")  # Append end of sentence token
        for token in vocab:
            if len(token) < 2:
                continue
            else:
                # print(normalized_sentence)
                normalized_sentence = bpe_utils.update_text(
                    normalized_sentence, token, track_progress=False
                )
                # print(normalized_sentence)
        normalized_sentences.append(normalized_sentence)
        if track_progress:
            pbar.update(1)

    # Join sentences with a space
    return normalized_sentences


def create_ngrams(sentences, n, track_progress=False):
    """
    Create n-grams from the given sentences.
    Uses window size from n down to 1 for backoff.
    Args:
        sentences (list of list of str): List of sentences, where each sentence is a list of tokens.
        n (int): The 'n' in n-grams, indicating the size of the n-grams to create.
        track_progress (bool): Whether to display a progress bar.
    Returns:
        tuple: A tuple containing two defaultdicts:
    """
    ngrams = defaultdict(int)
    contexts = defaultdict(int)
    if track_progress:
        pbar = tqdm(total=len(sentences) - 1, desc="Creating ngrams")

    for sentence in sentences:
        window_size = n  # start with n and decrease to 1 for backoff
        sentence_length = len(sentence)
        while window_size > 0:
            for i in range(sentence_length - window_size + 1):
                ngram = tuple(sentence[i : i + window_size])
                context = tuple(sentence[i : i + window_size - 1])
                ngrams[ngram] += 1
                contexts[context] += 1
            window_size -= 1
        if track_progress:
            pbar.update(1)

    return ngrams, contexts

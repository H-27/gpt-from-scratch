import nltk
from nltk.corpus import shakespeare
import re
from collections import defaultdict, Counter
from tqdm import tqdm
import random
import pickle


def extract_doc_text(text):
    # Remove XML headers
    text = re.sub(r"<\?xml.*?\?>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\!--.*?-->", "", text, flags=re.DOTALL)
    # Remove all XML/HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()


def get_shakespeare_doc_text(doc_name):
    """
    Returns the raw text of a specific Shakespeare document.
    """
    nltk.download("shakespeare", quiet=True)
    if doc_name in shakespeare.fileids():
        # return shakespeare.raw(doc_name)  # extract_doc_text(shakespeare.raw(doc_name))
        return extract_doc_text(shakespeare.raw(doc_name))
    else:
        raise ValueError(f"Document '{doc_name}' not found in NLTK Shakespeare corpus.")


if __name__ == "__main__":
    docs = shakespeare.fileids()
    print("Available Shakespeare documents:", docs)

    corpus = ""
    for doc in docs:
        text = get_shakespeare_doc_text(doc)
        corpus += text

    with open("Shakespeare_clean_full.txt", "w") as text_file:
        text_file.write(corpus)

    print("full_size: ", len(corpus))

    test_size = len(corpus.split()) // 100
    valid_size = len(corpus.split()) // 100
    test_data = ""
    valid_data = ""
    random = random.Random()
    corpus = " ".join(corpus.split()[:])
    for i in range(10):
        random_index = random.randint(0, len(corpus.split()) - test_size)
        sub_test_text = " ".join(
            corpus.split()[random_index : random_index + test_size]
        )
        test_data += " " + sub_test_text
        corpus = corpus.replace(sub_test_text, "")

    for i in range(10):
        random_index = random.randint(0, len(corpus.split()) - valid_size)
        sub_valid_text = " ".join(
            corpus.split()[random_index : random_index + valid_size]
        )
        valid_data += " " + sub_valid_text
        corpus = corpus.replace(sub_valid_text, "")

    print("train_size: ", len(corpus))
    with open("Shakespeare_clean_train.txt", "w") as text_file:
        text_file.write(corpus)

    print("valid_size: ", len(valid_data))
    with open("Shakespeare_clean_valid.txt", "w") as text_file:
        text_file.write(valid_data)



    print("test_size: ", len(test_data))
    with open("Shakespeare_clean_test.txt", "w") as text_file:
        text_file.write(test_data)

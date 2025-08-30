import json
import os
from collections import defaultdict

from src.ngram_engine.ngram import NgramEngine
from src.tokenizer import bpe

from .nge_utils import apply_bpe, create_ngrams

if __name__ == "__main__":
    # Grid definitions (original single values replaced by lists)
    n_values = [1, 2, 3, 4, 5]
    k_values = [2000]
    adv_values = [True, False]

    os.makedirs("data/ngram_outputs", exist_ok=True)
    results_file = "data/ngram_outputs/perplexity_grid.jsonl"
    existing = set()
    results = []
    # Load existing results (do not duplicate)
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    key = (obj.get("n"), obj.get("k"), bool(obj.get("advanced")))
                    existing.add(key)
                    results.append(obj)
                except Exception:
                    pass

    first_demo_done = False  # keep original sample generation only once

    for advanced in adv_values:
        for k in k_values:
            for n in n_values:
                key = (n, k, advanced)
                if key in existing:
                    # Already computed
                    continue
                adv_suffix = "_adv" if advanced else ""
                # load vocab / merges
                merges_path = f"data/bpe_outputs/merges_k{k}{adv_suffix}.txt"
                if not os.path.exists(merges_path):
                    print(f"Warning: merges file {merges_path} not found, creating...")
                    bpe.perform_bpe(
                        text="data/corpora/Shakespeare_clean_train.txt",
                        k=k,
                        normalization=adv_suffix.replace("_adv", "advanced"),
                        track_progress=False,
                        save_to=f"data/bpe_outputs/vocab_with_k{k}{adv_suffix}.txt",
                    )

                merge_rules = open(merges_path, "r").read().splitlines()
                text = open("data/corpora/Shakespeare_clean_train.txt", "r").read()
                # sentences cache (include adv suffix to distinguish)
                sentences_path = f"data/ngram_outputs/sentences_k{k}{adv_suffix}.txt"
                try:
                    with open(sentences_path, "r") as f:
                        sentences = [line.strip().split() for line in f.readlines()]
                except FileNotFoundError:
                    sentences = apply_bpe(text, merge_rules, track_progress=False)
                    with open(sentences_path, "w") as f:
                        for sentence in sentences:
                            f.write(" ".join(sentence) + "\n")
                # ngram/context cache paths
                ngram_file = f"data/ngram_outputs/ngrams_n{n}_k{k}{adv_suffix}.json"
                context_file = f"data/ngram_outputs/contexts_n{n}_k{k}{adv_suffix}.json"
                try:
                    with open(ngram_file, "r") as f:
                        ngrams_dict = json.load(f)
                        ngrams = defaultdict(int)
                        for tkey, val in ngrams_dict.items():
                            ngrams[tuple(eval(tkey))] = val
                    with open(context_file, "r") as f:
                        contexts_dict = json.load(f)
                        contexts = defaultdict(int)
                        for tkey, val in contexts_dict.items():
                            contexts[tuple(eval(tkey))] = val
                except FileNotFoundError:
                    ngrams, contexts = create_ngrams(sentences, n)
                    ngrams_dict = {str(key2): value2 for key2, value2 in ngrams.items()}
                    contexts_dict = {
                        str(key2): value2 for key2, value2 in contexts.items()
                    }
                    with open(ngram_file, "w") as f:
                        json.dump(ngrams_dict, f)
                    with open(context_file, "w") as f:
                        json.dump(contexts_dict, f)
                # Engine
                engine = NgramEngine(n=n, k=k, advanced=advanced)
                # One-time demo (reuse original logic at first n=3 occurrence if available)
                if not first_demo_done and n == 3:
                    first_demo_done = True
                    print("\nSample n-grams:")
                    sample_ngrams = list(engine.ngrams.items())[:1]
                    my_text = "The world is so small"
                    my_text_bpe = engine.apply_bpe(my_text, merge_rules)[0]
                    d = -n
                    my_text_bpe = my_text_bpe[:-1]
                    sample_ngrams.append((tuple(my_text_bpe[d:]), 1))
                    for ng_ex, count in sample_ngrams:
                        print(f"  {ng_ex}: {count}")
                        sentence = engine.generate_sentence(ng_ex)
                        sentence_str = (
                            " ".join(sentence)
                            .replace("<s>", "")
                            .replace("</s>", ".")
                            .replace(" ", "")
                            .replace("_", " ")
                            .strip()
                        )
                        ng_disp = " ".join(ng_ex).replace("_", " ")
                        sentence_str = " ".join(sentence_str.split())
                        print(f" Generated sentence: {ng_disp} -> {sentence_str}")
                # Perplexity
                ppl = engine.calculate_perplexity()
                rec = {
                    "n": n,
                    "k": k,
                    "advanced": advanced,
                    "perplexity": round(float(ppl), 4),
                }
                with open(results_file, "a") as rf:
                    rf.write(json.dumps(rec) + "\n")
                existing.add(key)
                results.append(rec)
                print(
                    f"n={n} k={k} adv={int(advanced)} perplexity={rec['perplexity']} (saved)"
                )

    # Final summary
    print("\nSummary:")
    print("n | k | adv | perplexity")
    print("-" * 40)
    for r in sorted(results, key=lambda x: (x["k"], x["n"], x["advanced"])):
        print(f"{r['n']} | {r['k']} | {int(r['advanced'])} | {r['perplexity']}")

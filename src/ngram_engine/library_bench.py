import argparse
import json
import math
import os
import time
from typing import List

from nltk.lm import MLE, KneserNeyInterpolated, Laplace
from nltk.lm.preprocessing import padded_everygram_pipeline

from src.ngram_engine.nge_utils import apply_bpe  # reuse BPE

# This benchmark file uses external libraries (NLTK language modeling module)
# to build comparable n-gram language models (MLE / Laplace (add-one) / Kneser-Ney) and
# measure perplexity vs the custom implementation. Designed for performance &
# quality comparison, not for minimal dependencies.
#
# Example:
# uv run -m src.ngram_engine.library_bench --n-max 5 --algo kneserney
# uv run -m src.ngram_engine.library_bench --n-max 3 --algo laplace

DEFAULT_K_VALUES = [50, 500, 1000, 1250, 1500, 2000]


def load_sentences(k: int, adv_suffix: str, merges: List[str]):
    """Load (or build) BPE-tokenized training sentences for a given k."""
    os.makedirs("data/ngram_outputs", exist_ok=True)
    path = f"data/ngram_outputs/sentences_k{k}{adv_suffix}.txt"
    if os.path.exists(path):
        with open(path, "r") as f:
            return [line.strip().split() for line in f]
    # Build
    text = open("data/corpora/Shakespeare_clean_train.txt", "r").read()
    sentences = apply_bpe(text, merges, track_progress=True)
    with open(path, "w") as f:
        for s in sentences:
            f.write(" ".join(s) + "\n")
    return sentences


def load_validation_sentences(k: int, adv_suffix: str, merges: List[str]):
    path = f"data/ngram_outputs/valid_sentences_k{k}{adv_suffix}.txt"
    if os.path.exists(path):
        with open(path, "r") as f:
            return [line.strip().split() for line in f]
    text = open("data/corpora/Shakespeare_clean_valid.txt", "r").read()
    sentences = apply_bpe(text, merges, track_progress=True)
    with open(path, "w") as f:
        for s in sentences:
            f.write(" ".join(s) + "\n")
    return sentences


def iter_clean_sentences(sentences: List[List[str]]):
    """Yield sentences stripped of existing <s>/</s>; skip if empty after stripping."""
    for s in sentences:
        if s and s[0] == "<s>":
            s = s[1:]
        if s and s[-1] == "</s>":
            s = s[:-1]
        if not s:  # skip empty
            continue
        yield s


def build_model(algo: str, n: int, train_sentences: List[List[str]]):
    cleaned_list = list(iter_clean_sentences(train_sentences))
    if not cleaned_list:
        raise ValueError(
            "No non-empty training sentences after stripping <s>/<eos> markers"
        )
    train_ngrams, vocab = padded_everygram_pipeline(n, cleaned_list)
    if algo == "mle":
        model = MLE(n)
    elif algo == "laplace":  # add-one smoothing equivalent to your implementation
        model = Laplace(n)
    elif algo == "kneserney":
        if n == 1:
            # NLTK's KneserNeyInterpolated is undefined for unigram; fallback to Laplace(1)
            print(
                "[warn] Kneser-Ney (n=1) unsupported; falling back to Laplace unigram."
            )
            model = Laplace(n)
        else:
            model = KneserNeyInterpolated(n)
    else:
        raise ValueError(f"Unknown algo {algo}")
    model.fit(train_ngrams, vocab)
    return model


def model_perplexity(model, n: int, valid_sentences: List[List[str]]):
    from nltk.lm.preprocessing import everygrams, pad_both_ends

    cleaned_valid = list(iter_clean_sentences(valid_sentences))
    if not cleaned_valid:
        return float("inf")
    log_prob_sum = 0.0
    token_count = 0
    for sent in cleaned_valid:
        padded = list(pad_both_ends(sent, n))
        for ngram in everygrams(padded, max_len=n):
            if len(ngram) == n:
                context = ngram[:-1]
                word = ngram[-1]
                prob = model.score(word, context)
                if prob <= 0.0:
                    prob = 1e-12
                log_prob_sum += math.log(prob)
                token_count += 1
    if token_count == 0:
        return float("inf")
    ppl = math.exp(-log_prob_sum / token_count)
    return ppl


def run_grid(n_max: int, k_values, advanced: bool, algo: str, repetitions: int):
    adv_suffix = "_adv" if advanced else ""
    results = []
    for k in k_values:
        merges_file = f"data/bpe_outputs/merges_k{k}{adv_suffix}.txt"
        if not os.path.exists(merges_file):
            print(f"Skipping k={k} (missing merges {merges_file})")
            continue
        merges = open(merges_file, "r").read().splitlines()
        train_sentences = load_sentences(k, adv_suffix, merges)
        valid_sentences = load_validation_sentences(k, adv_suffix, merges)
        for n in range(1, n_max + 1):
            times = []
            perplexities = []
            for _ in range(repetitions):
                t0 = time.time()
                model = build_model(algo, n, train_sentences)
                t1 = time.time()
                ppl = model_perplexity(model, n, valid_sentences)
                t2 = time.time()
                times.append((t1 - t0, t2 - t1))  # (train_time, eval_time)
                perplexities.append(ppl)
            train_avg = sum(t[0] for t in times) / repetitions
            eval_avg = sum(t[1] for t in times) / repetitions
            ppl_avg = sum(perplexities) / repetitions
            entry = {
                "k": k,
                "n": n,
                "advanced": advanced,
                "algo": algo,
                "perplexity": round(ppl_avg, 3),
                "train_time_sec_avg": round(train_avg, 4),
                "eval_time_sec_avg": round(eval_avg, 4),
            }
            results.append(entry)
            print(
                f"n={n} k={k} algo={algo} adv={advanced} ppl={entry['perplexity']} train={train_avg:.3f}s eval={eval_avg:.3f}s"
            )
    return results


def main():
    ap = argparse.ArgumentParser(
        description="Benchmark external library n-gram models (NLTK)"
    )
    ap.add_argument("--n-max", type=int, default=5)
    ap.add_argument(
        "--k-values", type=str, default=",".join(str(k) for k in DEFAULT_K_VALUES)
    )
    ap.add_argument("--advanced", action="store_true")
    ap.add_argument(
        "--algo", type=str, default="mle", choices=["mle", "laplace", "kneserney"]
    )
    ap.add_argument("--repetitions", type=int, default=1)
    ap.add_argument("--out-json", type=str, default="")
    args = ap.parse_args()

    k_values = [int(x) for x in args.k_values.split(",") if x.strip()]
    results = run_grid(
        n_max=args.n_max,
        k_values=k_values,
        advanced=args.advanced,
        algo=args.algo,
        repetitions=args.repetitions,
    )

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {args.out_json}")

    # Print summary table
    headers = ["n", "k", "adv", "algo", "ppl", "train_s", "eval_s"]
    print("\nSummary:")
    print(" | ".join(headers))
    print("-" * 72)
    for r in results:
        row = [
            str(r["n"]),
            str(r["k"]),
            str(int(r["advanced"])),
            r["algo"],
            f"{r['perplexity']:.2f}",
            f"{r['train_time_sec_avg']:.3f}",
            f"{r['eval_time_sec_avg']:.3f}",
        ]
        print(" | ".join(row))


if __name__ == "__main__":
    main()

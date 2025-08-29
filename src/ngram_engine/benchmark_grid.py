import json
import os
import time
from argparse import ArgumentParser

from src.ngram_engine.ngram import NgramEngine

from .nge_utils import apply_bpe, create_ngrams

# Available k values based on existing merge/vocab files
DEFAULT_K_VALUES = [50, 250, 500, 750, 1000, 1250, 1500, 2000]


def ensure_sentences(k: int, merges, adv_suffix: str, force=False, track=False):
    """Load or build BPE-tokenized training sentences for given k (and adv)."""
    os.makedirs("data/ngram_outputs", exist_ok=True)
    sentences_path = f"data/ngram_outputs/sentences_k{k}{adv_suffix}.txt"
    if not force:
        try:
            with open(sentences_path, "r") as f:
                return [line.strip().split() for line in f]
        except FileNotFoundError:
            pass
    # Build
    text = open("data/corpora/Shakespeare_clean_train.txt", "r").read()
    sentences = apply_bpe(text, merges, track_progress=track)
    with open(sentences_path, "w") as f:
        for s in sentences:
            f.write(" ".join(s) + "\n")
    return sentences


def ensure_ngrams(n: int, k: int, adv_suffix: str, sentences):
    """Load or create n-grams / contexts JSON for (n,k)."""
    ngram_file = f"data/ngram_outputs/ngrams_n{n}_k{k}{adv_suffix}.json"
    context_file = f"data/ngram_outputs/contexts_n{n}_k{k}{adv_suffix}.json"
    if os.path.exists(ngram_file) and os.path.exists(context_file):
        return
    ngrams, contexts = create_ngrams(sentences, n, track_progress=True)
    ngrams_dict = {str(key): value for key, value in ngrams.items()}
    contexts_dict = {str(key): value for key, value in contexts.items()}
    with open(ngram_file, "w") as f:
        json.dump(ngrams_dict, f)
    with open(context_file, "w") as f:
        json.dump(contexts_dict, f)


def run_grid(n_max: int, k_values, advanced: bool, method: str, repetitions: int):
    adv_suffix = "_adv" if advanced else ""
    results = []
    for k in k_values:
        # Load merges for this k
        merges_path = f"data/bpe_outputs/merges_k{k}{adv_suffix}.txt"
        if not os.path.exists(merges_path):
            print(f"Skipping k={k}: merges file missing {merges_path}")
            continue
        merges = open(merges_path, "r").read().splitlines()
        # Sentences (reused across n for given k)
        sentences = ensure_sentences(k, merges, adv_suffix, force=False, track=False)
        for n in range(1, n_max + 1):
            ensure_ngrams(n, k, adv_suffix, sentences)
            engine = NgramEngine(n=n, k=k, advanced=advanced)
            valid_sentences = engine.get_validation_sentences(
                cached=True, progress=False
            )
            # Warm up precompute
            if method == "fast":
                engine._precompute_context_raw_probs()
            elif method == "ultra":
                engine._precompute_context_log_probs()
            # Timing
            times = []
            for _ in range(repetitions):
                t0 = time.time()
                if method == "fast":
                    ppl = engine.calculate_perplexity_fast(
                        sentences=valid_sentences,
                        use_cached_sentences=True,
                        progress=False,
                    )
                elif method == "ultra":
                    ppl = engine.calculate_perplexity_ultrafast(
                        sentences=valid_sentences,
                        use_cached_sentences=True,
                        progress=False,
                    )
                else:
                    ppl = engine.calculate_perplexity(
                        sentences=valid_sentences,
                        use_cached_sentences=True,
                        progress=False,
                    )
                t1 = time.time()
                times.append(t1 - t0)
            avg_time = sum(times) / len(times)
            results.append(
                {
                    "n": n,
                    "k": k,
                    "advanced": advanced,
                    "method": method,
                    "perplexity": ppl,
                    "time_sec_avg": round(avg_time, 4),
                }
            )
            print(
                f"n={n} k={k} adv={advanced} method={method} perplexity={ppl:.2f} time={avg_time:.3f}s"
            )
    return results


def main():
    parser = ArgumentParser(
        description="Benchmark n-gram perplexities across n and k grid"
    )
    parser.add_argument("--n-max", type=int, default=5, help="Maximum n (inclusive)")
    parser.add_argument(
        "--k-values",
        type=str,
        default=",".join(str(k) for k in DEFAULT_K_VALUES),
        help="Comma separated list of k values",
    )
    parser.add_argument(
        "--advanced", action="store_true", help="Use advanced (_adv) merges/vocab"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="fast",
        choices=["baseline", "fast", "ultra"],
        help="Perplexity computation variant",
    )
    parser.add_argument(
        "--repetitions", type=int, default=1, help="Repeat each measurement N times"
    )
    parser.add_argument(
        "--out-json", type=str, default="", help="Optional path to save JSON results"
    )
    args = parser.parse_args()

    k_values = [int(x) for x in args.k_values.split(",") if x.strip()]

    results = run_grid(
        n_max=args.n_max,
        k_values=k_values,
        advanced=args.advanced,
        method="fast" if args.method == "baseline" else args.method,
        repetitions=args.repetitions,
    )

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {args.out_json}")

    # Quick summary table
    try:
        # Build simple aligned columns without external deps
        headers = ["n", "k", "adv", "method", "perplexity", "time_sec_avg"]
        print("\nSummary:")
        print(" | ".join(headers))
        print("-" * 72)
        for r in results:
            row = [
                str(r["n"]),
                str(r["k"]),
                str(int(r["advanced"])),
                r["method"],
                f"{r['perplexity']:.2f}",
                f"{r['time_sec_avg']:.3f}",
            ]
            print(" | ".join(row))
    except Exception:
        pass


if __name__ == "__main__":
    main()

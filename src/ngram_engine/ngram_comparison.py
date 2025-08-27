"""N-gram model comparison script (Task 2)

Builds n-gram language models (n=1..max_n) for multiple BPE vocab sizes (k values),
computes validation perplexities per order, tunes interpolation weights on validation,
then reports / saves test perplexities with interpolation.

Caching:
  - Tokenized sentences (after BPE-style merging via vocab) are cached in
    data/ngram_outputs/sentences_k{K}.txt
  - Reused if present to avoid re-tokenizing.

Outputs:
  - JSON summary: data/ngram_outputs/ngram_perplexity_summary.json
  - Per-k interpolation weights and test perplexity printed in table form.
  - Sample generations per k saved to data/ngram_outputs/generation_k{K}.txt

Usage (example):
  python -m src.n-gram_engine.ngram_comparison \
      --k-values 500 1000 1500 \
      --max-n 4 \
      --train data/corpora/Shakespeare_clean_train.txt \
      --val data/corpora/Shakespeare_clean_valid.txt \
      --test data/corpora/Shakespeare_clean_test.txt \
      --vocab-dir data/bpe_outputs \
      --suffix _adv \
      --interp-step 0.1

If --suffix is provided (e.g. _adv) it expects vocab filenames like
  vocab_with_k{K}_adv.txt
otherwise vocab_with_k{K}.txt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .nge_utils import split_into_sentences_and_normalize
from .ngram_model import NGramModel, build_and_evaluate

# Default configuration (used when arguments omitted)
DEFAULT_K_VALUES = [500, 1000, 1500, 2000]
DEFAULT_TRAIN = "data/corpora/Shakespeare_clean_train.txt"
DEFAULT_VAL = "data/corpora/Shakespeare_clean_valid.txt"
DEFAULT_TEST = "data/corpora/Shakespeare_clean_test.txt"
DEFAULT_MAX_N = 4
DEFAULT_INTERP_STEP = 0.1

# ---------------------------- Helpers ----------------------------


def load_vocab(k: int, vocab_dir: str, suffix: str) -> List[str]:
    fname = f"vocab_with_k{k}{suffix}.txt"
    path = Path(vocab_dir) / fname
    if not path.exists():
        raise FileNotFoundError(f"Vocab file not found: {path}")
    return path.read_text(encoding="utf-8").splitlines()


def load_or_build_sentences(
    raw_text_path: str, vocab_tokens: Sequence[str], k: int
) -> List[List[str]]:
    out_dir = Path("data/ngram_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_file = out_dir / f"sentences_k{k}.txt"
    if cache_file.exists():
        sentences = [
            ln.strip().split()
            for ln in cache_file.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        return sentences
    raw = Path(raw_text_path).read_text(encoding="utf-8")
    sentences = split_into_sentences_and_normalize(
        raw, vocab_tokens, track_progress=True
    )
    with cache_file.open("w", encoding="utf-8") as f:
        for s in sentences:
            f.write(" ".join(s) + "\n")
    return sentences


def summarize_perplexities(per_order: Dict[int, float]) -> str:
    return ", ".join(f"n={n}:{ppl:.1f}" for n, ppl in sorted(per_order.items()))


# ---------------------------- Main Flow ----------------------------


def run_experiments(args):
    summary: Dict[str, Any] = {
        "config": {
            "k_values": args.k_values,
            "max_n": args.max_n,
            "interp_step": args.interp_step,
            "suffix": args.suffix,
        },
        "results": {},
    }

    for k in args.k_values:
        print(f"\n===== Processing k={k} =====")
        vocab_tokens = load_vocab(k, args.vocab_dir, args.suffix)
        print(f"Loaded vocab size: {len(vocab_tokens)}")

        train_sentences = load_or_build_sentences(args.train, vocab_tokens, k)
        val_sentences = load_or_build_sentences(args.val, vocab_tokens, k)
        test_sentences = load_or_build_sentences(args.test, vocab_tokens, k)

        eval_res = build_and_evaluate(
            train_sentences=train_sentences,
            val_sentences=val_sentences,
            test_sentences=test_sentences,
            max_n=args.max_n,
            vocab_tokens=vocab_tokens,
            interp_step=args.interp_step,
        )

        per_order = eval_res["per_order_ppl"]
        lambdas = eval_res["lambdas"]
        interp_test_ppl = eval_res["interp_test_ppl"]

        print(f"Validation perplexities per order: {summarize_perplexities(per_order)}")
        print(
            f"Chosen interpolation lambdas (n=1..{args.max_n}): {[round(x, 3) for x in lambdas]}"
        )
        print(f"Test perplexity (interpolated): {interp_test_ppl:.2f}")

        # Sample generation
        model: NGramModel = eval_res["model"]  # type: ignore
        gen = model.generate(
            ["<s>"], max_tokens=args.sample_tokens, mode=args.generation_mode
        )
        print("Sample generation:", " ".join(gen))
        gen_path = Path("data/ngram_outputs") / f"generation_k{k}.txt"
        gen_path.write_text(" ".join(gen), encoding="utf-8")

        summary["results"][str(k)] = {
            "val_perplexity_per_order": per_order,
            "interp_lambdas": lambdas,
            "test_perplexity_interpolated": interp_test_ppl,
            "sample_generation_file": str(gen_path),
        }

    out_json = Path("data/ngram_outputs/ngram_perplexity_summary.json")
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved summary to {out_json}")

    # Pretty table
    print("\n=== Summary Table (Test Perplexity) ===")
    header = ["k"] + [f"n={n}" for n in range(1, args.max_n + 1)] + ["interp_test_ppl"]
    print(" | ".join(header))
    for k in args.k_values:
        res = summary["results"][str(k)]
        per_order = res["val_perplexity_per_order"]
        row = (
            [str(k)]
            + [
                f"{per_order[str(n)] if isinstance(per_order, dict) and str(n) in per_order else per_order[n]:.1f}"
                for n in range(1, args.max_n + 1)
            ]
            + [f"{res['test_perplexity_interpolated']:.1f}"]
        )
        print(" | ".join(row))


# ---------------------------- CLI ----------------------------


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="N-gram perplexity comparison (defaults used if no args supplied)"
    )
    p.add_argument(
        "--k-values",
        type=int,
        nargs="*",
        default=None,
        help=f"List of vocab sizes k to evaluate (default: {DEFAULT_K_VALUES})",
    )
    p.add_argument(
        "--max-n",
        type=int,
        default=DEFAULT_MAX_N,
        help=f"Maximum n-gram order (default: {DEFAULT_MAX_N})",
    )
    p.add_argument(
        "--train",
        type=str,
        default=None,
        help=f"Path to training text file (default: {DEFAULT_TRAIN})",
    )
    p.add_argument(
        "--val",
        type=str,
        default=None,
        help=f"Path to validation text file (default: {DEFAULT_VAL})",
    )
    p.add_argument(
        "--test",
        type=str,
        default=None,
        help=f"Path to test text file (default: {DEFAULT_TEST})",
    )
    p.add_argument(
        "--vocab-dir",
        type=str,
        default="data/bpe_outputs",
        help="Directory containing vocab_with_k*.txt files",
    )
    p.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Optional suffix in vocab filenames (e.g. _adv)",
    )
    p.add_argument(
        "--interp-step",
        type=float,
        default=DEFAULT_INTERP_STEP,
        help=f"Grid step for interpolation lambda search (default: {DEFAULT_INTERP_STEP})",
    )
    p.add_argument(
        "--sample-tokens",
        type=int,
        default=40,
        help="Max tokens to generate for sample output",
    )
    p.add_argument(
        "--generation-mode",
        type=str,
        choices=["argmax", "sample"],
        default="argmax",
        help="Generation strategy",
    )
    return p


def main():
    args = build_arg_parser().parse_args()
    # Fill defaults if omitted
    if not args.k_values:
        args.k_values = DEFAULT_K_VALUES
    if args.train is None:
        args.train = DEFAULT_TRAIN
    if args.val is None:
        args.val = DEFAULT_VAL
    if args.test is None:
        args.test = DEFAULT_TEST
    print("Running with configuration:")
    print("  k_values:", args.k_values)
    print("  train:", args.train)
    print("  val:", args.val)
    print("  test:", args.test)
    print("  max_n:", args.max_n)
    print("  interp_step:", args.interp_step)
    print("  suffix:", args.suffix or "<none>")
    run_experiments(args)


if __name__ == "__main__":
    main()

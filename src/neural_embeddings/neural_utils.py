import csv
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch

from src.ngram_engine.nge_utils import apply_bpe


def ensure_dir(path: str):
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def append_run_result(
    results_csv: str,
    config: Dict[str, Any],
    best_val_loss: float,
    best_val_ppl: Optional[float],
    history_len: int,
):
    """Append a single run summary (config + best metrics) to a CSV file.
    Creates the file with a header if missing.
    """
    ensure_dir(os.path.dirname(results_csv) or ".")
    header = list(config.keys()) + ["best_val_loss", "best_val_ppl", "epochs_trained"]
    row = [config[k] for k in config] + [best_val_loss, best_val_ppl, history_len]
    write_header = not os.path.exists(results_csv)
    with open(results_csv, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)


def save_history_csv(
    history: List[Tuple[int, float, float, float, float]], out_path: str
):
    """Save per-epoch history to CSV.
    history tuples: (epoch, train_loss, val_loss, val_ppl, val_ppl_interpolated)
    """
    ensure_dir(os.path.dirname(out_path) or ".")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["epoch", "train_loss", "val_loss", "val_ppl", "val_ppl_interpolated"]
        )
        for row in history:
            w.writerow(row)


def save_history_jsonl(
    history: List[Tuple[int, float, float, float, float]], out_path: str
):
    """Save history as JSONL (one JSON object per line)."""
    ensure_dir(os.path.dirname(out_path) or ".")
    with open(out_path, "w") as f:
        for epoch, tr, vl, ppl, ppl_i in history:
            f.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "train_loss": tr,
                        "val_loss": vl,
                        "val_ppl": ppl,
                        "val_ppl_interpolated": ppl_i,
                    }
                )
                + "\n"
            )


def load_checkpoint_histories(ckpt_dir: str):
    """Iterate over .pt checkpoints and yield (path, config, best_val_loss, history)."""
    for fname in os.listdir(ckpt_dir):
        if not fname.endswith(".pt"):
            continue
        fpath = os.path.join(ckpt_dir, fname)
        try:
            data = torch.load(fpath, map_location="cpu")
            cfg = data.get("config", {})
            best = data.get("best_val_loss")
            hist = data.get("history", [])
            yield fpath, cfg, best, hist
        except Exception as e:
            print(f"Failed to load checkpoint {fname}: {e}")


def build_aggregate_from_checkpoints(ckpt_dir: str, out_csv: str):
    """Create / overwrite an aggregate CSV from all checkpoints in a directory."""
    rows = []
    for path, cfg, best, hist in load_checkpoint_histories(ckpt_dir):
        if best is None:
            continue
        val_ppl = None
        if hist:
            # take last recorded val_ppl in history
            val_ppl = hist[-1][3]
        rows.append(
            {
                **cfg,
                "best_val_loss": best,
                "best_val_ppl": val_ppl,
                "epochs_trained": len(hist),
            }
        )
    if not rows:
        print("No checkpoint data to aggregate.")
        return
    # determine header order
    base_keys = sorted(
        {
            k
            for r in rows
            for k in r.keys()
            if k not in {"best_val_loss", "best_val_ppl", "epochs_trained"}
        }
    )
    header = base_keys + ["best_val_loss", "best_val_ppl", "epochs_trained"]
    ensure_dir(os.path.dirname(out_csv) or ".")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([r.get(k, "") for k in header])
    print(f"Wrote aggregate CSV with {len(rows)} rows to {out_csv}")


def try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore

        return plt
    except Exception:
        return None


def plot_history(
    history: List[Tuple[int, float, float, float, float]],
    out_path: Optional[str] = None,
    show: bool = False,
):
    """Plot training & validation loss (and optionally ppl) for a single run.
    Requires matplotlib. Silently skips if not available.
    """
    plt = try_import_matplotlib()
    if plt is None:
        print("matplotlib not available; skipping plot.")
        return
    if not history:
        print("Empty history; nothing to plot.")
        return
    epochs = [h[0] for h in history]
    train_loss = [h[1] for h in history]
    val_loss = [h[2] for h in history]
    val_ppl = [h[3] for h in history]
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(loc="upper right")
    ax2 = plt.twinx()
    ax2.plot(epochs, val_ppl, color="green", alpha=0.3, label="val_ppl")
    ax2.set_ylabel("val_ppl")
    lines, labels = plt.gca().get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines + lines2, labels + labels2, loc="upper center")
    plt.tight_layout()
    if out_path:
        ensure_dir(os.path.dirname(out_path) or ".")
        plt.savefig(out_path)
        print(f"Saved plot to {out_path}")
    if show:
        plt.show()
    plt.close()


def plot_aggregate_csv(
    csv_path: str,
    x_key: str,
    metric: str = "best_val_loss",
    out_path: Optional[str] = None,
    show: bool = False,
):
    """Plot metric vs x_key from an aggregate CSV (scatter).
    Assumes header row present. Ignores rows missing required columns.
    """
    plt = try_import_matplotlib()
    if plt is None:
        print("matplotlib not available; skipping plot.")
        return
    if not os.path.exists(csv_path):
        print("CSV not found:", csv_path)
        return
    import math

    xs, ys = [], []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if x_key not in row or metric not in row:
                continue
            try:
                x_val = float(row[x_key])
                y_val = float(row[metric])
                if math.isfinite(x_val) and math.isfinite(y_val):
                    xs.append(x_val)
                    ys.append(y_val)
            except ValueError:
                continue
    if not xs:
        print("No valid data to plot from", csv_path)
        return
    plt.figure(figsize=(6, 4))
    plt.scatter(xs, ys, alpha=0.7)
    plt.xlabel(x_key)
    plt.ylabel(metric)
    plt.title(f"{metric} vs {x_key}")
    plt.tight_layout()
    if out_path:
        ensure_dir(os.path.dirname(out_path) or ".")
        plt.savefig(out_path)
        print(f"Saved aggregate plot to {out_path}")
    if show:
        plt.show()
    plt.close()


def apply_bpe_and_encode(self, text: str):
    # track progress if train_ids not yet set
    track_progress = not hasattr(self, "train_ids")
    tokens = apply_bpe(text, self.merge_rules, track_progress=track_progress)
    # tokens is list of token sequences (sentences); flatten
    flat = []
    for sent in tokens:
        flat.extend(sent)
    flat = [token for token in flat if token != "0"]
    stoi = {c: i for i, c in enumerate(self.vocab)}

    def encode_list(seq):
        return [stoi[c] for c in seq]

    return encode_list(flat)


def load_vocab_and_merge_rules(self, k: int, suffix: str):
    vocab = (
        open(f"data/bpe_outputs/vocab_with_k{k}{suffix}.txt", "r").read().splitlines()
    )
    merge_rules = (
        open(f"data/bpe_outputs/merges_k{k}{suffix}.txt", "r").read().splitlines()
    )
    return vocab, merge_rules


def load_vocab_and_merge_rules(k: int, suffix: str):
    vocab = (
        open(f"data/bpe_outputs/vocab_with_k{k}{suffix}.txt", "r").read().splitlines()
    )
    merge_rules = (
        open(f"data/bpe_outputs/merges_k{k}{suffix}.txt", "r").read().splitlines()
    )
    return vocab, merge_rules


def get_or_build_bpe_cache(k: int, suffix: str):
    vocab, merge_rules = load_vocab_and_merge_rules(k, suffix)
    vocab.extend(["<s>", "</s>"])
    os.makedirs("data/emb_lm/encoded_texts", exist_ok=True)
    # Encode training set
    train_ids = apply_bpe_and_encode(
        open("data/corpora/Shakespeare_clean_train.txt", "r").read(),
        vocab,
        merge_rules,
    )
    with open(
        f"data/emb_lm/encoded_texts/bpe_cache_k{k}{suffix}_train.txt",
        "w",
        encoding="utf-8",
    ) as f:
        for token in train_ids:
            f.write(str(token) + " ")
    print("Encoding validation and test sets...")
    # Encode validation set
    val_ids = apply_bpe_and_encode(
        open("data/corpora/Shakespeare_clean_valid.txt", "r").read(),
        vocab,
        merge_rules,
    )
    with open(
        f"data/emb_lm/encoded_texts/bpe_cache_k{k}{suffix}_valid.txt",
        "w",
        encoding="utf-8",
    ) as f:
        for token in val_ids:
            f.write(str(token) + " ")
    # Encode test set
    test_ids = apply_bpe_and_encode(
        open("data/corpora/Shakespeare_clean_test.txt", "r").read(),
        vocab,
        merge_rules,
    )
    with open(
        f"data/emb_lm/encoded_texts/bpe_cache_k{k}{suffix}_test.txt",
        "w",
        encoding="utf-8",
    ) as f:
        for token in test_ids:
            f.write(str(token) + " ")


def apply_bpe_and_encode(text: str, vocab: List[str], merge_rules: List[str]):
    # track progress if train_ids not yet set
    track_progress = True
    tokens = apply_bpe(text, merge_rules, track_progress=track_progress)
    # tokens is list of token sequences (sentences); flatten
    flat = []
    for sent in tokens:
        flat.extend(sent)
    flat = [token for token in flat if token != "0"]
    stoi = {c: i for i, c in enumerate(vocab)}

    encode = lambda s: [stoi[c] for c in s]
    return encode(flat)


if __name__ == "__main__":
    # Example usage
    k_values = [50, 100, 150, 250, 350, 500, 750, 1000, 1250, 1500]
    suffix = "_adv"
    for k in k_values:
        bpe_cache = get_or_build_bpe_cache(k=k, suffix=suffix)

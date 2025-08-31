import json
import math
import os

import torch

from src.neural_embeddings.gpt_model import GPTModel
from src.neural_embeddings.neural_ngrams import NgramLM, train_with_early_stopping

NUM_STEPS = 1000
BATCH_SIZE = 16


def save_params(param_name, param_value, history):
    results_file = "data/hyperparams/gpt_search_results.jsonl"
    run_data = {"params": {param_name: param_value}, "history": []}
    for epoch_data in history:
        (epoch, avg_train, val_loss, val_ppl, val_ppl_interpolated) = epoch_data

        run_data["history"].append(
            {
                "epoch": epoch,
                "avg_train_loss": avg_train,
                "val_loss": val_loss,
                "val_ppl": val_ppl,
                "val_ppl_interpolated": val_ppl_interpolated,
            }
        )
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(run_data) + "\n")
    print(f"Saved results for {param_name}={param_value}")


def search_best_k():
    k_values = [50, 100, 150, 250, 350, 500, 750, 1000, 1250, 1500, 2000]
    print("Starting parameter search for k...")
    for k in k_values:
        # Instantiate, train, and evaluate the model
        model = GPTModel(n=3, k=k, embed_dim=256, hidden_dim=256, alpha=1e-3, lam=0.7)
        best_val, best_state, history = train_with_early_stopping(
            model, max_epochs=25, num_steps=NUM_STEPS, patience=5
        )

        # Get the final validation perplexity
        save_params("k", k, history)

    print("Parameter search for k complete.")


def search_best_hidden_dim():
    h_values = [128, 256]
    print("Starting parameter search for hidden_dim...")
    for h in h_values:
        model = GPTModel(n=3, k=1000, embed_dim=256, hidden_dim=h, alpha=1e-3, lam=0.7)
        best_val, best_state, history = train_with_early_stopping(
            model, max_epochs=25, num_steps=NUM_STEPS, patience=5
        )
        save_params("hidden_dim", h, history)
    print("Parameter search for hidden_dim complete.")


def search_best_lr():
    a_values = [1e-1, 1e-2, 1e-3, 1e-4]
    print("Starting parameter search for alpha...")
    for a in a_values:
        model = GPTModel(n=3, k=1000, embed_dim=256, hidden_dim=256, alpha=a, lam=0.7)
        best_val, best_state, history = train_with_early_stopping(
            model, max_epochs=25, num_steps=NUM_STEPS, patience=5
        )
        save_params("alpha", a, history)
    print("Parameter search for alpha complete.")


def search_best_lambda():
    l_values = [0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]
    print("Starting parameter search for lambda...")
    for l in l_values:
        model = GPTModel(n=3, k=1000, embed_dim=256, hidden_dim=256, alpha=1e-3, lam=l)
        best_val, best_state, history = train_with_early_stopping(
            model, max_epochs=25, num_steps=NUM_STEPS, patience=5
        )
        save_params("lambda", l, history)
    print("Parameter search for lambda complete.")


def plot_results():
    results_file = "data/hyperparams/gpt_search_results.jsonl"
    if not os.path.exists(results_file):
        print("No results file found at", results_file)
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; cannot plot.")
        return

    # Load runs
    runs = []
    with open(results_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                runs.append(json.loads(line))
            except Exception:
                pass
    if not runs:
        print("No parsable runs found.")
        return

    # Aggregate by parameter name/value
    agg = {}  # {param_name: {value: {best_val_loss, best_val_ppl, best_val_ppl_interp}}}
    for r in runs:
        params = r.get("params", {})
        history = r.get("history", [])
        if not params or not history:
            continue
        # Determine best (lowest) val_loss entry in history
        best_entry = min(history, key=lambda e: e.get("val_loss", float("inf")))
        for pname, pval in params.items():
            if pname not in agg:
                agg[pname] = {}
            cur = agg[pname].get(pval)
            metrics = {
                "val_loss": best_entry.get("val_loss"),
                "val_ppl": best_entry.get("val_ppl"),
                "val_ppl_interpolated": best_entry.get("val_ppl_interpolated"),
            }
            # Keep best (lowest val_loss)
            if cur is None or metrics["val_loss"] < cur["val_loss"]:
                agg[pname][pval] = metrics

    os.makedirs("data/hyperparams/plots", exist_ok=True)

    for pname, table in agg.items():
        # Sort by numeric value if possible else by string
        try:
            items = sorted(table.items(), key=lambda kv: float(kv[0]))
        except Exception:
            items = sorted(table.items(), key=lambda kv: str(kv[0]))
        x = [kv[0] for kv in items]
        val_ppl = [kv[1]["val_ppl"] for kv in items]
        val_ppl_interp = [kv[1]["val_ppl_interpolated"] for kv in items]
        val_loss = [kv[1]["val_loss"] for kv in items]

        plt.figure(figsize=(6, 4))
        if any(v is not None for v in val_ppl):
            plt.plot(x, val_ppl, marker="o", label="val_ppl")
        if any(v is not None for v in val_ppl_interp):
            plt.plot(x, val_ppl_interp, marker="s", label="val_ppl_interp")
        plt.title(f"Parameter sweep: {pname}")
        plt.xlabel(pname)
        plt.ylabel("Perplexity")
        plt.legend()
        plt.tight_layout()
        out_path = f"data/hyperparams/plots/{pname}_perplexity_gpt.png"
        plt.savefig(out_path)
        plt.close()
        # Also save val_loss
        plt.figure(figsize=(6, 4))
        plt.plot(x, val_loss, marker="o")
        plt.title(f"Parameter sweep (loss): {pname}")
        plt.xlabel(pname)
        plt.ylabel("Val Loss")
        plt.tight_layout()
        out_path_loss = f"data/hyperparams/plots/{pname}_val_loss.png"
        plt.savefig(out_path_loss)
        plt.close()
        print(f"Saved plots for {pname} -> {out_path}, {out_path_loss}")

    print("Plotting complete.")


# --- New: Post-hoc bits-per-character evaluation helpers ---


def compute_bpc(
    model: NgramLM, split: str = "val", raw_path: str | None = None
) -> float:
    """Compute bits-per-character (bpc) for a trained model (lam must be 1.0)."""
    assert model.lam == 1.0, "Set lam=1.0 before computing bpc"
    if split == "val":
        ids = model.val_ids
        if raw_path is None:
            raw_path = "data/corpora/Shakespeare_clean_valid.txt"
    elif split == "test":
        ids = model.test_ids
        if raw_path is None:
            raw_path = "data/corpora/Shakespeare_clean_test.txt"
    else:
        ids = model.train_ids
        if raw_path is None:
            raw_path = "data/corpora/Shakespeare_clean_train.txt"
    try:
        with open(raw_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
    except OSError:
        print(f"Raw text file not found: {raw_path}")
        return float("nan")
    char_count = len(raw_text)
    if char_count == 0:
        return float("nan")
    ctx_len = model.n - 1
    total_nll = 0.0
    token_count = 0
    model.eval()
    with torch.no_grad():
        for i in range(ctx_len, len(ids)):
            context = ids[i - ctx_len : i].unsqueeze(0).to(model.device)
            target = ids[i].unsqueeze(0).to(model.device)
            logits, _ = model.forward(context, target)
            log_probs = torch.log_softmax(logits, dim=-1)
            nll = -log_probs[0, target]
            total_nll += nll.item()
            token_count += 1
    if token_count == 0:
        return float("nan")
    bpc = (total_nll / char_count) / math.log(2)
    print(
        f"bpc={bpc:.4f} (tokens={token_count}, chars={char_count}, avg_nll={total_nll / token_count:.4f})"
    )
    return bpc


def compute_bpc_for_config(
    n: int,
    k: int,
    embed_dim: int,
    hidden_dim: int,
    alpha: float,
    ckpt_path: str | None = None,
    split: str = "val",
) -> float:
    """Rebuild model (lam=1.0), optionally load checkpoint, then compute bpc."""
    model = NgramLM(
        n=n, k=k, embed_dim=embed_dim, hidden_dim=hidden_dim, alpha=alpha, lam=1.0
    )
    if ckpt_path and os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state = ckpt.get("model_state") or ckpt
            model.load_state_dict(state, strict=False)
            model.to(model.device)
            print(f"Loaded state from {ckpt_path}")
        except Exception as e:
            print(f"Failed loading checkpoint {ckpt_path}: {e}")
    return compute_bpc(model, split=split)


def search_neural_ngrams(test_for="all"):
    if test_for != "all":
        if test_for == "k":
            search_best_k()
        elif test_for == "hidden_dim":
            search_best_hidden_dim()
        elif test_for == "lr":
            search_best_lr()
        elif test_for == "lambda":
            search_best_lambda()
        elif test_for == "n":
            search_best_n()
        else:
            print(f"Unknown test_for value: {test_for}")
            return
        return
    search_best_k()
    search_best_hidden_dim()
    search_best_lr()
    search_best_lambda()


def search_best_n():
    n_values = [2, 3, 4, 5]
    print("Starting parameter search for n...")
    for n in n_values:
        # Instantiate, train, and evaluate the model
        model = NgramLM(n=n, k=1000, embed_dim=256, hidden_dim=256, alpha=1e-3, lam=0.7)
        best_val, best_state, history = train_with_early_stopping(
            model, max_epochs=25, num_steps=10000, patience=5
        )

        # Get the final validation perplexity
        save_params("n", n, history)

    print("Parameter search for n complete.")


def get_best_k():
    k_values = [50, 100, 150, 250, 350, 500, 750, 1000, 1250, 1500, 2000]
    bpc_values = []
    for k in k_values:
        bpc = compute_bpc_for_config(
            n=3,
            k=k,
            embed_dim=256,
            hidden_dim=256,
            alpha=1e-3,
            ckpt_path="data/models/ngram_n3_k2000_embed256_hidden256_alpha0.001_lam0.7.pt",
            split="test",
        )
        bpc_values.append((k, bpc))
    print("Bits-per-character (bpc) results for various k values:")
    for k, bpc in bpc_values:
        print(f"k={k}: bpc={bpc:.4f}")


if __name__ == "__main__":
    search_neural_ngrams("all")
    get_best_k()
    plot_results()

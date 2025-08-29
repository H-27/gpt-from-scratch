import argparse
import json
import os
from datetime import datetime

# Placeholder training script structure. To be filled with torch logic.


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=1000, help="BPE merges / vocab size ref")
    p.add_argument("--n", type=int, default=3, help="context length (n-gram style)")
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--optimizer", type=str, default="adam")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--top-k", type=int, default=3, dest="top_k")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--interpolation",
        type=float,
        default=1.0,
        help="lambda weight if blending with ngram probs",
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs("data/emb_lm/checkpoints", exist_ok=True)
    meta = {
        "args": vars(args),
        "timestamp": datetime.utcnow().isoformat(),
    }
    with open("data/emb_lm/run_stub.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("Stub train script created metadata. Implement training loop next.")


if __name__ == "__main__":
    main()

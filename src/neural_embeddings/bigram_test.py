import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import optimizer


class Bigram_lm(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super(Bigram_lm, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embeddings = nn.Embedding(vocab_size, vocab_size)

    def forward(self, context, target=None):
        embeddings = self.embeddings(context)
        if target is None:
            loss = None
        else:
            # context_ids shape: (batch_size, 1)

            embedding_shape = embeddings.shape
            embeddings = embeddings.view(
                embedding_shape[0] * embedding_shape[1], embedding_shape[2]
            )
            targets = target.view(embedding_shape[0] * embedding_shape[1])

            loss = F.cross_entropy(embeddings, targets)
        return embeddings, loss

    def generate(self, context, n_new_tokens: int):
        for _ in range(n_new_tokens):
            embeddings, loss = self(context)

            embeddings = embeddings[:, -1, :]  # (batch_size, vocab_size)
            probs = F.softmax(embeddings, dim=-1)  # (batch_size, vocab_size)
            prediction = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
            context = torch.cat(
                (context, prediction), dim=1
            )  # (batch_size, context_len+1)
        return context


if __name__ == "__main__":
    with open("data/corpora/Shakespeare_clean_test.txt", "r") as f:
        text = f.read()
    print("Length of text:", len(text.split()))
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print("Unique characters:", "".join(chars))
    print("Vocab size:", vocab_size)

    # mapping
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}

    # encode the text to integers
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])

    test_text = "the world is so small"
    print("Encoded:", encode(test_text))
    print("Decoded:", decode(encode(test_text)))

    data = torch.tensor(encode(text), dtype=torch.long)
    print("Data tensor shape:", data.shape, "dtype:", data.dtype)
    print("Data tensor:", data[:100])
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    # data loader
    block_size = 8
    train_text = text[: block_size + 1]
    x = train_text[:block_size]
    y = train_text[1 : block_size + 1]
    for t in range(block_size):
        context = x[: t + 1]
        target = y[t]
        context = encode(context)
        target = encode(target)
        print(f"when input is {context!r}, predict {target!r}")
    split = "train"
    batch_size = 4

    def get_batch():
        data = train_data if split == "train" else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
        return x, y

    xb, yb = get_batch()
    print("Input: ", xb)
    print("input:", decode(xb[0].tolist()))
    print("input shape: ", xb.shape)
    print("Target:", yb)
    print("target:", decode(yb[0].tolist()))
    print("target shape:", yb.shape)

    model = Bigram_lm(vocab_size, embed_dim=32)

    starting_charachter = torch.zeros((1, 1), dtype=torch.long)

    generated_characters = model.generate(starting_charachter, 100)
    generated_text = decode(generated_characters[0].tolist())
    print("Generated text:", generated_text)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    batch_size = 32
    for step in range(100000):
        xb, yb = get_batch()
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 1000 == 0:
            print(f"step {step}: loss {loss.item():.4f}")

    # simple test
    context = torch.tensor([[1], [2], [3]])
    target = torch.tensor([[2], [3], [4]])
    embeddings, loss = model(context, target)
    print("Loss:", loss.item())

    generated_characters = model.generate(starting_charachter, 100)
    generated_text = decode(generated_characters[0].tolist())
    print("Generated text:", generated_text)

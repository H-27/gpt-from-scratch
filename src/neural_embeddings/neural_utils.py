import torch
import torch.nn as nn
import torch.nn.functional as F


class ngram_lm(nn.Module):
    def __init__(self, vocab_size: int, n: int, embed_dim: int, hidden_dim: int):
        super(ngram_lm, self).__init__()
        self.vocab_size = vocab_size
        self.n = n
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim * (n - 1), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, context, target):
        # context_ids shape: (batch_size, n-1)
        embeds = self.embeddings(context)  # (batch_size, n-1, embed_dim)
        embeds = embeds.view(embeds.size(0), -1)  # (batch_size, (n-1)*embed_dim)
        hidden = torch.relu(self.fc1(embeds))  # (batch_size, hidden_dim)
        output = self.fc2(hidden)  # (batch_size, vocab_size)
        loss = F.cross_entropy(output, target)
        return output

    def generate(self, context, n_new_tokens: int):
        raise NotImplementedError

    def calculate_perplexity(self, data_loader):
        raise NotImplementedError

    def train():
        # train, then validate, test only after training complete
        raise NotImplementedError

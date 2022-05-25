"""A deterministic trainable bag-of-token-embedding dual encoder."""

import re
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class TinyDualEncoder(nn.Module):
    """Mean-pooled token embeddings followed by a shared linear projection."""

    def __init__(self, vocabulary: Sequence[str], dimensions: int = 16, seed: int = 23) -> None:
        super().__init__()
        if not vocabulary or len(set(vocabulary)) != len(vocabulary):
            raise ValueError("vocabulary must be non-empty and contain unique tokens")
        if dimensions < 1:
            raise ValueError("dimensions must be positive")
        self.vocabulary = tuple(vocabulary)
        self.token_ids = {token: index + 1 for index, token in enumerate(vocabulary)}
        self.embedding = nn.Embedding(len(vocabulary) + 1, dimensions, padding_idx=0)
        self.projection = nn.Linear(dimensions, dimensions, bias=False)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            self.embedding.weight.normal_(mean=0.0, std=0.2, generator=generator)
            self.embedding.weight[0].zero_()
            self.projection.weight.copy_(torch.eye(dimensions))

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        if not texts:
            raise ValueError("at least one text is required")
        pooled = []
        for text in texts:
            ids = [self.token_ids[token] for token in tokenize(text) if token in self.token_ids]
            if ids:
                pooled.append(self.embedding(torch.tensor(ids, dtype=torch.long)).mean(dim=0))
            else:
                pooled.append(self.embedding.weight[0])
        return F.normalize(self.projection(torch.stack(pooled)), dim=1)

    def forward(self, texts: Sequence[str]) -> torch.Tensor:
        return self.encode(texts)

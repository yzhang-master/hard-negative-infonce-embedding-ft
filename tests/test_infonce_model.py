import math

import pytest
import torch

from hard_negative_ft.losses import info_nce
from hard_negative_ft.model import TinyDualEncoder, tokenize


def test_infonce_matches_manual_positive_and_hard_negative_loss() -> None:
    query = torch.tensor([[2.0, 0.0]], requires_grad=True)
    positive = torch.tensor([[1.0, 0.0]], requires_grad=True)
    easy_negative = torch.tensor([[[0.0, 1.0]]], requires_grad=True)
    hard_negative = torch.tensor([[[1.0, 0.0]]])

    easy = info_nce(query, positive, easy_negative, temperature=0.1)
    hard = info_nce(query, positive, hard_negative, temperature=0.1)

    assert easy.item() == pytest.approx(math.log1p(math.exp(-10.0)), abs=1e-6)
    assert hard.item() == pytest.approx(math.log(2.0), abs=1e-6)
    assert hard > easy
    easy.backward()
    assert query.grad is not None and torch.count_nonzero(query.grad) > 0
    assert easy_negative.grad is not None and torch.count_nonzero(easy_negative.grad) > 0


def test_infonce_uses_other_batch_positives_as_negatives() -> None:
    queries = torch.eye(2)
    batch_loss = info_nce(queries, queries, temperature=0.1)
    explicit_loss = info_nce(
        queries[:1], queries[:1], queries[1:].unsqueeze(0), temperature=0.1
    )

    assert batch_loss.item() == pytest.approx(explicit_loss.item(), abs=1e-6)
    with pytest.raises(ValueError, match="temperature"):
        info_nce(queries, queries, temperature=0.0)
    with pytest.raises(ValueError, match="negatives"):
        info_nce(queries, queries, torch.ones(2, 3))


def test_dual_encoder_is_deterministic_normalized_and_trainable() -> None:
    vocabulary = ("query", "answer", "shared")
    first = TinyDualEncoder(vocabulary, dimensions=8, seed=5)
    second = TinyDualEncoder(vocabulary, dimensions=8, seed=5)

    encoded = first(["query shared", "answer"])
    repeated = second(["query shared", "answer"])

    torch.testing.assert_close(encoded, repeated)
    torch.testing.assert_close(encoded.norm(dim=1), torch.ones(2))
    assert tokenize("Query, SHARED!") == ["query", "shared"]
    assert torch.count_nonzero(first(["unknown token"])) == 0
    encoded.sum().backward()
    assert first.embedding.weight.grad is not None

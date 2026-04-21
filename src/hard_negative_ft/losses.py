"""InfoNCE with in-batch and explicit hard negatives."""

import torch
import torch.nn.functional as F


def info_nce(
    queries: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor | None = None,
    *,
    temperature: float = 0.05,
) -> torch.Tensor:
    """Compute -log exp(positive/tau) / (positive + all allowed negatives).

    Other positives in the batch act as in-batch negatives. Explicit negatives have
    shape [batch, negatives, dimensions]. The positive logit is column zero.
    """
    if queries.ndim != 2 or positives.shape != queries.shape:
        raise ValueError("queries and positives must have matching [batch, dimensions] shapes")
    if queries.shape[0] == 0 or queries.shape[1] == 0:
        raise ValueError("batch and embedding dimensions must be non-empty")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if negatives is not None and (
        negatives.ndim != 3
        or negatives.shape[0] != queries.shape[0]
        or negatives.shape[2] != queries.shape[1]
    ):
        raise ValueError("negatives must have shape [batch, negatives, dimensions]")

    queries = F.normalize(queries, dim=1)
    positives = F.normalize(positives, dim=1)
    positive_logits = (queries * positives).sum(dim=1, keepdim=True) / temperature
    in_batch = queries @ positives.T / temperature
    in_batch = in_batch.masked_fill(torch.eye(len(queries), dtype=torch.bool), float("-inf"))
    logit_columns = [positive_logits, in_batch]
    if negatives is not None and negatives.shape[1] > 0:
        negatives = F.normalize(negatives, dim=2)
        explicit = torch.einsum("bd,bnd->bn", queries, negatives) / temperature
        logit_columns.append(explicit)
    logits = torch.cat(logit_columns, dim=1)
    targets = torch.zeros(len(queries), dtype=torch.long, device=queries.device)
    return F.cross_entropy(logits, targets)

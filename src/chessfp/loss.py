"""Supervised contrastive loss (Khosla et al. 2020).

Given L2-normalized embeddings z_i and labels y_i, pull same-label pairs together
and push different-label pairs apart in cosine space:

    L_i = -1/|P(i)| * sum_{p in P(i)} log [ exp(z_i·z_p/τ) / sum_{a≠i} exp(z_i·z_a/τ) ]

where P(i) = { j ≠ i : y_j = y_i }.

This is the right objective for our setup because at inference we compute cosine
similarity between game embeddings and per-player centroids — the loss directly
optimizes that geometry.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def supcon_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
    already_normalized: bool = False,
) -> torch.Tensor:
    """Supervised contrastive loss.

    Args:
        embeddings: (N, D) — game embeddings.
        labels:     (N,)   — integer player ids.
        temperature: cosine logits divisor.
        already_normalized: skip the L2 norm if you've already done it.

    Returns scalar loss. If no anchor has a positive in the batch, returns 0.
    """
    if not already_normalized:
        embeddings = F.normalize(embeddings, dim=-1)
    n = embeddings.size(0)
    sim = embeddings @ embeddings.T / temperature

    eye = torch.eye(n, dtype=torch.bool, device=embeddings.device)
    # Mask self-similarity with a large negative finite number to keep gradients safe
    # (-inf would propagate NaN through the masked-out terms in the positive sum).
    NEG = -1e4
    sim = sim.masked_fill(eye, NEG)

    label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
    pos_mask = label_eq & ~eye

    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)

    # Use where, not multiply: 0 * -inf would be NaN.
    masked_log_prob = torch.where(pos_mask, log_prob, torch.zeros_like(log_prob))
    pos_count = pos_mask.sum(dim=1).clamp(min=1).float()
    per_anchor = -masked_log_prob.sum(dim=1) / pos_count

    valid = pos_mask.any(dim=1)
    if not valid.any():
        return embeddings.sum() * 0.0
    return per_anchor[valid].mean()


def variance_regularization(embeddings: torch.Tensor, target_std: float = 1.0) -> torch.Tensor:
    """VICReg-style variance term: penalize per-dim std falling below `target_std`.

    Computed on the *un-normalized* embeddings so it has teeth — once you've
    L2-normalized to the unit sphere, per-dim std is constrained and the
    signal goes away. Apply this directly on model.forward(x, mask).
    """
    std = torch.sqrt(embeddings.var(dim=0) + 1e-6)
    return F.relu(target_std - std).mean()

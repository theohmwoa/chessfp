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


def arcface_logits(
    embeddings: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.30,
    scale: float = 30.0,
) -> torch.Tensor:
    """ArcFace (Deng et al. 2019) margin-adjusted logits.

    Computes cosine similarity between L2-normalized embeddings and L2-normalized
    class weights, then subtracts an additive angular margin from the target
    class's cos(θ) — i.e. uses cos(θ + m) instead of cos(θ) — and scales by `s`.
    Pass the returned logits into nn.functional.cross_entropy.

    Args:
        embeddings: (B, D) — raw embeddings (will be normalized here)
        weight:     (n_classes, D) — classifier weights (will be normalized here)
        labels:     (B,) long — true class indices
        margin:     angular margin in radians (~0.3 = 17°, ArcFace paper uses 0.5)
        scale:      logit scale; bigger → sharper softmax

    Returns logits of shape (B, n_classes).
    """
    import math
    cos_m = math.cos(margin)
    sin_m = math.sin(margin)

    emb_n = F.normalize(embeddings, dim=-1)
    w_n = F.normalize(weight, dim=-1)
    cos_theta = emb_n @ w_n.T  # (B, n_classes), each in [-1, 1]
    cos_theta = cos_theta.clamp(-1 + 1e-7, 1 - 1e-7)
    sin_theta = torch.sqrt(1.0 - cos_theta * cos_theta)
    # cos(θ + m) = cos·cos_m − sin·sin_m
    cos_theta_m = cos_theta * cos_m - sin_theta * sin_m

    # Numerical guard: when θ + m > π, fall back to cos(θ) − m·sin(m) so the
    # gradient stays well-defined (ArcFace "easy margin" trick).
    th = math.cos(math.pi - margin)
    mm = math.sin(math.pi - margin) * margin
    cos_theta_m = torch.where(cos_theta > th, cos_theta_m, cos_theta - mm)

    one_hot = torch.zeros_like(cos_theta).scatter_(1, labels.unsqueeze(1), 1.0)
    logits = one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta
    return logits * scale

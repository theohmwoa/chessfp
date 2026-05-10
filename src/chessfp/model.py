"""Style model: CNN(board+move) -> Transformer(game) -> game embedding.

Following the McIlroy-Young et al. 2021 chess-stylometry recipe.

Forward signature for the full StyleModel:
    x:    (B, T, 24, 8, 8) float — per-game padded sequence of (board, move) channels
    mask: (B, T)            bool  — True where padded
    -->   (B, D)            float — L2-normalized game embedding (when .encode used)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encode import N_INPUT_CHANNELS


def _gn(ch: int) -> nn.GroupNorm:
    # 32 channels per group works well; clamp to ch when ch is small.
    return nn.GroupNorm(num_groups=min(32, ch), num_channels=ch)


class _ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.n1 = _gn(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.n2 = _gn(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.n1(self.c1(x)), inplace=True)
        h = self.n2(self.c2(h))
        return F.relu(x + h, inplace=True)


class BoardMoveEncoder(nn.Module):
    """ResNet-style CNN: (B, 24, 8, 8) -> (B, d_model).

    Spatial pool: flatten(1) + Linear, NOT global average. The 8x8 board is
    small and every square matters — averaging it away erases the per-square
    structure that distinguishes players. The price is one biggish linear
    layer (ch*64 -> d_model).
    """

    def __init__(self, in_ch: int = N_INPUT_CHANNELS, ch: int = 128, n_blocks: int = 4, d_model: int = 256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, ch, 3, padding=1, bias=False),
            _gn(ch),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[_ResBlock(ch) for _ in range(n_blocks)])
        self.proj = nn.Linear(ch * 64, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(self.stem(x))
        h = h.flatten(1)         # (B, ch*64) — preserve spatial info
        return self.proj(h)


class GameEncoder(nn.Module):
    """Transformer over per-game decision embeddings, masked-mean-pooled.

    Mean pool is more robust than [CLS] pooling at init: a learnable [CLS]
    starts near zero, and L2-normalizing the resulting near-zero output causes
    every game to map to roughly the same direction — embedding collapse.
    Masked mean pool inherits the diversity of the input move embeddings
    directly.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        max_len: int = 128,
        ffn_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_len = max_len
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.tf = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, T, D); mask: (B, T) bool — True where padded
        B, T, _ = x.shape
        if T > self.max_len:
            raise ValueError(f"sequence length {T} exceeds max_len {self.max_len}")
        h = x + self.pos[:, :T]
        h = self.tf(h, src_key_padding_mask=mask)
        if mask is None:
            pooled = h.mean(dim=1)
        else:
            keep = (~mask).float().unsqueeze(-1)        # (B, T, 1)
            pooled = (h * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1)
        return self.norm(pooled)


class StyleModel(nn.Module):
    """Full pipeline: per-game (B, T, 24, 8, 8) -> per-game (B, D) embedding.

    If `n_classes` is supplied a linear classification head is added; train with
    cross-entropy on player labels. At inference, drop the head and use the
    L2-normalized embedding for cosine similarity.
    """

    def __init__(
        self,
        d_model: int = 256,
        cnn_ch: int = 128,
        cnn_blocks: int = 4,
        n_heads: int = 4,
        n_layers: int = 4,
        max_len: int = 128,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        n_classes: int | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_classes = n_classes
        self.move_enc = BoardMoveEncoder(ch=cnn_ch, n_blocks=cnn_blocks, d_model=d_model)
        self.game_enc = GameEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            max_len=max_len,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        self.classifier = nn.Linear(d_model, n_classes) if n_classes else None

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C, H, W = x.shape
        flat = x.reshape(B * T, C, H, W)
        if mask is not None and mask.any():
            # Only push real positions through the CNN; padded slots get zero
            # embeddings (the transformer mask hides them anyway). This avoids
            # wasted compute and keeps any stats out of contention.
            keep = ~mask.reshape(B * T)
            valid = self.move_enc(flat[keep])
            full = torch.zeros(B * T, valid.shape[-1], device=x.device, dtype=valid.dtype)
            full[keep] = valid
            moves = full.reshape(B, T, -1)
        else:
            moves = self.move_enc(flat).reshape(B, T, -1)
        return self.game_enc(moves, mask=mask)

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward + L2-normalize for cosine similarity."""
        return F.normalize(self.forward(x, mask), dim=-1)

    def classify(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (raw_embedding, class_logits)."""
        if self.classifier is None:
            raise RuntimeError("StyleModel was built without a classifier head")
        emb = self.forward(x, mask)
        return emb, self.classifier(emb)

    @staticmethod
    def num_parameters(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

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


class _ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.n1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.n2 = nn.BatchNorm2d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.n1(self.c1(x)), inplace=True)
        h = self.n2(self.c2(h))
        return F.relu(x + h, inplace=True)


class BoardMoveEncoder(nn.Module):
    """ResNet-style CNN: (B, 24, 8, 8) -> (B, d_model)."""

    def __init__(self, in_ch: int = N_INPUT_CHANNELS, ch: int = 128, n_blocks: int = 4, d_model: int = 256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[_ResBlock(ch) for _ in range(n_blocks)])
        self.proj = nn.Linear(ch, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(self.stem(x))
        h = h.mean(dim=(2, 3))   # global avg pool
        return self.proj(h)


class GameEncoder(nn.Module):
    """Transformer over per-game decision embeddings, [CLS]-pooled."""

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
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, max_len + 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)
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
        # x: (B, T, D); mask: (B, T) bool, True where padded
        B, T, _ = x.shape
        if T > self.max_len:
            raise ValueError(f"sequence length {T} exceeds max_len {self.max_len}")
        cls = self.cls.expand(B, -1, -1)
        h = torch.cat([cls, x], dim=1) + self.pos[:, : T + 1]
        if mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=mask.device)
            mask = torch.cat([cls_mask, mask], dim=1)
        h = self.tf(h, src_key_padding_mask=mask)
        return self.norm(h[:, 0])


class StyleModel(nn.Module):
    """Full pipeline: per-game (B, T, 24, 8, 8) -> per-game (B, D) embedding."""

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
    ):
        super().__init__()
        self.d_model = d_model
        self.move_enc = BoardMoveEncoder(ch=cnn_ch, n_blocks=cnn_blocks, d_model=d_model)
        self.game_enc = GameEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            max_len=max_len,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C, H, W = x.shape
        moves = self.move_enc(x.reshape(B * T, C, H, W)).reshape(B, T, -1)
        return self.game_enc(moves, mask=mask)

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward + L2-normalize for cosine similarity."""
        return F.normalize(self.forward(x, mask), dim=-1)

    @staticmethod
    def num_parameters(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

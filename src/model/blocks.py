from __future__ import annotations

import torch
import torch.nn as nn

from .attention import CrossAttention, SelfAttention


class FFN(nn.Module):
    def __init__(self, emb_dim: int = 512, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * expansion, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        emb_dim: int = 768,
        num_head: int = 8,
        dropout: float = 0.0,
        forward_expansion: int = 4,
        forward_drop: float = 0.0,
        use_cross: bool = False,
    ):
        super().__init__()
        self.use_cross = use_cross
        self.mod_table = nn.Parameter(torch.randn(1, 6, emb_dim) / emb_dim**0.5)
        self.norm1 = nn.LayerNorm(emb_dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(emb_dim, elementwise_affine=False, eps=1e-6)

        if use_cross:
            self.norm_g = nn.LayerNorm(emb_dim)
            self.norm_cond = nn.LayerNorm(emb_dim)
            self.cross_atten = CrossAttention(emb_dim, num_head, dropout)
        else:
            self.norm_g = None
            self.norm_cond = None
            self.cross_atten = None

        self.self_atten = nn.Sequential(
            SelfAttention(emb_dim, num_head, dropout),
            nn.Dropout(dropout),
        )
        self.ffn = nn.Sequential(
            FFN(emb_dim, forward_expansion, forward_drop),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, n_tok: int = 0) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = (
            self.mod_table + time_emb
        ).chunk(6, dim=1)
        garment = None
        if self.use_cross:
            x, garment = x[:, :n_tok], x[:, n_tok:]

        hidden = self.norm1(x) * (1 + scale1) + shift1
        x = x + gate1 * self.self_atten(hidden)
        if self.use_cross:
            x = x + self.cross_atten(self.norm_cond(x), self.norm_g(garment))

        hidden = self.norm2(x) * (1 + scale2) + shift2
        x = x + gate2 * self.ffn(hidden)
        if self.use_cross:
            x = torch.cat([x, garment], dim=1)
        return x


class FinalLayer(nn.Module):
    def __init__(self, emb_dim: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(emb_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(emb_dim, patch_size * patch_size * out_channels)
        self.mod_table = nn.Parameter(torch.randn(1, 2, emb_dim) / emb_dim**0.5)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        shift, scale = (self.mod_table + time_emb).chunk(2, dim=1)
        return self.linear(self.norm(x) * (1 + scale) + shift)

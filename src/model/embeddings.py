from __future__ import annotations

import math

import torch
import torch.nn as nn


class TimestepEmbedder(nn.Module):
    def __init__(self, emb_dim: int = 768, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.layer = nn.Sequential(
            nn.Linear(freq_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device) / half
        )
        args = t.float()[:, None] * freqs[None] * 1000.0
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.layer(emb)


def sincos_2d(dim: int, height: int, width: int) -> torch.Tensor:
    if dim % 4 != 0:
        raise ValueError("The embedding dimension must be divisible by four.")
    omega = 1.0 / 10000 ** (
        torch.arange(dim // 4, dtype=torch.float32) / (dim // 4)
    )
    grid_y, grid_x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    out = []
    for grid in (grid_y, grid_x):
        values = grid.flatten()[:, None] * omega[None]
        out.extend([torch.sin(values), torch.cos(values)])
    return torch.cat(out, dim=1)

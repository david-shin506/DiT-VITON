from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

from .blocks import FinalLayer, TransformerBlock
from .embeddings import TimestepEmbedder, sincos_2d


class DiT(nn.Module):
    def __init__(
        self,
        latent_size: tuple[int, int],
        in_channels: int = 4,
        emb_dim: int = 768,
        depth: int = 12,
        num_head: int = 8,
        drop_p: float = 0.0,
        forward_expansion: int = 4,
        patch_size: int = 2,
        forward_drop_p: float = 0.0,
        cross_end: int = -1,
    ):
        super().__init__()

        height, width = latent_size
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError(
                f"latent_size {latent_size} must be divisible by "
                f"patch_size={patch_size}."
            )

        self.cross_end = cross_end
        self.in_channels = in_channels * 2 + 1
        self.g_channels = in_channels
        self.patch = patch_size
        self.H, self.W = latent_size
        self.h, self.w = self.H // patch_size, self.W // patch_size
        self.emb_dim = emb_dim

        self.x_embed = nn.Conv2d(self.in_channels, emb_dim, patch_size, patch_size)
        self.g_embed = (
            nn.Conv2d(self.g_channels, emb_dim, patch_size, patch_size)
            if cross_end >= 0
            else None
        )

        pos = sincos_2d(emb_dim, self.h, self.w)
        self.stream_embed = nn.Parameter(torch.randn(2, emb_dim) * 0.02)
        self.register_buffer("pos_x", pos[None], persistent=False)
        self.register_buffer("pos_g", pos[None], persistent=False)

        self.t_embed = TimestepEmbedder(emb_dim)
        self.global_mod = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, 6 * emb_dim))
        nn.init.zeros_(self.global_mod[1].weight)
        nn.init.zeros_(self.global_mod[1].bias)

        self.encoder = nn.ModuleList(
            [
                TransformerBlock(
                    emb_dim,
                    num_head,
                    drop_p,
                    forward_expansion,
                    forward_drop_p,
                    i <= cross_end,
                )
                for i in range(depth)
            ]
        )
        self.final = FinalLayer(emb_dim, patch_size, in_channels)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(
            x,
            "b (h w) (p q c) -> b c (h p) (w q)",
            h=self.h,
            w=self.w,
            p=self.patch,
            q=self.patch,
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        agnostic: torch.Tensor,
        mask: torch.Tensor,
        garment: torch.Tensor | None = None,
    ) -> torch.Tensor:
        
        if x.shape[-2:] != (self.H, self.W):
            raise ValueError(
                f"Expected latent spatial size {(self.H, self.W)}, "
                f"but got {tuple(x.shape[-2:])}."
            ) 
        if self.cross_end >= 0 and garment is None:
            raise ValueError("A garment latent is required when cross attention is enabled.")
        if self.cross_end < 0 and garment is not None:
            raise ValueError("Set model.cross_attention_end >= 0 to use a garment latent.")

        n_tokens = self.h * self.w
        x = self.x_embed(torch.cat([x, agnostic, mask], dim=1))
        x = rearrange(x, "b d h w -> b (h w) d") + self.pos_x + self.stream_embed[0]
        if garment is not None:
            garment_tokens = self.g_embed(garment)
            garment_tokens = (
                rearrange(garment_tokens, "b d h w -> b (h w) d")
                + self.pos_g
                + self.stream_embed[1]
            )
            x = torch.cat([x, garment_tokens], dim=1)

        time_embedding = self.t_embed(t)
        time_modulation = rearrange(
            self.global_mod(time_embedding), "b (k e) -> b k e", k=6
        )
        for i, block in enumerate(self.encoder):
            x = block(
                x,
                time_modulation,
                n_tokens if i <= self.cross_end else x.size(1),
            )
        x = x[:, :n_tokens]
        return self.unpatchify(self.final(x, time_modulation[:, :2]))

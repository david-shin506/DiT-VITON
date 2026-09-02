from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SelfAttention(nn.Module):
    def __init__(self, emb_dim: int = 512, num_head: int = 8, dropout: float = 0.0):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_head = num_head
        self.head_dim = emb_dim // num_head
        self.low_rank = emb_dim // 2

        self.qk = nn.Linear(emb_dim, self.low_rank * 2)
        self.v = nn.Linear(emb_dim, emb_dim)
        self.dropout = dropout
        self.proj = nn.Linear(emb_dim, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qk, value = self.qk(x), self.v(x)
        qk = rearrange(qk, "b n (k h d) -> k b h n d", k=2, h=self.num_head)
        value = rearrange(value, "b n (h d) -> b h n d", h=self.num_head)
        query, key = qk[0], qk[1]
        out = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.proj(rearrange(out, "b h n d -> b n (h d)"))


class CrossAttention(nn.Module):
    def __init__(self, emb_dim: int = 512, num_head: int = 8, dropout: float = 0.0):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_head = num_head
        self.head_dim = emb_dim // num_head
        self.low_rank = emb_dim // 2

        self.q = nn.Linear(emb_dim, self.low_rank)
        self.k = nn.Linear(emb_dim, self.low_rank)
        self.v = nn.Linear(emb_dim, emb_dim)
        self.dropout = dropout
        self.proj = nn.Linear(emb_dim, emb_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        query, key, value = self.q(x), self.k(y), self.v(y)
        query = rearrange(query, "b n (h d) -> b h n d", h=self.num_head)
        key = rearrange(key, "b m (h d) -> b h m d", h=self.num_head)
        value = rearrange(value, "b m (h d) -> b h m d", h=self.num_head)
        out = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.proj(rearrange(out, "b h n d -> b n (h d)"))

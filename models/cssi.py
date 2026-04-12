from __future__ import annotations

import torch
import torch.nn as nn


class CSSIBlock(nn.Module):
    """Cross-Scale Spatial-Semantic Interaction module."""

    def __init__(
        self,
        semantic_dim: int,
        structural_dim: int,
        attn_dim: int = 256,
        num_heads: int = 8,
        kv_downsample: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if attn_dim % num_heads != 0:
            raise ValueError("`attn_dim` must be divisible by `num_heads`.")

        self.query_proj = nn.Conv2d(semantic_dim, attn_dim, kernel_size=1, bias=False)
        self.key_proj = nn.Conv2d(structural_dim, attn_dim, kernel_size=1, bias=False)
        self.value_proj = nn.Conv2d(structural_dim, attn_dim, kernel_size=1, bias=False)

        if kv_downsample > 1:
            self.kv_downsample = nn.AvgPool2d(kernel_size=kv_downsample, stride=kv_downsample)
        else:
            self.kv_downsample = nn.Identity()

        self.query_norm = nn.LayerNorm(attn_dim)
        self.key_norm = nn.LayerNorm(attn_dim)
        self.value_norm = nn.LayerNorm(attn_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out_proj = nn.Conv2d(attn_dim, semantic_dim, kernel_size=1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, semantic: torch.Tensor, structural: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = semantic.shape

        query = self.query_proj(semantic).flatten(2).transpose(1, 2)

        structural = self.kv_downsample(structural)
        key = self.key_proj(structural).flatten(2).transpose(1, 2)
        value = self.value_proj(structural).flatten(2).transpose(1, 2)

        attn_out, _ = self.cross_attn(
            query=self.query_norm(query),
            key=self.key_norm(key),
            value=self.value_norm(value),
        )
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, -1, height, width)
        attn_out = self.out_proj(attn_out)
        return semantic + self.gamma * attn_out

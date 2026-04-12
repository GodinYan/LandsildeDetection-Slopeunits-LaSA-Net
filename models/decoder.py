from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBNAct


class SemanticScaleTransform(nn.Module):
    """Project and resize ViT features to a target pyramid scale."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = ConvBNAct(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        x = self.proj(x)
        if x.shape[-2:] != target_size:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return x


class HRMSDDecoder(nn.Module):
    """High-Resolution Multi-Scale Decoder described in LaSA-Net."""

    def __init__(
        self,
        semantic_dim: int,
        structural_dims: list[int],
        decoder_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if len(structural_dims) != 4:
            raise ValueError("HRMSDDecoder expects four structural pyramid levels.")

        self.semantic_transforms = nn.ModuleList(
            [SemanticScaleTransform(semantic_dim, decoder_dim) for _ in range(4)]
        )
        self.structural_projections = nn.ModuleList(
            [ConvBNAct(channels, decoder_dim, kernel_size=1, padding=0) for channels in structural_dims]
        )
        self.level_fusions = nn.ModuleList(
            [ConvBNAct(decoder_dim * 2, decoder_dim, kernel_size=3) for _ in range(4)]
        )
        self.top_down_merges = nn.ModuleList(
            [ConvBNAct(decoder_dim * 2, decoder_dim, kernel_size=3) for _ in range(3)]
        )
        self.level_refines = nn.ModuleList(
            [ConvBNAct(decoder_dim, decoder_dim, kernel_size=3) for _ in range(4)]
        )
        self.multi_scale_fusion = nn.Sequential(
            ConvBNAct(decoder_dim * 4, decoder_dim, kernel_size=3),
            nn.Dropout2d(dropout),
        )
        self.seg_head = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)

    def forward(
        self,
        semantic_features: list[torch.Tensor],
        structural_features: list[torch.Tensor],
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        if len(semantic_features) != 4 or len(structural_features) != 4:
            raise ValueError("HRMSDDecoder expects exactly four semantic and four structural features.")

        fused_levels = []
        for index in range(4):
            structural = self.structural_projections[index](structural_features[index])
            semantic = self.semantic_transforms[index](semantic_features[index], structural.shape[-2:])
            fused = self.level_fusions[index](torch.cat([semantic, structural], dim=1))
            fused_levels.append(fused)

        decoded_features = [None] * 4
        x = self.level_refines[3](fused_levels[3])
        decoded_features[3] = x

        for level in range(2, -1, -1):
            x = F.interpolate(x, size=fused_levels[level].shape[-2:], mode="bilinear", align_corners=False)
            x = self.top_down_merges[level](torch.cat([fused_levels[level], x], dim=1))
            x = self.level_refines[level](x)
            decoded_features[level] = x

        base_size = decoded_features[0].shape[-2:]
        pyramid = [decoded_features[0]]
        for level in range(1, 4):
            pyramid.append(
                F.interpolate(decoded_features[level], size=base_size, mode="bilinear", align_corners=False)
            )

        logits = self.seg_head(self.multi_scale_fusion(torch.cat(pyramid, dim=1)))
        if output_size is not None and logits.shape[-2:] != output_size:
            logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        return logits
